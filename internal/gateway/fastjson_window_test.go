package gateway

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

// watch counts jsonValid's checks and validHash's sums of the slice at the
// same place as win, whatever else is decoded.
func watch(t *testing.T, win []byte) (scans, sums *atomic.Int32) {
	scans, sums = &atomic.Int32{}, &atomic.Int32{}
	p, n := uintptr(unsafe.Pointer(unsafe.SliceData(win))), len(win)
	sf := func(data []byte) {
		if uintptr(unsafe.Pointer(unsafe.SliceData(data))) == p && len(data) == n {
			scans.Add(1)
		}
	}
	hf := func(data []byte) {
		if uintptr(unsafe.Pointer(unsafe.SliceData(data))) == p && len(data) == n {
			sums.Add(1)
		}
	}
	validScanned.Store(&sf)
	validHashed.Store(&hf)
	t.Cleanup(func() { validScanned.Store(nil); validHashed.Store(nil) })
	return scans, sums
}

// A window of a held body — an image's block, a long message — is checked
// and split once while the request is served, not by each of its readers
// (#4836): a 20 MB request with 8 images checked its 2.5 MB blocks 59
// times, 315 MB in all, and summed them 120 times.
func TestHeldBodyWindowIsCheckedOnce(t *testing.T) {
	body := memoBody(t.Name())
	ctx, done := context.WithCancel(context.Background())
	holdBody(ctx, body)

	var q struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := decodeJSON(body, &q); err != nil || len(q.Messages) != 1 {
		t.Fatalf("decode: %v, %d messages", err, len(q.Messages))
	}
	win := q.Messages[0]
	if !sharesBody(win, body) || len(win) < validMemoMin {
		t.Fatalf("the message is not a long window of the body (%d bytes, shared %v)", len(win), sharesBody(win, body))
	}

	scans, sums := watch(t, win)
	// warmed once: what the request's readers do after it is what must
	// not check it again
	if valid, _ := longBody(win); !valid {
		t.Fatal("a window of a held body read as not valid")
	}
	warm := scans.Load()
	for range 6 {
		var m struct {
			Content string `json:"content"`
		}
		if err := decodeJSON(win, &m); err != nil || len(m.Content) == 0 {
			t.Fatalf("decode of a window: %v, %d bytes", err, len(m.Content))
		}
		// the body whole, in between: its own entry, not the window's
		var all struct {
			Model string `json:"model"`
		}
		if err := decodeJSON(body, &all); err != nil || all.Model != t.Name() {
			t.Fatalf("decode of the body: %v %q", err, all.Model)
		}
	}
	if n, want := scans.Load(), warm; n != want {
		t.Fatalf("a window of a held body was checked %d times over six readers, want %d", n-want, 0)
	}
	if n := sums.Load(); n != 0 {
		t.Fatalf("a window of a held body was summed %d times, want 0", n)
	}

	// the finding is the window's own: another window of the body, which
	// is not valid JSON, doesn't read as this one did
	off := int(uintptr(unsafe.Pointer(unsafe.SliceData(win))) - uintptr(unsafe.Pointer(unsafe.SliceData(body))))
	bad := body[off : off+len(win)+1]
	if jsonValid(bad) {
		t.Fatal("a window with the array's ] after it read as valid")
	}

	// released, the window is summed again, as any other body's
	done()
	for deadline := time.Now().Add(10 * time.Second); ; {
		heldBodies.Lock()
		_, kept := heldBodies.long[unsafe.SliceData(body)]
		heldBodies.Unlock()
		if !kept {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("a released body's finds were kept")
		}
		time.Sleep(time.Millisecond)
	}
	if !jsonValid(win) || sums.Load() == 0 {
		t.Fatal("a released body's window was not summed again")
	}
}

// A window that isn't JSON is said to be not valid while the body is
// held, as before.
func TestHeldBodyWindowNotJSON(t *testing.T) {
	body := append(memoBody(t.Name()), '}')
	ctx, done := context.WithCancel(context.Background())
	defer done()
	holdBody(ctx, body)
	if jsonValid(body) {
		t.Fatal("a held body with a stray } read as valid")
	}
	// a window of it, of a long body that isn't JSON either
	win := body[1 : len(body)-2]
	if len(win) < validMemoMin || jsonValid(win) {
		t.Fatalf("a window of a body that isn't JSON read as valid (%d bytes)", len(win))
	}
	for range 2 {
		if jsonValid(win) {
			t.Fatal("a window of a held body with a stray } read as valid")
		}
	}
}
