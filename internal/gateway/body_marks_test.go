package gateway

import (
	"bytes"
	"context"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

// bodyHas is bytes.Contains, said from where a needle that starts with a
// quote can only be, so it must answer exactly as bytes.Contains does.
func TestBodyHasIsContains(t *testing.T) {
	manyQuotes := append(bytes.Repeat([]byte(`"`), quotesMost+1), []byte(`"last"`)...)
	bodies := [][]byte{
		memoBody(t.Name()),
		[]byte(`{"model":"a","messages":[{"role":"user","content":"x"}]}`),
		[]byte(`{"a":"\"compaction_trigger\"","b":""}`),
		bytes.Repeat([]byte(`x`), validMemoMin+1),
		manyQuotes,
		nil,
	}
	needles := []string{
		`"model"`, `"compaction_trigger"`, `"additional_tools"`, `"compaction`,
		`"`, `""`, `"content"`, `"x"`, `"}`, `"nope`, `"last"`,
		`content`, `model`, ``, `x`,
	}
	for _, body := range bodies {
		for _, n := range needles {
			if got, want := bodyHas(body, n), bytes.Contains(body, []byte(n)); got != want {
				t.Fatalf("%d-byte body, needle %q: bodyHas %v, Contains %v", len(body), n, got, want)
			}
		}
	}
}

// A held body's quotes are walked once for every question its request asks
// of it, and dropped with the body: four questions over a 20 MB body were
// four walks of it, 45ms of the request (#4836).
func TestBodyHasWalksQuotesOnce(t *testing.T) {
	body := memoBody(t.Name())
	if len(body) < validMemoMin {
		t.Fatalf("the test's body is not long enough to be kept (%d bytes)", len(body))
	}
	at := uintptr(unsafe.Pointer(unsafe.SliceData(body)))
	var walks atomic.Int32
	w := func(data []byte) {
		if uintptr(unsafe.Pointer(unsafe.SliceData(data))) == at {
			walks.Add(1)
		}
	}
	marksWalked.Store(&w)
	t.Cleanup(func() { marksWalked.Store(nil) })

	ctx, done := context.WithCancel(context.Background())
	defer done()
	holdBody(ctx, body)
	for range 5 {
		if !bodyHas(body, `"model"`) {
			t.Fatal("a held body read as not having its own model")
		}
		if bodyHas(body, `"compaction_trigger"`) {
			t.Fatalf("a held body read as having a compaction trigger")
		}
	}
	if n := walks.Load(); n != 1 {
		t.Fatalf("a held body's quotes were walked %d times for ten questions, want 1", n)
	}

	// released, the offsets go with it: the next question walks the bytes
	// again, as any other body's
	done()
	for deadline := time.Now().Add(10 * time.Second); ; {
		heldBodies.Lock()
		_, kept := heldBodies.quotes[unsafe.SliceData(body)]
		heldBodies.Unlock()
		if !kept {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("a released body's quotes were kept")
		}
		time.Sleep(time.Millisecond)
	}
	if !bodyHas(body, `"model"`) {
		t.Fatal("a released body lost its model")
	}
	if n := walks.Load(); n != 2 {
		t.Fatalf("a released body's quotes were walked %d times, want 2", n)
	}
}
