package gateway

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
)

// A body with an image swapped in it is the bytes the client sent, around
// the image: a rewrite through a map ran every key of every turn through
// json.Marshal, which sorts them, so the provider saw a conversation in an
// order its cache had never seen (#4836).
func TestImageSwapKeepsTheRestOfTheBody(t *testing.T) {
	b64 := heicFixture(t, "opaque.heic")
	block := `{"type":"image_url","image_url":"data:image/heic;base64,` + b64 + `"}`
	// keys in no order a marshal would write them, and a space only the
	// client wrote
	body := []byte(`{"model":"fake/m1","zz":1,"messages":[{"role":"user","zz":2,"content":[` +
		block + `,{"text":"what is this?","type":"text"}]}]}`)
	out := heicAsJPEG(context.Background(), provider.Chat, body)
	if bytes.Equal(out, body) {
		t.Fatal("the HEIC was not swapped")
	}
	at := bytes.Index(body, []byte(block))
	if at < 0 {
		t.Fatal("the test's own block is not in its body")
	}
	if !bytes.HasPrefix(out, body[:at]) {
		t.Fatalf("the body before the image was rewritten:\n%q\n%q", body[:at], out[:min(len(out), at)])
	}
	if !bytes.HasSuffix(out, body[at+len(block):]) {
		t.Fatalf("the body after the image was rewritten:\n%q\n%q", body[at+len(block):], out[max(0, len(out)-(len(body)-at-len(block))):])
	}
	if !bytes.Contains(out, []byte(`{"role":"user","zz":2,"content":`)) {
		t.Fatalf("the turn's keys were reordered: %.200s", out)
	}
}

// A rewritten body is held while its request is served, as the body that
// came in is: each of the request's readers would otherwise copy it and
// check it again from the start.
func TestSwappedImageBodyIsHeld(t *testing.T) {
	body := heicChat(1, 40<<10) // past validMemoMin, where a body is held
	ctx, done := context.WithCancel(context.Background())
	defer done()
	out := heicAsJPEG(ctx, provider.Chat, body)
	if bytes.Equal(out, body) {
		t.Fatal("the HEIC was not swapped")
	}
	if _, _, ok := heldBodyAt(out); !ok {
		t.Fatal("the rewritten body is not held")
	}
	done()
	for deadline := time.Now().Add(10 * time.Second); ; {
		if _, _, ok := heldBodyAt(out); !ok {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("the rewritten body was kept after its request")
		}
		time.Sleep(time.Millisecond)
	}
}
