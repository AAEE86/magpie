package gateway

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// smallChat is a chat request of about size bytes: a system prompt padded
// to it and one turn, as a desktop agent's ordinary request arrives
// (most are a few KB, and every one of them walks the paths these
// benchmarks measure).
func smallChat(size int) []byte {
	type m = map[string]any
	build := func(pad int) []byte {
		b, _ := json.Marshal(m{
			"model":  "fake/m1",
			"stream": false,
			"messages": []any{
				m{"role": "system", "content": strings.Repeat("You are a coding agent. ", pad/25+1)},
				m{"role": "user", "content": "hello"},
			},
		})
		return b
	}
	b := build(0)
	if len(b) < size {
		b = build(size - len(b))
	}
	return b
}

// heicChat is a chat request of images HEIC images about a conversation
// of about pad bytes, spread over its turns as a conversation's long
// context is — not one huge message, which no client sends. The fixtures
// are the 32x24 HEICs of testdata (macOS's own encoder), so what they
// measure is the request's way there — finding the image, and the JSON
// around it — not the decode of a 48 MP phone photo, which no encoder
// here can write.
func heicChat(images, pad int) []byte {
	raw, err := os.ReadFile("testdata/opaque.heic")
	if err != nil {
		panic(err)
	}
	url := "data:image/heic;base64," + base64.StdEncoding.EncodeToString(raw)
	type m = map[string]any
	const turns = 32
	per := 0
	if pad > 0 {
		per = pad / turns
	}
	msgs := []any{m{"role": "system", "content": "You are a coding agent."}}
	for i := range turns {
		text := "go on"
		if per > 0 {
			text = strings.Repeat("The function reads the file and returns the lines. ", per/54+1)
		}
		var content any = text
		if i < images {
			content = []any{m{"type": "image_url", "image_url": m{"url": url}}, m{"type": "text", "text": text}}
		}
		msgs = append(msgs, m{"role": "user", "content": content}, m{"role": "assistant", "content": "ok"})
	}
	b, _ := json.Marshal(m{"model": "fake/m1", "stream": false, "messages": msgs})
	return b
}

// benchGateway is a gateway answering through the fake provider, and the
// handler its routes make once — Handler() builds every route, and
// registering them costs more than a small request does.
func benchGateway(b testing.TB, up provider.Protocol) http.Handler {
	b.Helper()
	b.Setenv("XDG_CONFIG_HOME", b.TempDir())
	b.Setenv("XDG_CACHE_HOME", b.TempDir())
	f := &fake{ctype: "", reply: "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n"}
	up1 := httptest.NewServer(f)
	b.Cleanup(up1.Close)
	p := provider.Provider{ID: "fake", Name: "Fake", Key: "k", Models: []string{"m1"}}
	if up == provider.Responses {
		p.Responses = up1.URL + "/v1"
	} else {
		p.Chat = up1.URL + "/v1"
	}
	if err := provider.Save(p); err != nil {
		b.Fatal(err)
	}
	return New().Handler()
}

// benchServe is one request through the gateway, failed when it doesn't
// answer.
func benchServe(b testing.TB, h http.Handler, path string, body []byte) {
	b.Helper()
	ctx, done := context.WithCancel(context.Background())
	defer done()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", path, bytes.NewReader(body)).WithContext(ctx))
	if rec.Code != 200 {
		b.Fatalf("%d %.300s", rec.Code, rec.Body.String())
	}
}

// BenchmarkSmallRequest is the ordinary request: a few KB of chat that
// every feature reads, and that no memo of a long body covers. The
// gateway's long-body paths (validMemoMin, the held body's windows, its
// quotes) must cost it nothing.
func BenchmarkSmallRequest(b *testing.B) {
	h := benchGateway(b, provider.Chat)
	for _, size := range []int{1 << 10, 4 << 10, 16 << 10, 64 << 10, 256 << 10} {
		b.Run(fmt.Sprintf("%dKB", size>>10), func(b *testing.B) {
			body := smallChat(size)
			b.SetBytes(int64(len(body)))
			b.ResetTimer()
			for range b.N {
				benchServe(b, h, "/v1/chat/completions", body)
			}
			b.ReportMetric(float64(len(body))/1e3, "KB/req")
		})
	}
}

// BenchmarkSmallParse is what the gateway does to a small body with the
// request around it taken off: check it, split it at its members, decode
// it, and ask it the questions features ask. A body below validMemoMin
// takes none of the long-body memos, so this is the path an ordinary 1 KB
// request takes, where the long ones' savings must cost nothing.
func BenchmarkSmallParse(b *testing.B) {
	body := smallChat(1 << 10)
	b.Run("valid", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		for range b.N {
			if !jsonValid(body) {
				b.Fatal("not valid")
			}
		}
	})
	b.Run("longBody", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		for range b.N {
			if valid, _ := longBody(body); !valid {
				b.Fatal("not valid")
			}
		}
	})
	b.Run("decode", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		for range b.N {
			var v struct {
				Model string `json:"model"`
			}
			if err := decodeJSON(body, &v); err != nil || v.Model != "fake/m1" {
				b.Fatalf("%v %q", err, v.Model)
			}
		}
	})
	b.Run("bodyHas", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		for range b.N {
			if bodyHas(body, `"compaction_trigger"`) {
				b.Fatal("found")
			}
		}
	})
	b.Run("contains", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		for range b.N {
			if bytes.Contains(body, []byte(`"compaction_trigger"`)) {
				b.Fatal("found")
			}
		}
	})
}

// BenchmarkHEICRequest is a conversation with HEIC images in it, which
// the gateway finds by scanning the body and converts before any route
// reads it. The fixtures are 32x24, so the conversion is small: what is
// measured is the scan, and the request around it.
func BenchmarkHEICRequest(b *testing.B) {
	h := benchGateway(b, provider.Chat)
	for _, tc := range []struct {
		name          string
		images, padKB int
	}{
		{"one-image", 1, 0},
		{"eight-images", 8, 0},
		{"no-image-2MB-conversation", 0, 2 << 10},
		{"one-image-2MB-conversation", 1, 2 << 10},
	} {
		b.Run(tc.name, func(b *testing.B) {
			body := heicChat(tc.images, tc.padKB<<10)
			b.SetBytes(int64(len(body)))
			b.ResetTimer()
			for range b.N {
				benchServe(b, h, "/v1/chat/completions", body)
			}
			b.ReportMetric(float64(len(body))/1e3, "KB/req")
		})
	}
}
