package gateway

import (
	"bytes"
	"slices"
	"sync/atomic"
	"unsafe"
)

// A feature asks the whole body a question before it parses: does this
// body mention a compaction trigger, a tool to lift, a type it converts?
// Each asked it of its own, with bytes.Contains, and each walked the body
// for it (#4836): four such questions over a 20 MB image request were 45ms
// of its 250ms.
//
// bytes.Contains(body, needle) walks for the needle's first byte, then
// compares at each place it finds one — 2 GB/s measured on that body,
// since a needle like "compaction_trigger" is only found after a quote.
// Nearly every needle a feature asks for starts with a quote, and a
// needle that starts with one can only be found at a quote: find the
// body's quotes once (memchr, 18 GB/s) and any number of needles is
// answered from them, in the same walk.

// quotesMost bounds the quotes kept of one body: a body of plain text has
// hundreds of thousands of them, and keeping those would cost more than
// the walks they save. Past it the body is asked with bytes.Contains, as
// before.
const quotesMost = 1 << 16

// heldQuotes are a held body's quote offsets (quotesOf).
type heldQuotes struct {
	n    int
	offs []int32
	many bool // more than quotesMost: not kept
}

// bodyHas reports whether body contains needle, as bytes.Contains does.
// A needle that doesn't start with a quote — an internal one — is asked of
// bytes.Contains, since only a quote-headed one is found at a quote.
func bodyHas(body []byte, needle string) bool {
	if len(needle) == 0 || needle[0] != '"' || len(body) < validMemoMin {
		return bytes.Contains(body, []byte(needle))
	}
	offs, many := quotesOf(body)
	if many {
		return bytes.Contains(body, []byte(needle))
	}
	for _, off := range offs {
		if i := int(off); len(body)-i >= len(needle) && string(body[i:i+len(needle)]) == needle {
			return true
		}
	}
	return false
}

// quotesOf is the offsets of body's quote bytes, kept while body is held
// (holdBody): a body is not written while held, so the same quotes answer
// every question its request asks of it.
func quotesOf(body []byte) ([]int32, bool) {
	at := unsafe.SliceData(body)
	heldBodies.Lock()
	e, ok := heldBodies.quotes[at]
	heldBodies.Unlock()
	if ok && e.n == len(body) {
		return e.offs, e.many
	}
	offs, many := quoteOffsets(body)
	heldBodies.Lock()
	defer heldBodies.Unlock()
	// only while body is held: one released may be freed, and its memory
	// given to another body that would read as this one (factsOf)
	if l := heldBodies.list.Load(); l != nil && slices.ContainsFunc(*l, func(b []byte) bool {
		return unsafe.SliceData(b) == at && len(b) == len(body)
	}) {
		if heldBodies.quotes == nil {
			heldBodies.quotes = map[*byte]heldQuotes{}
		}
		heldBodies.quotes[at] = heldQuotes{n: len(body), offs: offs, many: many}
	}
	return offs, many
}

// quoteOffsets is the offsets of body's quote bytes, or many set when there
// are more than quotesMost of them.
func quoteOffsets(body []byte) (offs []int32, many bool) {
	if len(body) > 1<<31-1 {
		return nil, true // past what an int32 holds, and past any body served
	}
	if f := marksWalked.Load(); f != nil {
		(*f)(body)
	}
	offs = make([]int32, 0, 64)
	for i := 0; i < len(body); {
		j := bytes.IndexByte(body[i:], '"')
		if j < 0 {
			break
		}
		i += j
		if len(offs) == quotesMost {
			return nil, true
		}
		offs = append(offs, int32(i))
		i++
	}
	return offs, false
}

// marksWalked, when set (tests), is told of each body whose quotes are
// walked for bodyHas.
var marksWalked atomic.Pointer[func(body []byte)]
