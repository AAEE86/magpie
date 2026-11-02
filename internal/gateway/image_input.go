package gateway

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"

	"github.com/yetone/magpie/internal/provider"
)

const imageOmitted = "[Image omitted: this model accepts text only.]"

// errCurrentImage stops a walk at an image the latest turn gives the model.
var errCurrentImage = errors.New("an image in the latest turn")

// textOnlyBody rejects a user image in the latest turn and omits tool images
// and images in older turns. It edits the wire JSON so passthrough keeps fields
// the IR does not represent (such as provider-specific request options).
func textOnlyBody(proto provider.Protocol, body []byte) ([]byte, bool) {
	out, err := walkImages(proto, body, func(im imageAt) (json.RawMessage, error) {
		if im.Current() {
			return nil, errCurrentImage
		}
		return textBlock(proto, imageOmitted), nil
	})
	if err != nil {
		return body, true
	}
	return out, false
}

// hasImage is whether the body has an image in any turn.
func hasImage(proto provider.Protocol, body []byte) bool {
	found := errors.New("found")
	_, err := walkImages(proto, body, func(imageAt) (json.RawMessage, error) { return nil, found })
	return err == found
}

// hasUnportableCurrentImage reports a current Responses image with no URL to
// carry to another protocol. A file_id belongs to the Responses provider that
// stored it; translating it as a URL or dropping it would mislead the model.
func hasUnportableCurrentImage(body []byte) bool {
	missing := errors.New("current image has no portable source")
	_, err := walkImages(provider.Responses, body, func(im imageAt) (json.RawMessage, error) {
		if im.Current() && im.Src == "" {
			return nil, missing
		}
		return nil, nil
	})
	return err == missing
}

// imageAt is an image in a request, where it is.
type imageAt struct {
	// Src is the image as a URL, a data: URL for one sent inline; empty for
	// one only its vendor can open (a Responses file_id, a Gemini file).
	Src string
	// Last is set on one in the latest turn, Tool on one a tool gave back.
	Last, Tool bool
	// Block is the image's block as it is in the body.
	Block json.RawMessage
}

// Current is whether the image is one the user gives the model now, which
// the model is asked about.
func (im imageAt) Current() bool { return im.Last && !im.Tool }

// walkImages gives every image in the body's turns to swap, and puts the
// block it returns in the image's place (nil keeps it). An error from swap
// ends the walk. The body is returned as it was when nothing was swapped.
func walkImages(proto provider.Protocol, body []byte, swap func(imageAt) (json.RawMessage, error)) ([]byte, error) {
	turnsKey, contentKey := "messages", "content"
	switch proto {
	case provider.Responses:
		turnsKey = "input"
	case provider.Gemini:
		turnsKey, contentKey = "contents", "parts"
	}
	if !anyImageTurn(proto, body, turnsKey, contentKey) {
		// Most requests have no image, and this walk runs several times
		// on each: decoding every turn and block cost ~2.6 MB for each
		// 1.5 MB conversation (yg6888 on X), and the request's top level
		// alone was a copy of it each time (#3725).
		return body, nil
	}
	var request map[string]json.RawMessage
	if decodeJSON(body, &request) != nil {
		return body, nil
	}
	var turns []json.RawMessage
	if decodeJSON(request[turnsKey], &turns) != nil {
		return body, nil
	}
	// where each turn is in the body's own bytes, to write a swapped
	// block in its place rather than write the whole body again: a 1.6 KB
	// HEIC in a 2 MB conversation cost 31ms, two thirds of it marshalling
	// the body around it (#4836)
	spans, spliceable := turnSpans(body, turns)
	at := &swapAt{ok: spliceable}
	changed := false
	for i, raw := range turns {
		if !anyImageInTurn(proto, raw, contentKey) {
			// most turns are text and tool output, and each was decoded
			// to find that out: on a 2 MB conversation the walk was 13ms,
			// nine of them decoding turns with no image in them
			continue
		}
		var turn map[string]json.RawMessage
		if decodeJSON(raw, &turn) != nil {
			continue
		}
		key := contentKey
		tool := false
		var kind string
		if proto == provider.Responses {
			decodeJSON(turn["type"], &kind)
			if kind == "function_call_output" {
				key, tool = "output", true
			}
		}
		if proto == provider.Chat {
			decodeJSON(turn["role"], &kind)
			tool = kind == "tool"
		}
		if spliceable {
			at.loc = &locator{body: body, from: spans[i][0], to: spans[i][1]}
		} else {
			at.loc = nil
		}
		content, swapped, err := swapImageBlocks(proto, turn[key], i == len(turns)-1, tool, swap, at)
		if err != nil {
			return body, err
		}
		if !swapped {
			continue
		}
		turn[key] = content
		turns[i], _ = json.Marshal(turn)
		changed = true
	}
	if !changed {
		return body, nil
	}
	if at.ok {
		return spliceEdits(body, at.edits), nil
	}
	request[turnsKey], _ = json.Marshal(turns)
	return json.Marshal(request)
}

// imageEdit is a block of the body replaced by another: what a walk
// swapped, and where in the body's own bytes it was.
type imageEdit struct {
	at, end int
	out     json.RawMessage
}

// swapAt is where a walk's swaps go. When every block it swapped was
// found in the body as it came, they are spliced in and the rest of the
// body is left the bytes it arrived with, in the order it arrived in; when
// one wasn't (a value escaped or spaced differently from what the decode
// reads), ok is false and the body is written again as before.
type swapAt struct {
	loc   *locator
	edits []imageEdit
	ok    bool
}

// locator is where a turn's blocks are, by the turn's place in the body.
// Its search goes only forward: a turn's blocks are visited in the order
// the turn holds them, so the next one is after the last.
type locator struct {
	body     []byte
	from, to int
}

// find is where raw is in the turn, and false when it is not there as it
// came.
func (l *locator) find(raw []byte) (int, bool) {
	if len(raw) == 0 || l.from >= l.to {
		return 0, false
	}
	j := bytes.Index(l.body[l.from:l.to], raw)
	if j < 0 {
		return 0, false
	}
	i := l.from + j
	l.from = i + len(raw)
	return i, true
}

// spliceEdits is body with each edit's block in its place.
func spliceEdits(body []byte, edits []imageEdit) []byte {
	n := len(body)
	for _, e := range edits {
		n += len(e.out) - (e.end - e.at)
	}
	out := make([]byte, 0, n)
	last := 0
	for _, e := range edits {
		out = append(out, body[last:e.at]...)
		out = append(out, e.out...)
		last = e.end
	}
	return append(out, body[last:]...)
}

// turnSpans is where each of body's turns is, and false when one of them
// is not found in it as it came.
func turnSpans(body []byte, turns []json.RawMessage) ([][2]int, bool) {
	spans := make([][2]int, len(turns))
	from := 0
	for i, raw := range turns {
		if len(raw) == 0 {
			return nil, false
		}
		j := bytes.Index(body[from:], raw)
		if j < 0 {
			return nil, false
		}
		spans[i] = [2]int{from + j, from + j + len(raw)}
		from += j + len(raw)
	}
	return spans, true
}

// anyImageTurn is whether body may hold a block walkImages gives to swap:
// it is a JSON object whose turnsKey is a list anyImageBlock says yes to,
// read in place. A key given twice is looked at each time.
func anyImageTurn(proto provider.Protocol, body []byte, turnsKey, contentKey string) bool {
	if !jsonValid(body) {
		return false
	}
	raw := trimJSON(body)
	if raw[0] != '{' {
		return false
	}
	found := false
	eachMember(raw, func(key, val []byte) bool {
		if val[0] == '[' && jsonStringIs(key, turnsKey) {
			found = anyImageBlock(proto, val, contentKey)
		}
		return !found
	})
	return found
}

// anyImageBlock is whether turns, a valid JSON array, may hold a block
// walkImages gives to swap, read in place without decoding: a turn's
// contentKey (or a Responses function_call_output's output) that is a list
// with a block among it of an image's type, or one inside an Anthropic
// tool_result's content or a Gemini functionResponse's parts, or a Gemini
// part with inlineData or fileData. It errs toward yes, as for a key
// spelled twice.
func anyImageBlock(proto provider.Protocol, turns []byte, contentKey string) bool {
	found := false
	eachElement(turns, func(turn []byte) bool {
		found = anyImageInTurn(proto, turn, contentKey)
		return !found
	})
	return found
}

// anyImageInTurn is whether one of body's turns may hold a block
// walkImages gives to swap (anyImageBlock), read in place: a turn with no
// image in it is not decoded at all.
func anyImageInTurn(proto provider.Protocol, turn []byte, contentKey string) bool {
	if len(turn) == 0 || turn[0] != '{' {
		return false
	}
	found := false
	eachMember(turn, func(key, val []byte) bool {
		if len(val) > 0 && val[0] == '[' && (jsonStringIs(key, contentKey) || (proto == provider.Responses && jsonStringIs(key, "output"))) {
			found = anyImageIn(proto, val)
		}
		return !found
	})
	return found
}

// anyImageIn is whether blocks, a valid JSON array, has a block
// swapImageBlocks would look into (anyImageBlock).
func anyImageIn(proto provider.Protocol, blocks []byte) bool {
	kind := "image" // Anthropic's
	switch proto {
	case provider.Chat:
		kind = "image_url"
	case provider.Responses:
		kind = "input_image"
	}
	found := false
	eachElement(blocks, func(block []byte) bool {
		if block[0] != '{' {
			return true
		}
		// a tool's images are in its own blocks, looked into as
		// swapImageBlocks does: most agents' turns are tool results,
		// nearly all text (#3725)
		var inner []byte
		toolResult := false
		eachMember(block, func(key, val []byte) bool {
			switch {
			case proto == provider.Gemini && jsonStringIs(key, "functionResponse"):
				if val[0] == '{' {
					eachMember(val, func(k, v []byte) bool {
						if jsonStringIs(k, "parts") {
							inner = v // the last, as a decode keeps
						}
						return true
					})
				}
			case proto == provider.Gemini:
				found = jsonStringIs(key, "inlineData") || jsonStringIs(key, "fileData")
			case jsonStringIs(key, "type"):
				found = jsonStringIs(val, kind)
				toolResult = toolResult || (proto == provider.Anthropic && jsonStringIs(val, "tool_result"))
			case proto == provider.Anthropic && jsonStringIs(key, "content"):
				inner = val
			}
			return !found
		})
		if !found && len(inner) > 0 && inner[0] == '[' && (proto == provider.Gemini || toolResult) {
			found = anyImageIn(proto, inner)
		}
		return !found
	})
	return found
}

// swapImageBlocks swaps the images among a turn's blocks, and those in an
// Anthropic tool_result's, which are a tool's.
func swapImageBlocks(proto provider.Protocol, raw json.RawMessage, last, tool bool, swap func(imageAt) (json.RawMessage, error), at *swapAt) (json.RawMessage, bool, error) {
	var blocks []json.RawMessage
	if decodeJSON(raw, &blocks) != nil {
		return raw, false, nil
	}
	swapped := false
	for i, rawBlock := range blocks {
		var block map[string]json.RawMessage
		if decodeJSON(rawBlock, &block) != nil {
			continue
		}
		var kind string
		decodeJSON(block["type"], &kind)
		src, isImage := imageSrc(proto, kind, block)
		if isImage {
			out, err := swap(imageAt{Src: src, Last: last, Tool: tool, Block: rawBlock})
			if err != nil {
				return raw, false, err
			}
			if out != nil {
				if at != nil && at.ok {
					if j, ok := at.loc.find(rawBlock); ok {
						at.edits = append(at.edits, imageEdit{at: j, end: j + len(rawBlock), out: out})
					} else {
						at.ok = false
					}
				}
				blocks[i] = out
				swapped = true
			}
			continue
		}
		// Gemini's function response holds a tool's images in its own parts
		if proto == provider.Gemini && len(block["functionResponse"]) > 0 {
			var fr map[string]json.RawMessage
			if decodeJSON(block["functionResponse"], &fr) == nil && len(fr["parts"]) > 0 {
				content, nested, err := swapImageBlocks(proto, fr["parts"], last, true, swap, at)
				if err != nil {
					return raw, false, err
				}
				if nested {
					fr["parts"] = content
					block["functionResponse"], _ = json.Marshal(fr)
					blocks[i], _ = json.Marshal(block)
					swapped = true
				}
			}
		}
		if proto == provider.Anthropic && kind == "tool_result" {
			content, nested, err := swapImageBlocks(proto, block["content"], last, true, swap, at)
			if err != nil {
				return raw, false, err
			}
			if nested {
				block["content"] = content
				blocks[i], _ = json.Marshal(block)
				swapped = true
				if at != nil {
					at.ok = false // the block around the swap is written again
				}
			}
		}
	}
	if !swapped {
		return raw, false, nil
	}
	out, _ := json.Marshal(blocks)
	return out, true, nil
}

// imageSrc is whether a block is an image, and the image as a URL when it
// can be read from the block.
func imageSrc(proto provider.Protocol, kind string, block map[string]json.RawMessage) (string, bool) {
	switch proto {
	case provider.Chat:
		if kind != "image_url" {
			return "", false
		}
		var url string
		if decodeJSON(block["image_url"], &url) != nil {
			var obj struct {
				URL string `json:"url"`
			}
			decodeJSON(block["image_url"], &obj)
			url = obj.URL
		}
		return url, true
	case provider.Responses:
		if kind != "input_image" {
			return "", false
		}
		var url string
		decodeJSON(block["image_url"], &url)
		return url, true
	case provider.Anthropic:
		if kind != "image" {
			return "", false
		}
		var src struct {
			Type      string `json:"type"`
			MediaType string `json:"media_type"`
			Data      string `json:"data"`
			URL       string `json:"url"`
		}
		decodeJSON(block["source"], &src)
		switch {
		case src.Type == "base64" && src.Data != "":
			return "data:" + src.MediaType + ";base64," + src.Data, true
		case src.Type == "url":
			return src.URL, true
		}
		return "", true
	case provider.Gemini:
		var data struct {
			MimeType string `json:"mimeType"`
			Data     string `json:"data"`
		}
		decodeJSON(block["inlineData"], &data)
		if strings.HasPrefix(data.MimeType, "image/") {
			return "data:" + data.MimeType + ";base64," + data.Data, true
		}
		if len(block["fileData"]) > 0 {
			var file struct {
				MimeType string `json:"mimeType"`
			}
			if decodeJSON(block["fileData"], &file) != nil || file.MimeType == "" || strings.HasPrefix(file.MimeType, "image/") {
				return "", true // unknown file types may be images; fail closed
			}
		}
	}
	return "", false
}

// textBlock is a text block of the protocol's.
func textBlock(proto provider.Protocol, text string) json.RawMessage {
	var b []byte
	switch proto {
	case provider.Responses:
		b, _ = json.Marshal(map[string]string{"type": "input_text", "text": text})
	case provider.Gemini:
		b, _ = json.Marshal(map[string]string{"text": text})
	default:
		b, _ = json.Marshal(map[string]string{"type": "text", "text": text})
	}
	return b
}
