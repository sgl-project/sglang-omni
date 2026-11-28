package llm

import (
	"fmt"
	"maps"
	"strings"
	"testing"
)

func TestResponsesStreamTextPartFallback(t *testing.T) {
	type part struct {
		outputIndex, contentIndex int
		text                      string
		deltas                    []string
	}
	for _, tc := range []struct {
		name  string
		parts []part
	}{
		{"done content parts", []part{{0, 0, "Hello ", nil}, {0, 1, "world.", nil}}},
		{"done output items", []part{{0, 0, "Hello ", nil}, {1, 0, "world.", nil}}},
		{"delta then done part", []part{{0, 0, "Hello ", []string{"Hel", "lo "}}, {0, 1, "world.", nil}}},
		{"done then delta part", []part{{0, 0, "Hello ", nil}, {0, 1, "world.", []string{"wor", "ld."}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream := &rsStream{}
			stream.emit("response.created", map[string]any{"response": rsObject("in_progress", nil, nil, nil, nil)})
			var output []any
			var wantDeltas []string
			for _, p := range tc.parts {
				itemID := fmt.Sprintf("msg_%d", p.outputIndex)
				loc := map[string]any{"item_id": itemID, "output_index": p.outputIndex, "content_index": p.contentIndex}
				for _, delta := range p.deltas {
					fields := maps.Clone(loc)
					fields["delta"] = delta
					stream.emit("response.output_text.delta", fields)
					wantDeltas = append(wantDeltas, delta)
				}
				fields := maps.Clone(loc)
				fields["text"] = p.text
				stream.emit("response.output_text.done", fields)
				if len(p.deltas) == 0 {
					wantDeltas = append(wantDeltas, p.text)
				}
				if p.outputIndex == len(output) {
					item := rsMessage(rsTextPart(p.text))
					item["id"] = itemID
					output = append(output, item)
				} else {
					item := output[p.outputIndex].(map[string]any)
					item["content"] = append(item["content"].([]any), rsTextPart(p.text))
				}
			}
			rsCompleted(stream, output)
			f := newFakeServer(t, sse(stream.out...))
			var deltas []string
			resp, err := rsChat(t, f, Request{Messages: msgs("hi"), OnDelta: func(s string) { deltas = append(deltas, s) }})
			if err != nil || resp.Text != "Hello world." || resp.FinishReason != FinishStop {
				t.Fatalf("response %+v, error %v", resp, err)
			}
			if strings.Join(deltas, "|") != strings.Join(wantDeltas, "|") {
				t.Fatalf("deltas %q, want %q", deltas, wantDeltas)
			}
			if resp.Usage.TotalTokens != 18 || resp.Model != "gpt-reported" {
				t.Fatalf("terminal metadata lost: %+v", resp)
			}
			if requests := f.requests(); len(requests) != 1 || requests[0].Body["stream"] != true {
				t.Fatalf("streaming request was not used: %+v", requests)
			}
		})
	}
}
