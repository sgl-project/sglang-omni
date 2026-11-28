package llm

import (
	"context"
	"fmt"
	"io"
	"slices"
	"strings"
	"testing"
	"testing/iotest"
)

func TestReadSSELineFraming(t *testing.T) {
	for name, ending := range map[string]string{"LF": "\n", "CRLF": "\r\n", "CR": "\r"} {
		for _, bom := range []string{"", "\ufeff"} {
			t.Run(fmt.Sprintf("%s/BOM=%t", name, bom != ""), func(t *testing.T) {
				raw := bom + "data: first" + ending + "data: second" + ending + ending
				raw += ": keep-alive" + ending + ending + "data: \ufeffthird" + ending + ending
				for _, reader := range []io.Reader{strings.NewReader(raw), iotest.OneByteReader(strings.NewReader(raw))} {
					var events []string
					err := readSSE(reader, func(data string) bool {
						events = append(events, data)
						return true
					})
					if err != nil || !slices.Equal(events, []string{"first\nsecond", "\ufeffthird"}) {
						t.Fatalf("events %q, error %v", events, err)
					}
				}
			})
		}
	}
}

func TestOpenAIStreamLineFraming(t *testing.T) {
	for name, ending := range map[string]string{"LF": "\n", "CRLF": "\r\n", "CR": "\r"} {
		for _, bom := range []string{"", "\ufeff"} {
			t.Run(fmt.Sprintf("%s/BOM=%t", name, bom != ""), func(t *testing.T) {
				raw := bom + `data: {"model":"m1","choices":[{"delta":{"content":"H"}}]}` + ending + ending
				raw += `data: {"choices":[{"delta":{"content":"i"},"finish_reason":"stop"}]}` + ending + ending
				raw += "data: [DONE]" + ending + ending
				f := newFakeServer(t, sse(raw))
				var deltas []string
				c := NewOpenAI(presetAt(t, "openai", f.srv.URL), "sk-test", f.srv.Client())
				resp, err := c.Chat(context.Background(), Request{Messages: msgs("hi"), OnDelta: func(s string) { deltas = append(deltas, s) }})
				if err != nil || resp.Text != "Hi" || resp.Model != "m1" || !slices.Equal(deltas, []string{"H", "i"}) {
					t.Fatalf("response %+v, deltas %q, error %v", resp, deltas, err)
				}
			})
		}
	}
}
