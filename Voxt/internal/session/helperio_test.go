package session

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// oneCall answers one helper call with result and keeps what was asked.
type oneCall struct {
	method string
	params any
	result string
}

func (c *oneCall) Call(_ context.Context, _, method string, params any, _ func(proto.Message)) (json.RawMessage, error) {
	c.method, c.params = method, params
	return json.RawMessage(c.result), nil
}

// TestHelperTextRead: Read asks the helper for textio.read with the
// params as given and returns its answer.
func TestHelperTextRead(t *testing.T) {
	c := &oneCall{result: `{"text":"Claude","range":{"location":3,"length":6},"length":20,"element":7}`}
	p := proto.ReadParams{PID: 812, Range: proto.TextRange{Location: 3, Length: 6}}
	got, err := HelperText{H: c}.Read(context.Background(), p)
	want := proto.ReadResult{Text: "Claude", Range: proto.TextRange{Location: 3, Length: 6}, Length: 20, Element: 7}
	if err != nil || got != want || c.method != proto.MethodTextRead || c.params != p {
		t.Errorf("Read: %+v %v; asked %s %+v", got, err, c.method, c.params)
	}
}
