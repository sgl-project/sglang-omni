package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func FuzzExtractJSON(f *testing.F) {
	for _, s := range []string{`{"a":1}`, "```json\n[1]\n```", `x {"a":"}"} y`, "<think>{</think>{}", `[{"id":"1","text":"\"q\""}]`} {
		f.Add(s)
	}
	schema := Object(map[string]*Schema{"lines": Array(Object(map[string]*Schema{"id": String(), "text": String()}, "id", "text"))}, "lines")
	f.Fuzz(func(t *testing.T, s string) {
		raw := ExtractJSON(s)
		if raw != nil && !json.Valid(raw) {
			t.Fatalf("invalid JSON %q from %q", raw, s)
		}
		if raw != nil {
			_ = schema.Validate(raw)
		}
		_ = StripThinking(s)
		var out strings.Builder
		tf := &thinkFilter{emit: func(d string) { out.WriteString(d) }}
		for i := 0; i < len(s); i += 3 {
			tf.write(s[i:min(i+3, len(s))])
		}
		tf.flush()
	})
}
