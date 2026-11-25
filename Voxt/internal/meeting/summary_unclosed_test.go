package meeting

import "testing"

// A model may leave the summary's last brace out (seen from a remote
// "flash" model: two objects opened, one closed). The summary is still
// read, not shown to the user as raw JSON.
func TestParseSummaryUnclosedObject(t *testing.T) {
	for name, raw := range map[string]string{
		"the outer brace is missing":   `{"transcript_summary":{"title":"Short talk","content":"They said hello.","todo_list":[]}`,
		"both braces are missing":      `{"transcript_summary":{"title":"Short talk","content":"They said hello."`,
		"fenced, one brace missing":    "```json\n{\"transcript_summary\":{\"title\":\"Short talk\",\"content\":\"They said hello.\"}\n```",
		"todos inside the inner block": `{"transcript_summary":{"title":"Short talk","content":"They said hello.","todo_list":["Call back"]}`,
	} {
		s, err := ParseSummary(raw)
		if err != nil || s.Title != "Short talk" || s.Body != "They said hello." {
			t.Errorf("%s: %+v, %v", name, s, err)
		}
		if name == "todos inside the inner block" && (len(s.Todos) != 1 || s.Todos[0] != "Call back") {
			t.Errorf("%s: todos %q", name, s.Todos)
		}
	}
	// Plain text that only mentions a brace stays the body it is.
	if s, err := ParseSummary("They talked about {braces} and left."); err != nil || s.Body != "They talked about {braces} and left." {
		t.Errorf("plain text: %+v, %v", s, err)
	}
	// A string cut in the middle cannot be repaired by braces alone: it
	// stays text rather than a guess.
	if s, err := ParseSummary(`{"title":"Short talk","content":"They said he`); err != nil || s.Title != "Transcript Summary" {
		t.Errorf("cut inside a string: %+v, %v", s, err)
	}
}
