package meeting_test

import (
	"context"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

type recordingChat struct {
	calls []string
	reply func(prompt string) string
}

func (c *recordingChat) Chat(_ context.Context, req llm.Request) (llm.Response, error) {
	text := ""
	if len(req.Messages) > 0 {
		text = req.Messages[len(req.Messages)-1].Content
	}
	c.calls = append(c.calls, text)
	if c.reply != nil {
		return llm.Response{Text: c.reply(text)}, nil
	}
	return llm.Response{Text: `{"transcript_summary":{"title":"T","content":"B"},"todo_list":[]}`}, nil
}

func TestRenderSummaryPromptReplacesTranscriptAndSpeakerNames(t *testing.T) {
	segs := []meeting.Segment{
		{Speaker: "s1", Text: "hello"},
	}
	names := map[string]string{"s1": "Alex"}
	record := meeting.TranscriptRecord(segs, names)
	prompt := meeting.RenderSummaryPrompt("x {{USER_MAIN_LANGUAGE}} {{TRANSCRIPT_RECORD}}", "English", record)
	if !strings.Contains(prompt, "Alex：hello") {
		t.Fatalf("speaker name missing: %q", prompt)
	}
	if strings.Contains(prompt, "{{TRANSCRIPT_RECORD}}") {
		t.Fatal("placeholder not replaced")
	}
}

func TestParseSummaryKeysAndFences(t *testing.T) {
	cases := []struct {
		raw      string
		wantBody string
		wantTodo int
	}{
		{`{"transcript_summary":{"title":"A","content":"body"},"todo_list":["one"]}`, "body", 1},
		{`{"title":"B","body":"plain","todoItems":["x"]}`, "plain", 1},
		{"```json\n{\"title\":\"C\",\"content\":\"in fence\"}\n```", "in fence", 0},
		{"Just plain prose.", "Just plain prose.", 0},
	}
	for _, tc := range cases {
		sum, err := meeting.ParseSummary(tc.raw)
		if err != nil {
			t.Fatalf("%q: %v", tc.raw, err)
		}
		if sum.Body != tc.wantBody {
			t.Fatalf("%q body %q want %q", tc.raw, sum.Body, tc.wantBody)
		}
		if len(sum.Todos) != tc.wantTodo {
			t.Fatalf("%q todos %d", tc.raw, len(sum.Todos))
		}
	}
}

func TestSummaryWindows30000IntoThree(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 300; i++ {
		b.WriteString("This is sentence number ")
		b.WriteString(strings.Repeat("x", 90))
		b.WriteString(".\n")
	}
	windows := meeting.SummaryWindows(b.String(), 12000)
	if len(windows) != 3 {
		t.Fatalf("want 3 windows got %d", len(windows))
	}
	for _, w := range windows {
		if strings.HasSuffix(strings.TrimSpace(w), "Th") || strings.HasSuffix(strings.TrimSpace(w), "se") {
			t.Fatalf("mid-sentence cut: ...%q", w[len(w)-20:])
		}
	}
}

func TestGenerateSummaryHierarchicalCalls(t *testing.T) {
	transcript := strings.Repeat("a", 11000) + "\n" + strings.Repeat("b", 11000) + "\n" + strings.Repeat("c", 11000)
	if len(meeting.SummaryWindows(transcript, 12000)) != 3 {
		t.Fatal("setup: want three windows")
	}
	chat := &recordingChat{}
	_, err := meeting.GenerateSummary(context.Background(), chat, meeting.DefaultSummaryPrompt, "English", transcript)
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.calls) != 4 {
		t.Fatalf("want 4 LLM calls (3 windows + merge) got %d", len(chat.calls))
	}
}
