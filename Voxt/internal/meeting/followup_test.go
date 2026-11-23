package meeting_test

import (
	"context"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

type followUpChat struct {
	reqs []llm.Request
	text string
}

func (r *followUpChat) Chat(_ context.Context, req llm.Request) (llm.Response, error) {
	r.reqs = append(r.reqs, req)
	if r.text == "" {
		r.text = "answer"
	}
	return llm.Response{Text: r.text}, nil
}

func TestRelevantFollowUpContextWithinLimit(t *testing.T) {
	var lines []string
	for i := 0; i < 40; i++ {
		lines = append(lines, strings.Repeat("filler ", 120)+strings.Repeat("x", 40))
	}
	transcript := strings.Join(lines, "\n")
	ctx := meeting.RelevantFollowUpContext(transcript, "specific keyword xyzzy")
	if ctx == "" {
		t.Fatal("expected context")
	}
	if len([]rune(ctx)) > meeting.FollowUpWindowMax {
		t.Fatalf("context %d runes > %d", len([]rune(ctx)), meeting.FollowUpWindowMax)
	}
}

func TestRelevantFollowUpContextSelectsRelevantWindow(t *testing.T) {
	transcript := strings.Repeat("opening discussion. ", 350) + "\n" +
		strings.Repeat("budget approval and finance decision. ", 50) + "\n" +
		strings.Repeat("closing remarks. ", 350)
	ctx := meeting.RelevantFollowUpContext(transcript, "budget approval")
	if !strings.Contains(ctx, "budget approval") {
		t.Fatalf("missing relevant window: %q", ctx)
	}
	if len([]rune(ctx)) > meeting.FollowUpWindowMax {
		t.Fatalf("context too long: %d", len([]rune(ctx)))
	}
}

func TestFollowUpSystemIncludesSummary(t *testing.T) {
	sum := &meeting.Summary{Title: "Budget", Body: "We approved Q2 spend."}
	sys := meeting.RenderFollowUpSystem("English", sum, "Me: hello")
	if !strings.Contains(sys, "Budget") || !strings.Contains(sys, "Q2 spend") {
		t.Fatalf("summary missing from system: %q", sys)
	}
}

func TestBuildFollowUpRequestHistoryOrder(t *testing.T) {
	history := []meeting.ChatMessage{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "one"},
		{Role: "user", Content: "second"},
	}
	req := meeting.BuildFollowUpRequest("English", nil, "transcript bit", history, "third")
	if len(req.Messages) != 5 {
		t.Fatalf("messages %d", len(req.Messages))
	}
	if req.Messages[0].Role != llm.System {
		t.Fatalf("first role %q", req.Messages[0].Role)
	}
	for i, want := range []struct {
		role llm.Role
		text string
	}{
		{llm.User, "first"},
		{llm.Assistant, "one"},
		{llm.User, "second"},
		{llm.User, "third"},
	} {
		got := req.Messages[i+1]
		if got.Role != want.role || got.Content != want.text {
			t.Fatalf("[%d] got %q/%q want %q/%q", i, got.Role, got.Content, want.role, want.text)
		}
	}
}

func TestAnswerFollowUpCallsChat(t *testing.T) {
	chat := &followUpChat{text: "Budget was approved."}
	history := []meeting.ChatMessage{{Role: "user", Content: "prior"}, {Role: "assistant", Content: "prior answer"}}
	got, err := meeting.AnswerFollowUp(context.Background(), chat, "English", "Me: budget approval noted.", nil, history, "Who approved?")
	if err != nil || got != "Budget was approved." {
		t.Fatalf("%q %v", got, err)
	}
	if len(chat.reqs) != 1 || len(chat.reqs[0].Messages) < 4 {
		t.Fatalf("req %v", chat.reqs)
	}
}

func TestEncodeParseChatRoundTrip(t *testing.T) {
	msgs := []meeting.ChatMessage{{Role: "user", Content: "q"}, {Role: "assistant", Content: "a"}}
	raw, err := meeting.EncodeChatMessages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	out, err := meeting.ParseChatMessages(raw)
	if err != nil || len(out) != 2 || out[1].Content != "a" {
		t.Fatalf("%v %v", out, err)
	}
}
