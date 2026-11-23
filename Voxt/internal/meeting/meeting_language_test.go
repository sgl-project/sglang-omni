package meeting_test

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// promptChat keeps every message the model was sent.
type promptChat struct{ sent []string }

func (c *promptChat) Chat(_ context.Context, req llm.Request) (llm.Response, error) {
	for _, m := range req.Messages {
		c.sent = append(c.sent, m.Content)
	}
	return llm.Response{Text: `{"transcript_summary":{"title":"T","content":"B"},"todo_list":[]}`}, nil
}

// A Chinese meeting with no main language set was summarized in English
// (#4011); the prompts now ask for the transcript's language instead.
func TestMeetingPromptsLanguage(t *testing.T) {
	for _, tc := range []struct {
		name, main, summary, followUp string
	}{
		{"unset", "", "User's main language:\nthe language most of the transcript is in", "User main language: the language most of the transcript is in"},
		{"blank", "  ", "User's main language:\nthe language most of the transcript is in", "User main language: the language most of the transcript is in"},
		{"set", "Japanese · 日本語", "User's main language:\nJapanese · 日本語", "User main language: Japanese · 日本語"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			s, err := sqlitestore.Open(ctx, filepath.Join(t.TempDir(), "db"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { s.Close() })
			created, err := s.Meetings().CreateMeeting(ctx, store.Meeting{ID: "ml", CreatedAt: time.Now(), Status: store.MeetingDone, Mode: store.MeetingModeMeeting})
			if err != nil {
				t.Fatal(err)
			}
			if err := s.Meetings().AppendSegments(ctx, created.Seq, []store.MeetingSegment{
				{Idx: 0, Source: store.SegmentMic, Text: "我们下周把预算定下来。"},
				{Idx: 1, Source: store.SegmentSystem, Text: "好的，财务那边我来跟进。"},
			}); err != nil {
				t.Fatal(err)
			}
			chat := &promptChat{}
			m := meeting.NewManager(meeting.Config{
				Store:        s.Meetings(),
				LLM:          func(context.Context, string) (meeting.SummaryChat, error) { return chat, nil },
				Settings:     func() settings.Meeting { return settings.Meeting{} },
				MainLanguage: func() string { return tc.main },
			})
			if err := m.GenerateMeetingSummary(ctx, "ml"); err != nil {
				t.Fatal(err)
			}
			if got := strings.Join(chat.sent, "\n---\n"); !strings.Contains(got, tc.summary) {
				t.Errorf("summary prompt lacks %q:\n%s", tc.summary, got)
			}
			chat.sent = nil
			if err := m.AnswerMeetingFollowUp(ctx, "ml", "谁跟进财务？"); err != nil {
				t.Fatal(err)
			}
			if got := strings.Join(chat.sent, "\n---\n"); !strings.Contains(got, tc.followUp) {
				t.Errorf("follow-up prompt lacks %q:\n%s", tc.followUp, got)
			}
		})
	}
	if got := meeting.RenderFollowUpSystem("", nil, "我：好的"); !strings.Contains(got, "User main language: the language most of the transcript is in") {
		t.Errorf("follow-up system with no language:\n%s", got)
	}
}
