package meeting_test

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

func TestAnswerMeetingFollowUpPersists(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	meet := store.Meeting{ID: "mf1", CreatedAt: time.Now(), Status: store.MeetingDone, Mode: store.MeetingModeMeeting}
	created, _ := s.Meetings().CreateMeeting(ctx, meet)
	_ = s.Meetings().AppendSegments(ctx, created.Seq, []store.MeetingSegment{
		{Source: store.SegmentMic, Text: "We approved the budget."},
	})
	chat := &followUpChat{text: "Finance led the approval."}
	m := meeting.NewManager(meeting.Config{
		Store: s.Meetings(),
		LLM: func(context.Context, string) (meeting.SummaryChat, error) {
			return chat, nil
		},
		Settings: func() settings.Meeting { return settings.Meeting{} },
	})
	if err := m.AnswerMeetingFollowUp(ctx, "mf1", "Who approved?"); err != nil {
		t.Fatal(err)
	}
	mt, _, _ := s.Meetings().Meeting(ctx, "mf1")
	msgs, err := meeting.ParseChatMessages(mt.Chat)
	if err != nil || len(msgs) != 2 || msgs[0].Content != "Who approved?" || msgs[1].Content != "Finance led the approval." {
		t.Fatalf("%v %v", msgs, err)
	}
	// Re-open persists the same blob.
	if err := m.AnswerMeetingFollowUp(ctx, "mf1", "Any risks?"); err != nil {
		t.Fatal(err)
	}
	mt, _, _ = s.Meetings().Meeting(ctx, "mf1")
	msgs, _ = meeting.ParseChatMessages(mt.Chat)
	if len(msgs) != 4 || msgs[2].Content != "Any risks?" {
		t.Fatalf("round 2: %+v", msgs)
	}
	if len(chat.reqs) != 2 || len(chat.reqs[1].Messages) != 4 {
		t.Fatalf("second request should include prior history: %d msgs", len(chat.reqs[1].Messages))
	}
}
