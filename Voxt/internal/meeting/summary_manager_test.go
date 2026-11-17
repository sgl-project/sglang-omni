package meeting_test

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

type stubLLM struct{}

func (stubLLM) Chat(context.Context, llm.Request) (llm.Response, error) {
	return llm.Response{Text: `{"transcript_summary":{"title":"OK","content":"done"},"todo_list":[]}`}, nil
}

func TestGenerateSummaryNoLLMConfigured(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	meet := store.Meeting{ID: "m1", CreatedAt: time.Now(), Status: store.MeetingDone, Mode: store.MeetingModeMeeting}
	created, _ := s.Meetings().CreateMeeting(ctx, meet)
	_ = s.Meetings().AppendSegments(ctx, created.Seq, []store.MeetingSegment{{Source: store.SegmentMic, Text: "hi"}})
	m := meeting.NewManager(meeting.Config{
		Store:    s.Meetings(),
		Settings: func() settings.Meeting { return settings.Meeting{} },
	})
	if err := m.GenerateMeetingSummary(ctx, "m1"); err == nil {
		t.Fatal("expected error without LLM")
	}
	mt, _, _ := s.Meetings().Meeting(ctx, "m1")
	if mt.Summary != "" {
		t.Fatal("summary should stay empty when LLM missing")
	}
}

func TestGenerateSummaryPersistsOnLLMError(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	meet := store.Meeting{ID: "m2", CreatedAt: time.Now(), Status: store.MeetingDone, Mode: store.MeetingModeMeeting}
	created, _ := s.Meetings().CreateMeeting(ctx, meet)
	_ = s.Meetings().AppendSegments(ctx, created.Seq, []store.MeetingSegment{{Source: store.SegmentMic, Text: "kept"}})
	m := meeting.NewManager(meeting.Config{
		Store: s.Meetings(),
		LLM: func(context.Context, string) (meeting.SummaryChat, error) {
			return nil, errors.New("no llm")
		},
		Settings: func() settings.Meeting { return settings.Meeting{} },
	})
	_ = m.GenerateMeetingSummary(ctx, "m2")
	mt, segs, _ := s.Meetings().Meeting(ctx, "m2")
	if len(segs) != 1 || mt.Summary != "" {
		t.Fatal("prior transcript should remain without summary")
	}
}

func TestGenerateSummarySuccess(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	meet := store.Meeting{ID: "m3", CreatedAt: time.Now(), Status: store.MeetingDone, Mode: store.MeetingModeMeeting}
	created, _ := s.Meetings().CreateMeeting(ctx, meet)
	_ = s.Meetings().AppendSegments(ctx, created.Seq, []store.MeetingSegment{{Source: store.SegmentMic, Text: "note"}})
	m := meeting.NewManager(meeting.Config{
		Store: s.Meetings(),
		LLM: func(context.Context, string) (meeting.SummaryChat, error) {
			return stubLLM{}, nil
		},
		Settings: func() settings.Meeting { return settings.Meeting{} },
	})
	if err := m.GenerateMeetingSummary(ctx, "m3"); err != nil {
		t.Fatal(err)
	}
	mt, _, _ := s.Meetings().Meeting(ctx, "m3")
	if mt.Title != "OK" || mt.Summary == "" {
		t.Fatalf("summary %+v", mt)
	}
}
