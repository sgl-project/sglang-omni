package meeting_test

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// heardHints is a speech model that keeps the hot words of every request.
type heardHints struct {
	meetTestStreamer
	mu    sync.Mutex
	hints [][]string
}

func (h *heardHints) keep(hints []string) {
	h.mu.Lock()
	h.hints = append(h.hints, hints)
	h.mu.Unlock()
}

func (h *heardHints) got() [][]string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.hints)
}

func (h *heardHints) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	h.keep(req.Hints)
	return h.meetTestStreamer.Open(ctx, req)
}

func (h *heardHints) Transcribe(_ context.Context, req asr.FileRequest) (asr.Result, error) {
	h.keep(req.Hints)
	return asr.Result{Text: "heard"}, nil
}

// TestMeetingSendsHotWords: the hot words of a meeting's speech setup (the
// personal dictionary's) go with every request to the speech model: the
// live stream, the live transcription of a model without streaming, and
// the transcription after the meeting.
func TestMeetingSendsHotWords(t *testing.T) {
	want := []string{"SGLang", "Voxt"}
	ctx := context.Background()
	check := func(t *testing.T, h *heardHints) {
		t.Helper()
		got := h.got()
		if len(got) == 0 {
			t.Fatal("the speech model got no request")
		}
		for i, hints := range got {
			if !slices.Equal(hints, want) {
				t.Errorf("request %d: hot words %q, want %q", i, hints, want)
			}
		}
	}
	live := func(t *testing.T, h *heardHints, ta meeting.TrackASR) {
		dir := t.TempDir()
		s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { s.Close() })
		m := meeting.NewManager(meeting.Config{
			Mic:       newFakeAudio(liveTestPCM(t)),
			Store:     s.Meetings(),
			Dir:       filepath.Join(dir, "meetings"),
			Privacy:   func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
			Settings:  func() settings.Meeting { return settings.Meeting{NoFinalPass: true, NoAutoSummary: true} },
			FreeSpace: func(string) (int64, error) { return meeting.DiskMinStartBytes() + 1, nil },
			ASR:       func(context.Context, meeting.Source) (meeting.TrackASR, error) { return ta, nil },
		})
		id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
		if err != nil {
			t.Fatal(err)
		}
		for deadline := time.Now().Add(10 * time.Second); len(h.got()) == 0 && time.Now().Before(deadline); {
			time.Sleep(5 * time.Millisecond)
		}
		if len(h.got()) == 0 {
			t.Fatal("the speech model got no live request before Finish")
		}
		if err := m.Finish(ctx, id); err != nil {
			t.Fatal(err)
		}
		mt, _, err := s.Meetings().Meeting(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		if mt.Status != store.MeetingDone {
			t.Fatalf("meeting status = %s, want %s", mt.Status, store.MeetingDone)
		}
		check(t, h)
	}
	t.Run("live stream", func(t *testing.T) {
		h := &heardHints{meetTestStreamer: meetTestStreamer{text: "hello"}}
		live(t, h, meeting.TrackASR{Caps: asr.Caps{Stream: true}, Streamer: h, Hints: want, StreamReq: asr.StreamRequest{Hints: want}})
	})
	t.Run("live without streaming", func(t *testing.T) {
		h := &heardHints{}
		live(t, h, meeting.TrackASR{Caps: asr.Caps{File: true}, File: h, Hints: want, StreamReq: asr.StreamRequest{Hints: want}})
	})
	t.Run("after the meeting", func(t *testing.T) {
		root := t.TempDir()
		db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { db.Close() })
		meetDir := filepath.Join(root, "meetings", "m1")
		if err := os.MkdirAll(meetDir, 0o755); err != nil {
			t.Fatal(err)
		}
		writeWAV(t, meetDir, "mic.wav", tonePCM(rateSamples(2*time.Second)))
		created, err := db.Meetings().CreateMeeting(ctx, store.Meeting{
			ID: "m1", CreatedAt: time.Now(), Status: store.MeetingDone, Mode: store.MeetingModeRecording, Audio: "m1", DurationMS: 2000,
		})
		if err != nil {
			t.Fatal(err)
		}
		h := &heardHints{}
		cfg := meeting.Config{
			Store:    db.Meetings(),
			Dir:      filepath.Join(root, "meetings"),
			Settings: func() settings.Meeting { return settings.Meeting{} },
			Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
			ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
				return meeting.TrackASR{Caps: asr.Caps{File: true}, File: h, Hints: want}, nil
			},
		}
		if err := meeting.FinalizePass(ctx, cfg, created, nil); err != nil {
			t.Fatal(err)
		}
		check(t, h)
	})
}
