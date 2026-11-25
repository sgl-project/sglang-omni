package meeting_test

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// mistakeMeeting records a meeting that hears text and runs for ran, then
// ends it. It returns the manager, its store, the meeting's id and folder,
// and the snapshots its subscriber got.
func mistakeMeeting(t *testing.T, text string, ran time.Duration, keepHistory bool) (*meeting.Manager, *sqlitestore.Store, string, string, *told) {
	t.Helper()
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	clk := newFakeClock(time.Unix(1_700_000_000, 0))
	streamer := &meetTestStreamer{text: text}
	m := meeting.NewManager(meeting.Config{
		Mic:       newFakeAudio(liveTestPCM(t)),
		Store:     s.Meetings(),
		Dir:       filepath.Join(dir, "meetings"),
		Now:       clk.Now,
		Privacy:   func() settings.Privacy { return settings.Privacy{KeepHistory: keepHistory} },
		Settings:  func() settings.Meeting { return settings.Meeting{NoFinalPass: true, NoAutoSummary: true} },
		FreeSpace: func(string) (int64, error) { return meeting.DiskMinStartBytes() + 1, nil },
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: streamer.Caps(), Streamer: streamer}, nil
		},
	})
	got := &told{heard: make(chan struct{}, 1)}
	m.Subscribe(got.tell)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-got.heard:
	case <-time.After(10 * time.Second):
		t.Fatal("no live text from the microphone")
	}
	clk.Advance(ran)
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	return m, s, id, filepath.Join(dir, "meetings", id), got
}

// told are the snapshots a subscriber got; heard gets a value once one
// has live text from the microphone.
type told struct {
	mu    sync.Mutex
	list  []meeting.Snapshot
	heard chan struct{}
}

func (g *told) tell(s meeting.Snapshot) {
	g.mu.Lock()
	g.list = append(g.list, s)
	g.mu.Unlock()
	if s.Live[meeting.SourceMic] != "" {
		select {
		case g.heard <- struct{}{}:
		default:
		}
	}
}

// last is the newest snapshot the subscriber got.
func (g *told) last() meeting.Snapshot {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.list[len(g.list)-1]
}

func meetingStatus(t *testing.T, s *sqlitestore.Store, id string) store.MeetingStatus {
	t.Helper()
	mt, _, err := s.Meetings().Meeting(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	return mt.Status
}

// A meeting that ends within moments with hardly a word heard asks
// whether it was started by mistake; kept, it is filed as any meeting.
func TestFewWordsAskAndKeep(t *testing.T) {
	m, s, id, _, got := mistakeMeeting(t, "hello live", 3*time.Second, true)
	if now := m.Current(); now.State != meeting.StateAsking || !now.Mistake || meetingStatus(t, s, id) != store.MeetingAsking {
		t.Fatalf("after two words in three seconds: %s, mistake %v, row %s", now.State, now.Mistake, meetingStatus(t, s, id))
	}
	if err := m.Save(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	last := got.last()
	if now := m.Current(); now.State != meeting.StateIdle || now.Mistake || last.State != meeting.StateIdle || meetingStatus(t, s, id) != store.MeetingDone {
		t.Fatalf("kept: %s, told %s, row %s", now.State, last.State, meetingStatus(t, s, id))
	}
	_, segs, _ := s.Meetings().Meeting(context.Background(), id)
	if len(segs) != 1 || segs[0].Text != "hello live" {
		t.Fatalf("the kept meeting's transcript: %+v", segs)
	}
}

// Discarded, the meeting and its recording are gone, and the subscriber
// hears of it (the overlay that asked goes).
func TestFewWordsAskAndDiscard(t *testing.T) {
	m, s, id, folder, got := mistakeMeeting(t, "你好", 10*time.Second, true)
	if now := m.Current(); now.State != meeting.StateAsking || !now.Mistake {
		t.Fatalf("after two characters: %s, mistake %v", now.State, now.Mistake)
	}
	if err := m.Discard(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if last := got.last(); last.State != meeting.StateIdle || m.Current().State != meeting.StateIdle {
		t.Fatalf("discarded: %s, told %s", m.Current().State, last.State)
	}
	if list, _ := s.Meetings().ListMeetings(context.Background(), store.MeetingQuery{}); len(list) != 0 {
		t.Fatalf("rows left: %d", len(list))
	}
	if _, err := os.Stat(folder); !os.IsNotExist(err) {
		t.Fatalf("the recording's folder is still there: %v", err)
	}
}

// Enough words, or a meeting that ran long, ends without the question:
// a long recording is never one click from being deleted.
func TestEnoughWordsOrTimeDoNotAsk(t *testing.T) {
	for name, c := range map[string]struct {
		text string
		ran  time.Duration
	}{
		"five words":        {"let us start the meeting", 3 * time.Second},
		"a Chinese clause":  {"我们现在开始开会", 3 * time.Second},
		"two words, longer": {"hello live", meeting.MistakeWithin},
	} {
		m, s, id, _, _ := mistakeMeeting(t, c.text, c.ran, true)
		if now := m.Current(); now.State != meeting.StateIdle || meetingStatus(t, s, id) != store.MeetingDone {
			t.Errorf("%s: %s, row %s", name, now.State, meetingStatus(t, s, id))
		}
	}
}

// With history off every meeting asks; one with few words still says so,
// and Save keeps it as before.
func TestFewWordsWithHistoryOff(t *testing.T) {
	m, s, id, _, got := mistakeMeeting(t, "hello", time.Second, false)
	if now := m.Current(); now.State != meeting.StateAsking || !now.Mistake {
		t.Fatalf("history off, one word: %s, mistake %v", now.State, now.Mistake)
	}
	if err := m.Save(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if last := got.last(); last.State != meeting.StateIdle || meetingStatus(t, s, id) != store.MeetingDone {
		t.Fatalf("saved: told %s, row %s", last.State, meetingStatus(t, s, id))
	}
	long, _, _, _, _ := mistakeMeeting(t, "let us start the meeting", time.Minute, false)
	if now := long.Current(); now.State != meeting.StateAsking || now.Mistake {
		t.Fatalf("history off, a real meeting: %s, mistake %v", now.State, now.Mistake)
	}
}
