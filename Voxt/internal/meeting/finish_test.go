package meeting_test

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// End shows the meeting as ending while its tracks recognise what they
// still hold, and stops them together: the wait is the longer one's, not
// the sum of both (11-25: a 17-minute meeting stayed "recording" ~4 s
// after End).
func TestFinishEndsTracksTogether(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	mic, sys := newFakeAudio(nil), newFakeAudio(nil)
	m := meeting.NewManager(meeting.Config{
		Mic: mic, System: sys, Store: db.Meetings(), Dir: filepath.Join(root, "meetings"),
		Now:       newFakeClock(time.Unix(1_700_000_000, 0)).Now,
		Privacy:   func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) { return meeting.DiskMinStartBytes() + 1, nil },
	})
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
	if err != nil {
		t.Fatal(err)
	}
	var repeated atomic.Bool
	m.Subscribe(func(s meeting.Snapshot) {
		if s.Stage == meeting.StageSaving && repeated.CompareAndSwap(false, true) {
			if err := m.Finish(ctx, id); err != nil {
				t.Errorf("End again during its first notification: %v", err)
			}
		}
	})
	var mu sync.Mutex
	var seen []meeting.Snapshot
	micIn, sysIn := make(chan struct{}), make(chan struct{})
	together := func(mine, other chan struct{}) func() {
		return func() {
			mu.Lock()
			seen = append(seen, m.Current())
			mu.Unlock()
			close(mine)
			select {
			case <-other:
			case <-time.After(2 * time.Second):
				t.Error("the tracks stopped one after the other")
			}
		}
	}
	mic.onStop, sys.onStop = together(micIn, sysIn), together(sysIn, micIn)
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 {
		t.Fatalf("%d tracks stopped, want 2", len(seen))
	}
	if !repeated.Load() {
		t.Fatal("the regression never repeated End during its first notification")
	}
	for _, s := range seen {
		if s.State != meeting.StateFinishing || s.Stage != meeting.StageSaving {
			t.Errorf("while the tracks stopped: %s/%s, want finishing/saving", s.State, s.Stage)
		}
	}
}

// The final pass's progress is the share of all the speech recognised:
// a track weighs what it speaks, not half each.
func TestFinalizeProgressFollowsSpeech(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	dir := filepath.Join(root, "meetings", "m1")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeWAV(t, dir, "mic.wav", tonePCM(rateSamples(6*time.Second)))
	writeWAV(t, dir, "system.wav", tonePCM(rateSamples(2*time.Second)))
	created, err := db.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "m1", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeMeeting, Audio: "m1", DurationMS: 6000,
	})
	if err != nil {
		t.Fatal(err)
	}
	speech := &seqFileASR{texts: []string{"mic", "sys"}}
	cfg := meeting.Config{
		Store: db.Meetings(), Dir: filepath.Join(root, "meetings"),
		Settings: func() settings.Meeting { return settings.Meeting{} },
		Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: speech.Caps(), File: speech}, nil
		},
	}
	var progress []float64
	if err := meeting.FinalizePass(ctx, cfg, created, func(p float64) { progress = append(progress, p) }); err != nil {
		t.Fatal(err)
	}
	if len(progress) == 0 || progress[len(progress)-1] != 1 {
		t.Fatalf("progress %v: want it to end at 1", progress)
	}
	micDone := -1.0
	for i, p := range progress {
		if i > 0 && p < progress[i-1] {
			t.Fatalf("progress %v went back", progress)
		}
		if p > 0 && p < 1 && micDone < 0 {
			micDone = p
		}
	}
	if math.Abs(micDone-0.75) > 0.05 {
		t.Fatalf("progress %v: the microphone's 6 s of 8 s of speech should end at 75%%", progress)
	}
}
