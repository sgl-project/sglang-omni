package meeting_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// captureRig is a manager over fake captures the test can look at.
type captureRig struct {
	m        *meeting.Manager
	mic, sys *fakeAudio
	meetings store.Meetings
	dir      string // the folder of the meetings' recordings
}

func newCaptureRig(t *testing.T) captureRig {
	t.Helper()
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	r := captureRig{mic: newFakeAudio(nil), sys: newFakeAudio(nil), meetings: db.Meetings(), dir: filepath.Join(root, "meetings")}
	r.m = meeting.NewManager(meeting.Config{
		Mic: r.mic, System: r.sys, Store: r.meetings, Dir: r.dir,
		Now:       newFakeClock(time.Unix(1_700_000_000, 0)).Now,
		Privacy:   func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) { return meeting.DiskMinStartBytes() + 1, nil },
	})
	return r
}

// TestSetModeKeepsSharedTracks: a change of mode while recording leaves
// the tracks both modes have open, on the capture they had (it used to
// stop them all and start only the new ones, so a meeting switched to
// Recording went on with nothing recording), starts the new mode's and
// stops the old one's.
func TestSetModeKeepsSharedTracks(t *testing.T) {
	for _, tc := range []struct {
		name             string
		from, to         meeting.Mode
		micJobs, sysJobs int // captures ever started
		micOpen, sysOpen int // captures open after the change
		micSame, sysSame bool
	}{
		{"meeting to recording", meeting.ModeMeeting, meeting.ModeRecording, 1, 1, 1, 0, true, false},
		{"meeting to subtitles", meeting.ModeMeeting, meeting.ModeSubtitles, 1, 1, 0, 1, false, true},
		{"recording to meeting", meeting.ModeRecording, meeting.ModeMeeting, 1, 1, 1, 1, true, false},
		{"subtitles to meeting", meeting.ModeSubtitles, meeting.ModeMeeting, 1, 1, 1, 1, false, true},
		{"subtitles to recording", meeting.ModeSubtitles, meeting.ModeRecording, 1, 1, 1, 0, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			r := newCaptureRig(t)
			id, err := r.m.Start(ctx, meeting.StartRequest{Mode: tc.from})
			if err != nil {
				t.Fatal(err)
			}
			micBefore, sysBefore := r.mic.jobIDs(), r.sys.jobIDs()
			if err := r.m.SetMode(ctx, id, tc.to); err != nil {
				t.Fatal(err)
			}
			if got := r.m.Current(); got.Mode != tc.to || got.State != meeting.StateRecording {
				t.Fatalf("after the change: mode %s, state %s", got.Mode, got.State)
			}
			if n, open := len(r.mic.jobIDs()), r.mic.running(); n != tc.micJobs || open != tc.micOpen {
				t.Errorf("microphone: %d captures started, %d open; want %d and %d", n, open, tc.micJobs, tc.micOpen)
			}
			if n, open := len(r.sys.jobIDs()), r.sys.running(); n != tc.sysJobs || open != tc.sysOpen {
				t.Errorf("system audio: %d captures started, %d open; want %d and %d", n, open, tc.sysJobs, tc.sysOpen)
			}
			if tc.micSame && (len(micBefore) != 1 || r.mic.jobIDs()[0] != micBefore[0]) {
				t.Errorf("the microphone's capture changed: %v, was %v", r.mic.jobIDs(), micBefore)
			}
			if tc.sysSame && (len(sysBefore) != 1 || r.sys.jobIDs()[0] != sysBefore[0]) {
				t.Errorf("system audio's capture changed: %v, was %v", r.sys.jobIDs(), sysBefore)
			}
			if err := r.m.Finish(ctx, id); err != nil {
				t.Fatal(err)
			}
			if r.mic.running() != 0 || r.sys.running() != 0 {
				t.Errorf("after Finish: %d microphone and %d system captures open", r.mic.running(), r.sys.running())
			}
		})
	}
}

// TestSetModeFailureKeepsRecording: the new mode's track not starting
// (system audio not allowed) leaves the meeting in its mode, recording.
func TestSetModeFailureKeepsRecording(t *testing.T) {
	ctx := context.Background()
	r := newCaptureRig(t)
	id, err := r.m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	r.sys.startErr = errors.New("permission denied")
	var se *meeting.StartError
	if err := r.m.SetMode(ctx, id, meeting.ModeMeeting); !errors.As(err, &se) || se.Source != meeting.SourceSystem {
		t.Fatalf("SetMode: %v, want system audio's start error", err)
	}
	if got := r.m.Current(); got.Mode != meeting.ModeRecording || got.State != meeting.StateRecording {
		t.Fatalf("after the failure: mode %s, state %s", got.Mode, got.State)
	}
	if n, open := len(r.mic.jobIDs()), r.mic.running(); n != 1 || open != 1 {
		t.Fatalf("microphone: %d captures started, %d open; want the first one, open", n, open)
	}
	// Allowed now: the same change works.
	r.sys.startErr = nil
	if err := r.m.SetMode(ctx, id, meeting.ModeMeeting); err != nil {
		t.Fatal(err)
	}
	if r.mic.running() != 1 || r.sys.running() != 1 {
		t.Fatalf("after the second change: %d microphone and %d system captures open", r.mic.running(), r.sys.running())
	}
}

// TestSetModeWhilePaused: nothing records while paused; Resume opens the
// new mode's tracks.
func TestSetModeWhilePaused(t *testing.T) {
	ctx := context.Background()
	r := newCaptureRig(t)
	id, err := r.m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
	if err != nil {
		t.Fatal(err)
	}
	if err := r.m.Pause(ctx, id); err != nil {
		t.Fatal(err)
	}
	var seen meeting.Snapshot
	cancel := r.m.Subscribe(func(s meeting.Snapshot) { seen = s })
	defer cancel()
	if err := r.m.SetMode(ctx, id, meeting.ModeRecording); err != nil {
		t.Fatal(err)
	}
	if seen.Mode != meeting.ModeRecording || seen.State != meeting.StatePaused {
		t.Fatalf("the snapshot after the change: mode %s, state %s", seen.Mode, seen.State)
	}
	if r.mic.running() != 0 || r.sys.running() != 0 {
		t.Fatalf("paused: %d microphone and %d system captures open", r.mic.running(), r.sys.running())
	}
	if err := r.m.Resume(ctx, id); err != nil {
		t.Fatal(err)
	}
	if r.mic.running() != 1 || r.sys.running() != 0 {
		t.Fatalf("resumed in Recording mode: %d microphone and %d system captures open", r.mic.running(), r.sys.running())
	}
}

// TestDiscardWhileRecording: Cancel meeting stops the captures before
// the meeting goes (left open, the overlay went on recording a meeting
// that was gone), deletes its row and files, and says so
// to the subscribers.
func TestDiscardWhileRecording(t *testing.T) {
	for _, paused := range []bool{false, true} {
		ctx := context.Background()
		r := newCaptureRig(t)
		id, err := r.m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
		if err != nil {
			t.Fatal(err)
		}
		if paused {
			if err := r.m.Pause(ctx, id); err != nil {
				t.Fatal(err)
			}
		}
		var seen meeting.Snapshot
		cancel := r.m.Subscribe(func(s meeting.Snapshot) { seen = s })
		if err := r.m.Discard(ctx, id); err != nil {
			t.Fatal(err)
		}
		cancel()
		if r.mic.running() != 0 || r.sys.running() != 0 {
			t.Errorf("paused %v: %d microphone and %d system captures still open", paused, r.mic.running(), r.sys.running())
		}
		if seen.State != meeting.StateIdle || seen.ID != "" || r.m.Current().State != meeting.StateIdle {
			t.Errorf("paused %v: the snapshot after Discard: state %s, id %q", paused, seen.State, seen.ID)
		}
		if _, _, err := r.meetings.Meeting(ctx, id); !errors.Is(err, core.ErrNotFound) {
			t.Errorf("paused %v: the meeting's row: %v, want not found", paused, err)
		}
		if _, err := os.Stat(filepath.Join(r.dir, id)); !os.IsNotExist(err) {
			t.Errorf("paused %v: the meeting's folder: %v, want gone", paused, err)
		}
		// The next meeting starts on its own captures.
		next, err := r.m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
		if err != nil {
			t.Fatal(err)
		}
		if err := r.m.Finish(ctx, next); err != nil {
			t.Fatal(err)
		}
	}
}

// TestDiscardWhileEnding: End pressed while Cancel waits for the captures
// to stop wins: its final pass has the meeting, whose row and files stay,
// and Discard says so (review of #4553: Discard lets go of the lock while
// they stop).
func TestDiscardWhileEnding(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	r := captureRig{mic: newFakeAudio(nil), sys: newFakeAudio(nil), meetings: db.Meetings(), dir: filepath.Join(root, "meetings")}
	speech := &seqFileASR{texts: []string{"hello"}}
	var ending atomic.Bool
	release := make(chan struct{})
	r.m = meeting.NewManager(meeting.Config{
		Mic: r.mic, System: r.sys, Store: r.meetings, Dir: r.dir,
		Now:       newFakeClock(time.Unix(1_700_000_000, 0)).Now,
		FreeSpace: func(string) (int64, error) { return meeting.DiskMinStartBytes() + 1, nil },
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			if ending.Load() { // the final pass: held until the test is done
				<-release
				return meeting.TrackASR{}, errors.New("released")
			}
			return meeting.TrackASR{Caps: speech.Caps(), File: speech}, nil
		},
	})
	id, err := r.m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
	if err != nil {
		t.Fatal(err)
	}
	var finishErr error
	r.mic.onStop = func() {
		ending.Store(true)
		finishErr = r.m.Finish(ctx, id)
	}
	if err := r.m.Discard(ctx, id); err == nil {
		t.Error("Discard after End: no error")
	}
	if finishErr != nil {
		t.Fatalf("Finish: %v", finishErr)
	}
	if got := r.m.Current(); got.State != meeting.StateFinishing || got.ID != id {
		t.Errorf("after both: state %s, id %q; want the meeting finishing", got.State, got.ID)
	}
	if _, _, err := r.meetings.Meeting(ctx, id); err != nil {
		t.Errorf("the ended meeting's row: %v", err)
	}
	if _, err := os.Stat(filepath.Join(r.dir, id)); err != nil {
		t.Errorf("the ended meeting's folder: %v", err)
	}
	if r.mic.running() != 0 || r.sys.running() != 0 {
		t.Errorf("%d microphone and %d system captures still open", r.mic.running(), r.sys.running())
	}
	ended := make(chan struct{})
	var once sync.Once
	stop := r.m.Subscribe(func(s meeting.Snapshot) {
		if s.State != meeting.StateFinishing {
			once.Do(func() { close(ended) })
		}
	})
	defer stop()
	close(release)
	select {
	case <-ended:
	case <-time.After(5 * time.Second):
		t.Fatal("the final pass did not end")
	}
}

// TestSetModePausedMeanwhile: a meeting paused while a mode change's new
// tracks start takes the new mode, as a paused one does (review of
// #4553: it used to fail with "not active" and keep the old mode).
func TestSetModePausedMeanwhile(t *testing.T) {
	ctx := context.Background()
	r := newCaptureRig(t)
	id, err := r.m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	meeting.SetModeStarted(r.m, func() {
		if err := r.m.Pause(ctx, id); err != nil {
			t.Errorf("Pause: %v", err)
		}
	})
	if err := r.m.SetMode(ctx, id, meeting.ModeMeeting); err != nil {
		t.Fatalf("SetMode: %v", err)
	}
	meeting.SetModeStarted(r.m, nil)
	if got := r.m.Current(); got.Mode != meeting.ModeMeeting || got.State != meeting.StatePaused {
		t.Fatalf("after the change: mode %s, state %s; want meeting, paused", got.Mode, got.State)
	}
	if r.mic.running() != 0 || r.sys.running() != 0 {
		t.Fatalf("paused: %d microphone and %d system captures open", r.mic.running(), r.sys.running())
	}
	if err := r.m.Resume(ctx, id); err != nil {
		t.Fatal(err)
	}
	if r.mic.running() != 1 || r.sys.running() != 1 {
		t.Fatalf("resumed in Meeting mode: %d microphone and %d system captures open", r.mic.running(), r.sys.running())
	}
}
