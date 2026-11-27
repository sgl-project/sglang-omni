package meeting_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// TestReprocessKeepsOtherRecordings: a finished meeting whose recording
// was deleted has none to transcribe again, and asking for it must leave
// the other meetings' recordings alone. With the default settings the
// first pass deletes the recording, so the second one named the folder of
// every meeting's recordings and removed it.
func TestReprocessKeepsOtherRecordings(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	dir := filepath.Join(root, "meetings")
	for _, id := range []string{"talk", "memo"} {
		if err := os.MkdirAll(filepath.Join(dir, id), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	writeWAV(t, filepath.Join(dir, "talk"), "mic.wav", tonePCM(rateSamples(2*time.Second)))
	writeWAV(t, filepath.Join(dir, "talk"), "system.wav", tonePCM(rateSamples(time.Second)))
	memo := filepath.Join(dir, "memo", "mic.wav")
	writeWAV(t, filepath.Join(dir, "memo"), "mic.wav", tonePCM(rateSamples(time.Second)))
	meetings := db.Meetings()
	if _, err := meetings.CreateMeeting(ctx, store.Meeting{
		ID: "talk", CreatedAt: time.Now(), Status: store.MeetingInterrupted,
		Mode: store.MeetingModeMeeting, Audio: "talk", DurationMS: 2000,
	}); err != nil {
		t.Fatal(err)
	}
	// Recording mode keeps its recording whatever the settings say.
	if _, err := meetings.CreateMeeting(ctx, store.Meeting{
		ID: "memo", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeRecording, Audio: "memo",
	}); err != nil {
		t.Fatal(err)
	}
	speech := &seqFileASR{texts: []string{"mine", "theirs"}}
	mgr := meeting.NewManager(meeting.Config{
		Store:    meetings,
		Dir:      dir,
		Settings: func() settings.Meeting { return settings.Defaults().Features.Meeting },
		Privacy:  func() settings.Privacy { return settings.Defaults().Privacy },
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: speech.Caps(), File: speech}, nil
		},
	})
	if err := mgr.Reprocess(ctx, "talk"); err != nil {
		t.Fatal(err)
	}
	done, before, err := meetings.Meeting(ctx, "talk")
	if err != nil || done.Status != store.MeetingDone || done.Audio != "" || len(before) == 0 {
		t.Fatalf("after the first pass: status %s, audio %q, %d segments, err %v", done.Status, done.Audio, len(before), err)
	}
	if err := mgr.Reprocess(ctx, "talk"); !errors.Is(err, meeting.ErrNoRecording) {
		t.Fatalf("transcribing again without a recording: %v, want ErrNoRecording", err)
	}
	if _, err := os.Stat(memo); err != nil {
		t.Fatalf("the other meeting's recording: %v", err)
	}
	again, after, err := meetings.Meeting(ctx, "talk")
	if err != nil || again.Status != store.MeetingDone || len(after) != len(before) {
		t.Fatalf("the meeting changed: status %s, %d segments (was %d), err %v", again.Status, len(after), len(before), err)
	}
	// Continue processing: an interrupted meeting whose recording was swept
	// ends Done on its live transcript, and deletes nothing either.
	if _, err := meetings.CreateMeeting(ctx, store.Meeting{
		ID: "swept", CreatedAt: time.Now(), Status: store.MeetingInterrupted, Mode: store.MeetingModeMeeting,
	}); err != nil {
		t.Fatal(err)
	}
	if err := mgr.Reprocess(ctx, "swept"); err != nil {
		t.Fatal(err)
	}
	if swept, _, err := meetings.Meeting(ctx, "swept"); err != nil || swept.Status != store.MeetingDone {
		t.Fatalf("the swept meeting: status %s, err %v", swept.Status, err)
	}
	if _, err := os.Stat(memo); err != nil {
		t.Fatalf("the other meeting's recording after Continue: %v", err)
	}
}
