package meeting_test

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// TestManagerStoresNoLoop: a speech model that loops on a phrase leaves it
// once in the meeting's stored segments (#3981).
func TestManagerStoresNoLoop(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	streamer := &meetTestStreamer{text: "Charlie, " + strings.Repeat("você não quer esse Charlie, ", 300)}
	m := meeting.NewManager(meeting.Config{
		Mic:     newFakeAudio(liveTestPCM(t)),
		Store:   s.Meetings(),
		Dir:     filepath.Join(dir, "meetings"),
		Privacy: func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) {
			return meeting.DiskMinStartBytes() + 1, nil
		},
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: streamer.Caps(), Streamer: streamer}, nil
		},
	})
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	for deadline := time.Now().Add(10 * time.Second); len(m.Current().Segments) == 0; time.Sleep(5 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("no segment from the microphone")
		}
	}
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	_, segs, err := s.Meetings().Meeting(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) == 0 {
		t.Fatal("no stored segments")
	}
	for _, g := range segs {
		if g.Text != "Charlie, você não quer esse Charlie," {
			t.Errorf("stored segment %.80q…, want the phrase once", g.Text)
		}
	}
}
