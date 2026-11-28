package meeting_test

import (
	"context"
	"encoding/binary"
	"path/filepath"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// The tracks' capture times reach the files: a pause pads the shorter
// file, the resume starts where the files end (not a whole pause later),
// and Finish leaves both files one length.
func TestManagerAlignsTracksAcrossPause(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	pcm := func(v int16, ms int) []byte {
		out := make([]byte, 2*ms*asr.SampleRate/1000)
		for i := 0; i < len(out); i += 2 {
			binary.LittleEndian.PutUint16(out[i:], uint16(v))
		}
		return out
	}
	mic, sys := newFakeAudio(pcm(1, 100)), newFakeAudio(pcm(5, 50))
	mic.hostMS = []float64{1000, 90_000}
	sys.hostMS = []float64{1000, 90_000}
	m := meeting.NewManager(meeting.Config{
		Mic: mic, System: sys, Store: s.Meetings(), Dir: filepath.Join(dir, "meetings"),
		Now:       newFakeClock(time.Unix(1_700_000_000, 0)).Now,
		Privacy:   func() settings.Privacy { return settings.Privacy{KeepHistory: true, KeepAudio: true} },
		FreeSpace: func(string) (int64, error) { return meeting.DiskMinStartBytes() + 1, nil },
	})
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Pause(ctx, id); err != nil {
		t.Fatal(err)
	}
	if err := m.Resume(ctx, id); err != nil {
		t.Fatal(err)
	}
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{"mic.wav": "1×200", "system.wav": "5×50 0×50 5×50 0×50"} {
		got, err := meeting.WAVRuns(filepath.Join(dir, "meetings", id, name))
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("%s: %q, want %q", name, got, want)
		}
	}
}
