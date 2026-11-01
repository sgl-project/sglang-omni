package sqlitestore

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// TestExpireAudio: a recording kept only for a retry (AudioUntil) goes
// once its time is up, file and all, and the entry stays; one kept for
// good, or not yet due, stays. AudioUntil round-trips, and DeleteAudio
// clears it.
func TestExpireAudio(t *testing.T) {
	s := openTest(t, t.TempDir())
	h := s.History()
	now := time.Now().Truncate(time.Millisecond)
	for _, c := range []struct {
		id    string
		until time.Time
	}{{"due", now.Add(-time.Minute)}, {"later", now.Add(time.Hour)}, {"kept", time.Time{}}} {
		e := entry(c.id, store.KindTranscribe, now, "", "")
		e.Status, e.Audio, e.AudioUntil = store.StatusFailed, c.id+".wav", c.until
		if err := os.WriteFile(filepath.Join(h.AudioDir(), e.Audio), []byte("RIFF"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := h.Put(ctx, e); err != nil {
			t.Fatal(err)
		}
	}
	if e, _ := h.Get(ctx, "later"); !e.AudioUntil.Equal(now.Add(time.Hour)) {
		t.Fatalf("AudioUntil read back as %v", e.AudioUntil)
	}
	n, err := h.ExpireAudio(ctx, now)
	if err != nil || n != 1 {
		t.Fatalf("expired %d, %v", n, err)
	}
	if got := audioFiles(t, h.AudioDir()); !slices.Equal(got, []string{"kept.wav", "later.wav"}) {
		t.Errorf("files %q", got)
	}
	if e, err := h.Get(ctx, "due"); err != nil || e.Audio != "" || !e.AudioUntil.IsZero() {
		t.Errorf("due %+v %v", e, err)
	}
	if n, _ := h.ExpireAudio(ctx, now); n != 0 {
		t.Errorf("expired again: %d", n)
	}
	if _, err := h.DeleteAudio(ctx); err != nil {
		t.Fatal(err)
	}
	if e, _ := h.Get(ctx, "later"); e.Audio != "" || !e.AudioUntil.IsZero() {
		t.Errorf("after DeleteAudio %+v", e)
	}
}
