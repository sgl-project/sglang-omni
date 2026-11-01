package sqlitestore

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// seedRetention puts n entries of each kind, one a day back from t0
// (kind k's i-th is "<k>-<i>", i = 0 the newest), each with a recording.
func seedRetention(t *testing.T, h store.History, t0 time.Time, n int) {
	t.Helper()
	for _, k := range store.Kinds {
		for i := range n {
			id := fmt.Sprintf("%s-%d", k, i)
			e := entry(id, k, t0.Add(-time.Duration(i)*24*time.Hour), "the words of "+id, "result of "+id)
			e.Audio = id + ".wav"
			if err := os.WriteFile(filepath.Join(h.AudioDir(), e.Audio), []byte("RIFF"+id), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := h.Put(ctx, e); err != nil {
				t.Fatal(err)
			}
		}
	}
}

// historyIDs are the ids of every entry, sorted.
func historyIDs(t *testing.T, h store.History) []string {
	t.Helper()
	list, err := h.List(ctx, store.HistoryQuery{})
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, e := range list {
		ids = append(ids, e.ID)
	}
	slices.Sort(ids)
	return ids
}

// audioFiles are the files in dir, sorted.
func audioFiles(t *testing.T, dir string) []string {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		t.Fatal(err)
	}
	var out []string
	for _, e := range ents {
		out = append(out, e.Name())
	}
	return out
}

// ftsRows counts the full-text index's rows, and the rows matching q.
func ftsRows(t *testing.T, s *Store, q string) (all, match int) {
	t.Helper()
	rows, err := s.db.Query(ctx, `SELECT (SELECT COUNT(*) FROM history_fts), (SELECT COUNT(*) FROM history_fts WHERE history_fts MATCH ?)`, ftsPhrase(q))
	if err != nil {
		t.Fatal(err)
	}
	return int(asInt(rows.Values[0][0])), int(asInt(rows.Values[0][1]))
}

// TestPruneByAge: entries older than the cut go with their recordings;
// CountPrune said how many first; the index follows.
func TestPruneByAge(t *testing.T) {
	s := openTest(t, t.TempDir())
	h := s.History()
	t0 := time.Date(2026, 10, 17, 12, 0, 0, 0, time.UTC)
	seedRetention(t, h, t0, 5)                                           // days 0..4 back, 15 entries
	rt := store.Retention{Before: t0.Add(-2*24*time.Hour - time.Minute)} // keeps days 0, 1, 2
	if n, err := h.CountPrune(ctx, rt); n != 6 || err != nil {
		t.Fatalf("count %d %v", n, err)
	}
	if got := len(historyIDs(t, h)); got != 15 {
		t.Fatalf("CountPrune deleted: %d left", got)
	}
	if n, err := h.Prune(ctx, rt); n != 6 || err != nil {
		t.Fatalf("prune %d %v", n, err)
	}
	ids := historyIDs(t, h)
	if len(ids) != 9 || slices.Contains(ids, "transcribe-3") || !slices.Contains(ids, "rewrite-2") {
		t.Errorf("left %v", ids)
	}
	files := audioFiles(t, h.AudioDir())
	if len(files) != 9 || slices.Contains(files, "translate-4.wav") {
		t.Errorf("recordings left %v", files)
	}
	if all, m := ftsRows(t, s, "words of translate-4"); all != 9 || m != 0 {
		t.Errorf("index: %d rows, %d match a deleted entry", all, m)
	}
	if all, m := ftsRows(t, s, "words of translate-1"); all != 9 || m != 1 {
		t.Errorf("index lost a kept entry: %d %d", all, m)
	}
	if n, _ := h.CountPrune(ctx, rt); n != 0 {
		t.Errorf("after: %d more", n)
	}
	if n, err := h.Prune(ctx, store.Retention{}); n != 0 || err != nil {
		t.Errorf("zero retention: %d %v", n, err)
	}
}

// TestPruneByCount: each kind keeps its newest N; age and count together
// drop what either drops.
func TestPruneByCount(t *testing.T) {
	s := openTest(t, t.TempDir())
	h := s.History()
	t0 := time.Date(2026, 10, 17, 12, 0, 0, 0, time.UTC)
	seedRetention(t, h, t0, 4)
	// One more transcription, the oldest: transcribe keeps 3 of 5, the
	// others 3 of 4.
	old := entry("transcribe-old", store.KindTranscribe, t0.Add(-30*24*time.Hour), "old", "old")
	h.Put(ctx, old)
	rt := store.Retention{PerKind: 3}
	if n, _ := h.CountPrune(ctx, rt); n != 4 {
		t.Fatalf("count %d", n)
	}
	if n, err := h.Prune(ctx, rt); n != 4 || err != nil {
		t.Fatalf("prune %d %v", n, err)
	}
	want := []string{"rewrite-0", "rewrite-1", "rewrite-2", "transcribe-0", "transcribe-1", "transcribe-2", "translate-0", "translate-1", "translate-2"}
	if got := historyIDs(t, h); !slices.Equal(got, want) {
		t.Errorf("left %v", got)
	}
	if files := audioFiles(t, h.AudioDir()); len(files) != 9 {
		t.Errorf("recordings %v", files)
	}
	if all, _ := ftsRows(t, s, "x"); all != 9 {
		t.Errorf("index has %d rows", all)
	}
	both := store.Retention{PerKind: 2, Before: t0.Add(-12 * time.Hour)}
	if n, _ := h.Prune(ctx, both); n != 6 {
		t.Errorf("age and count: %d", n)
	}
	if got := strings.Join(historyIDs(t, h), ","); got != "rewrite-0,transcribe-0,translate-0" {
		t.Errorf("left %s", got)
	}
}

// TestAudioDir: recordings go to the folder set, are found in the
// store's own when kept there, and go from both; DeleteAudio keeps the
// entries.
func TestAudioDir(t *testing.T) {
	s := openTest(t, t.TempDir())
	h := s.History()
	def := h.DefaultAudioDir()
	if h.AudioDir() != def {
		t.Fatalf("audio dir %s", h.AudioDir())
	}
	other := t.TempDir()
	h.SetAudioDir(other)
	if h.AudioDir() != other {
		t.Fatalf("set: %s", h.AudioDir())
	}
	for _, f := range []struct{ dir, name string }{{other, "a.wav"}, {def, "b.wav"}} {
		os.WriteFile(filepath.Join(f.dir, f.name), []byte("RIFF"), 0o644)
		e := entry(strings.TrimSuffix(f.name, ".wav"), store.KindTranscribe, time.Now(), "x", "y")
		e.Audio = f.name
		h.Put(ctx, e)
	}
	h.Put(ctx, entry("c", store.KindTranscribe, time.Now(), "no audio", "z"))
	if p := h.AudioPath("a.wav"); p != filepath.Join(other, "a.wav") {
		t.Errorf("a: %s", p)
	}
	if p := h.AudioPath("b.wav"); p != filepath.Join(def, "b.wav") {
		t.Errorf("b, kept in the store's folder: %s", p)
	}
	if p := h.AudioPath("gone.wav"); p != filepath.Join(other, "gone.wav") {
		t.Errorf("missing: %s", p)
	}
	if list, _ := h.List(ctx, store.HistoryQuery{WithAudio: true}); len(list) != 2 {
		t.Errorf("with audio: %d", len(list))
	}
	if n, err := h.DeleteAudio(ctx); n != 2 || err != nil {
		t.Fatalf("delete audio: %d %v", n, err)
	}
	if len(audioFiles(t, other))+len(audioFiles(t, def)) != 0 {
		t.Errorf("files left: %v %v", audioFiles(t, other), audioFiles(t, def))
	}
	if got := historyIDs(t, h); len(got) != 3 {
		t.Errorf("entries %v", got)
	}
	if e, _ := h.Get(ctx, "a"); e.Audio != "" || e.Result != "y" {
		t.Errorf("entry a %+v", e)
	}
	h.SetAudioDir(def)
	if h.AudioDir() != def {
		t.Errorf("back: %s", h.AudioDir())
	}
}
