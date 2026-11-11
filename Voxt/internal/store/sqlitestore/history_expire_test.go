package sqlitestore

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
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

// TestListBoundsAndOrder: HistoryQuery's After (inclusive), AfterID
// (strictly past that row), Before (exclusive), Oldest and Finished;
// Count agrees; the zero query stays newest first.
func TestListBoundsAndOrder(t *testing.T) {
	s := openTest(t, t.TempDir())
	h := s.History()
	base := time.Now().Truncate(time.Millisecond)
	put := func(id string, offset time.Duration, st store.Status, result string) {
		e := entry(id, store.KindTranscribe, base.Add(offset), "", "")
		e.Status, e.Result = st, result
		if err := h.Put(ctx, e); err != nil {
			t.Fatal(err)
		}
	}
	put("a", 0, store.StatusDone, "x")
	put("b", time.Second, store.StatusDone, "x")
	put("c", time.Second, store.StatusDone, "x") // ties with b on time
	put("d", 2*time.Second, store.StatusDone, "x")
	put("fail", 3*time.Second, store.StatusFailed, "x")
	put("empty", 4*time.Second, store.StatusDone, "")
	ids := func(q store.HistoryQuery) []string {
		t.Helper()
		list, err := h.List(ctx, q)
		if err != nil {
			t.Fatal(err)
		}
		out := []string{}
		for _, e := range list {
			out = append(out, e.ID)
		}
		n, err := h.Count(ctx, store.HistoryQuery{Finished: q.Finished, After: q.After, AfterID: q.AfterID, Before: q.Before})
		if q.Limit == 0 && (err != nil || n != len(list)) {
			t.Errorf("Count %d, %v for %d rows", n, err, len(list))
		}
		return out
	}
	for name, c := range map[string]struct {
		q    store.HistoryQuery
		want []string
	}{
		"zero, newest first": {store.HistoryQuery{}, []string{"empty", "fail", "d", "c", "b", "a"}},
		"oldest":             {store.HistoryQuery{Oldest: true}, []string{"a", "b", "c", "d", "fail", "empty"}},
		"finished":           {store.HistoryQuery{Finished: true, Oldest: true}, []string{"a", "b", "c", "d"}},
		"after inclusive":    {store.HistoryQuery{After: base.Add(time.Second), Finished: true, Oldest: true}, []string{"b", "c", "d"}},
		"before exclusive":   {store.HistoryQuery{Before: base.Add(time.Second), Oldest: true}, []string{"a"}},
		"past a tied row":    {store.HistoryQuery{After: base.Add(time.Second), AfterID: "b", Finished: true, Oldest: true}, []string{"c", "d"}},
		"page":               {store.HistoryQuery{Finished: true, Oldest: true, Limit: 2}, []string{"a", "b"}},
	} {
		if got := ids(c.q); !slices.Equal(got, c.want) {
			t.Errorf("%s: %v, want %v", name, got, c.want)
		}
	}
}
