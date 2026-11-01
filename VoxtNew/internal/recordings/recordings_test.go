package recordings

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/datadir"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store/sqlitestore"
)

var ctx = context.Background()

// openHistory is a store's history in a new folder, with n entries, the
// first n-1 with a recording ("e<i>.wav") and the last without.
func openHistory(t *testing.T, n int) store.History {
	t.Helper()
	s, err := sqlitestore.Open(ctx, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	h := s.History()
	t0 := time.Date(2026, 10, 17, 14, 30, 5, 0, time.Local)
	for i := range n {
		e := store.Entry{ID: fmt.Sprintf("e%d", i), Kind: store.KindTranscribe, CreatedAt: t0.Add(-time.Duration(i) * time.Minute), Result: fmt.Sprintf("result %d", i)}
		if i < n-1 {
			e.Audio = e.ID + ".wav"
			put(t, h, h.AudioDir(), e.Audio, "RIFF "+e.ID)
		}
		if err := h.Put(ctx, e); err != nil {
			t.Fatal(err)
		}
	}
	return h
}

func put(t *testing.T, h store.History, dir, name, data string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

func files(t *testing.T, dir string) []string {
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

// TestUsage counts the recordings the entries name that are there.
func TestUsage(t *testing.T) {
	h := openHistory(t, 4)
	os.Remove(filepath.Join(h.AudioDir(), "e2.wav"))
	put(t, h, h.AudioDir(), "orphan.wav", "not an entry's")
	n, bytes, err := Usage(ctx, h)
	if n != 2 || bytes != int64(len("RIFF e0")*2) || err != nil {
		t.Errorf("usage %d %d %v", n, bytes, err)
	}
}

// TestMove: the recordings move, checked, the folder switches in
// commit, a recording kept meanwhile follows, the originals go; a
// missing one is counted.
func TestMove(t *testing.T) {
	h := openHistory(t, 5)
	from := h.AudioDir()
	os.Remove(filepath.Join(from, "e3.wav")) // deleted by hand
	to := filepath.Join(t.TempDir(), "Recordings")
	var steps []int
	commit := func() error {
		// A session kept a recording in the old folder while it copied.
		put(t, h, from, "late.wav", "RIFF late")
		h.Put(ctx, store.Entry{ID: "late", Kind: store.KindRewrite, CreatedAt: time.Now(), Audio: "late.wav"})
		h.SetAudioDir(to)
		return nil
	}
	res, err := Move(ctx, h, to, commit, func(done, total int) { steps = append(steps, done) })
	if err != nil {
		t.Fatal(err)
	}
	if res.Moved != 4 || res.Missing != 1 {
		t.Errorf("result %+v", res)
	}
	if got := files(t, to); !slices.Equal(got, []string{"e0.wav", "e1.wav", "e2.wav", "late.wav"}) {
		t.Errorf("moved %v", got)
	}
	if got := files(t, from); len(got) != 0 {
		t.Errorf("left in the old folder %v", got)
	}
	if data, _ := os.ReadFile(h.AudioPath("e1.wav")); string(data) != "RIFF e1" || h.AudioPath("e1.wav") != filepath.Join(to, "e1.wav") {
		t.Errorf("e1 at %s: %q", h.AudioPath("e1.wav"), data)
	}
	if len(steps) == 0 || steps[len(steps)-1] != 4 { // the 4 with a recording
		t.Errorf("progress %v", steps)
	}
	// Back to the store's own folder.
	if _, err := Move(ctx, h, h.DefaultAudioDir(), func() error { h.SetAudioDir(""); return nil }, nil); err != nil {
		t.Fatal(err)
	}
	if len(files(t, to)) != 0 || len(files(t, h.DefaultAudioDir())) != 4 {
		t.Errorf("back: %v %v", files(t, to), files(t, h.DefaultAudioDir()))
	}
	if err := CheckTarget(from, from+"/sub"); err == nil {
		t.Error("a folder inside the current one")
	}
	if _, err := Move(ctx, h, h.AudioDir(), commit, nil); err == nil {
		t.Error("moved to the folder in use")
	}
}

// TestMoveFails: a failed copy partway, a cancel, a failed commit and a
// file in the way each leave the recordings where they were, in use, and
// no copy behind.
func TestMoveFails(t *testing.T) {
	h := openHistory(t, 6)
	from := h.AudioDir()
	before := files(t, from)
	n := 0
	copyFile = func(src, dst string) (int64, error) {
		if n++; n == 3 {
			os.WriteFile(dst, []byte("half"), 0o644) // a partial copy
			return 4, errors.New("disk full")
		}
		return datadir.CopyVerified(src, dst)
	}
	t.Cleanup(func() { copyFile = datadir.CopyVerified })
	commits := 0
	commit := func() error { commits++; h.SetAudioDir(""); return nil }
	check := func(what string, to string) {
		t.Helper()
		if got := files(t, from); !slices.Equal(got, before) {
			t.Errorf("%s: the old folder has %v", what, got)
		}
		if got := files(t, to); len(got) != 0 {
			t.Errorf("%s: left behind %v", what, got)
		}
		if h.AudioDir() != from {
			t.Errorf("%s: in use %s", what, h.AudioDir())
		}
		if commits != 0 {
			t.Errorf("%s: committed", what)
		}
		if data, _ := os.ReadFile(h.AudioPath("e2.wav")); string(data) != "RIFF e2" {
			t.Errorf("%s: e2 reads %q", what, data)
		}
	}
	to := filepath.Join(t.TempDir(), "a")
	if _, err := Move(ctx, h, to, commit, nil); err == nil || err.Error() != "disk full" {
		t.Errorf("copy failure: %v", err)
	}
	check("copy failure", to)

	copyFile = datadir.CopyVerified
	cctx, cancel := context.WithCancel(ctx)
	if _, err := Move(cctx, h, to, commit, func(done, _ int) {
		if done == 2 {
			cancel()
		}
	}); !errors.Is(err, context.Canceled) {
		t.Errorf("cancel: %v", err)
	}
	check("cancel", to)

	if _, err := Move(ctx, h, to, func() error { return errors.New("settings not saved") }, nil); err == nil {
		t.Error("commit failure: no error")
	}
	check("commit failure", to)

	put(t, h, to, "e1.wav", "someone else's") // in the way
	if _, err := Move(ctx, h, to, commit, nil); err == nil {
		t.Error("a different file in the way: no error")
	}
	os.Remove(filepath.Join(to, "e1.wav"))
	check("a file in the way", to)
}

// TestExportName: the feature's folder, the local time, the first words
// sanitised (Voxt's letters, digits and "-"), Chinese kept, a fallback.
func TestExportName(t *testing.T) {
	at := time.Date(2026, 10, 17, 14, 30, 5, 0, time.Local)
	for _, c := range []struct {
		e    store.Entry
		want string
	}{
		{store.Entry{Kind: store.KindTranscribe, CreatedAt: at, Result: "Let's move the design review to Thursday afternoon.\nI'll send it."},
			"transcription/20261017-143005-Lets-move-the-design-review-to.wav"},
		{store.Entry{Kind: store.KindTranslate, CreatedAt: at, Result: "明天上午十点的站会改到线上，链接我稍后发到群里。"},
			"translation/20261017-143005-明天上午十点的站会改到线上-链接我稍后发到群里.wav"},
		{store.Entry{Kind: store.KindRewrite, CreatedAt: at, PromptText: "  make it / more: formal?  "},
			"rewrite/20261017-143005-make-it-more-formal.wav"},
		{store.Entry{Kind: store.KindTranscribe, CreatedAt: at, Status: store.StatusFailed, RawText: "../../etc/passwd"},
			"transcription/20261017-143005-etc-passwd.wav"},
		{store.Entry{Kind: store.KindTranscribe, CreatedAt: at, Result: "…!?"},
			"transcription/20261017-143005-transcription.wav"},
		{store.Entry{Kind: store.KindTranscribe, CreatedAt: at, Result: "Supercalifragilisticexpialidociousandmore words"},
			"transcription/20261017-143005-Supercalifragilisticexpialidocio.wav"},
	} {
		if got := ExportName(c.e); got != c.want {
			t.Errorf("%q:\n got %s\nwant %s", c.e.Result+c.e.PromptText+c.e.RawText, got, c.want)
		}
	}
}

// TestExport copies under the names, skips the missing, numbers a name
// taken, and leaves a recording exported before as it is.
func TestExport(t *testing.T) {
	h := openHistory(t, 4)
	list, _ := h.List(ctx, store.HistoryQuery{})
	// Two entries with the same words at the same second.
	twin := list[1]
	twin.ID, twin.Audio = "twin", "twin.wav"
	put(t, h, h.AudioDir(), "twin.wav", "RIFF twin")
	h.Put(ctx, twin)
	list = append(list, twin)
	list[0].Result, list[2].Audio = list[1].Result, "gone.wav"
	list[0].CreatedAt = list[1].CreatedAt
	out := t.TempDir()
	res, err := Export(ctx, h, list, out, nil)
	if err != nil || res != (ExportResult{Exported: 3, Skipped: 1}) {
		t.Fatalf("export %+v %v", res, err)
	}
	got := files(t, filepath.Join(out, "transcription"))
	want := []string{"20261017-142905-result-1-2.wav", "20261017-142905-result-1-3.wav", "20261017-142905-result-1.wav"}
	if !slices.Equal(got, want) {
		t.Errorf("files %v", got)
	}
	// Again: the same files, no new numbers.
	if res, _ := Export(ctx, h, list, out, nil); res.Exported != 3 || len(files(t, filepath.Join(out, "transcription"))) != 3 {
		t.Errorf("again: %+v %v", res, files(t, filepath.Join(out, "transcription")))
	}
}
