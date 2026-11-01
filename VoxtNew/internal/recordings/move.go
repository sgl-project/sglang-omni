// Package recordings is what the History settings dialog does with the
// history's recordings as files: how much room they take, moving them to
// another folder, and exporting them under readable names. The entries
// name their recording by file name only (store.Entry.Audio), so a move
// changes no row: it copies, checks, switches the folder
// (store.History.SetAudioDir and the setting, through commit) and only
// then deletes the originals.
//
// Imports: store, datadir.
package recordings

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/datadir"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// copyFile copies a file and checks the copy (a test seam: a failure on
// the n-th file).
var copyFile = datadir.CopyVerified

// Usage is how many recordings the history keeps and their size.
func Usage(ctx context.Context, hs store.History) (files int, bytes int64, err error) {
	list, err := hs.List(ctx, store.HistoryQuery{WithAudio: true})
	if err != nil {
		return 0, 0, err
	}
	for _, e := range list {
		if fi, err := os.Stat(hs.AudioPath(e.Audio)); err == nil && fi.Mode().IsRegular() {
			files++
			bytes += fi.Size()
		}
	}
	return files, bytes, nil
}

// MoveResult is what a move did: the recordings moved, and those an
// entry names that were not found (deleted by hand, or on a drive that
// is not connected).
type MoveResult struct {
	Moved, Missing int
}

// CheckTarget refuses to move the recordings from the folder from to to:
// the same folder, or one inside it.
func CheckTarget(from, to string) error {
	from, to = filepath.Clean(from), filepath.Clean(to)
	if from == to {
		return errors.New("the recordings are there already")
	}
	if rel, err := filepath.Rel(from, to); err == nil && rel != "." && !strings.HasPrefix(rel, "..") {
		return errors.New("the new folder cannot be inside the current one")
	}
	return nil
}

// Move moves every recording an entry names to the folder to. It copies
// each one there and checks the copy, then calls commit, which makes to
// the folder in use (hs.AudioDir() must be to after it), copies what was
// kept in the old folder meanwhile, and only then deletes the originals.
// A failure or a cancel before commit deletes the copies it made and
// leaves everything as it was; after commit, an original that could not
// be deleted stays (a file left over, never an entry without its
// recording). progress gets the recordings done and their number.
func Move(ctx context.Context, hs store.History, to string, commit func() error, progress func(done, total int)) (MoveResult, error) {
	var res MoveResult
	from, def := hs.AudioDir(), hs.DefaultAudioDir()
	if err := CheckTarget(from, to); err != nil {
		return res, err
	}
	if err := os.MkdirAll(to, 0o755); err != nil {
		return res, err
	}
	list, err := hs.List(ctx, store.HistoryQuery{WithAudio: true})
	if err != nil {
		return res, err
	}
	var made []string      // the copies made, deleted on a rollback
	var originals []string // the files copied, deleted at the end
	done := map[string]bool{}
	rollback := func(err error) (MoveResult, error) {
		for _, p := range made {
			os.Remove(p)
		}
		return MoveResult{}, err
	}
	// place copies a recording from src into to.
	place := func(name, src string) error {
		dst := filepath.Join(to, name)
		if fi, err := os.Stat(dst); err == nil {
			if !fi.Mode().IsRegular() || !sameFile(src, dst) {
				return fmt.Errorf("%s holds another %s already", to, name)
			}
		} else {
			tmp := dst + ".part"
			if _, err := copyFile(src, tmp); err != nil {
				os.Remove(tmp)
				return err
			}
			if err := os.Rename(tmp, dst); err != nil {
				os.Remove(tmp)
				return err
			}
			made = append(made, dst)
		}
		originals = append(originals, src)
		done[name] = true
		return nil
	}
	for i, e := range list {
		if err := ctx.Err(); err != nil {
			return rollback(err)
		}
		if progress != nil {
			progress(i, len(list))
		}
		if done[e.Audio] || filepath.Base(e.Audio) != e.Audio {
			continue
		}
		src := hs.AudioPath(e.Audio)
		if _, err := os.Stat(src); err != nil {
			res.Missing++
			continue
		}
		if err := place(e.Audio, src); err != nil {
			return rollback(err)
		}
		res.Moved++
	}
	if progress != nil {
		progress(len(list), len(list))
	}
	if err := ctx.Err(); err != nil {
		return rollback(err)
	}
	if err := commit(); err != nil {
		return rollback(err)
	}
	if got := hs.AudioDir(); filepath.Clean(got) != filepath.Clean(to) {
		// Not switched after all: keep both copies.
		return res, fmt.Errorf("the recordings folder is %s, not %s", got, to)
	}
	// A recording kept while the copies were made went to the old folder.
	if later, err := hs.List(context.WithoutCancel(ctx), store.HistoryQuery{WithAudio: true}); err == nil {
		for _, e := range later {
			if done[e.Audio] || filepath.Base(e.Audio) != e.Audio {
				continue
			}
			for _, dir := range []string{from, def} {
				src := filepath.Join(dir, e.Audio)
				if _, err := os.Stat(src); err != nil {
					continue
				}
				if err := place(e.Audio, src); err != nil {
					log.Printf("recordings: %s stays in %s: %v", e.Audio, dir, err)
				} else {
					res.Moved++
				}
				break
			}
		}
	}
	for _, p := range originals {
		if err := os.Remove(p); err != nil && !errors.Is(err, fs.ErrNotExist) {
			log.Printf("recordings: the original %s stays: %v", p, err)
		}
	}
	return res, nil
}

// sameFile reports whether two files hold the same bytes.
func sameFile(a, b string) bool {
	fa, err1 := os.Stat(a)
	fb, err2 := os.Stat(b)
	if err1 != nil || err2 != nil || fa.Size() != fb.Size() {
		return false
	}
	da, err1 := os.ReadFile(a)
	db, err2 := os.ReadFile(b)
	return err1 == nil && err2 == nil && string(da) == string(db)
}
