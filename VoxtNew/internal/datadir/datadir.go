// Package datadir is where VoxtNew's data folder is (from AudioSmith): the
// default one (mygo.PathUserData), or another the user moved it to in
// Settings → Storage, which a small file in the default folder points at
// (location.json). Moving and erasing finish at the next start, before
// the library opens:
//
//   - Move: the app copies every file but the database while it runs
//     (and checks the copies), records the move (move.json) and restarts.
//     At the start, with the database closed, it copies what changed
//     since, then the database, rewrites the paths the database holds to
//     the new folder, checks it, and only then switches location.json.
//     Any failure removes the copy and keeps the old folder (rollback).
//   - Erase everything: the app records it (reset.json) and restarts; the
//     start deletes the data folder, the caches and the logs.
//
// It is pure Go (files and the database's paths): package main calls
// Resolve before the library opens, and Settings → Storage the rest.
package datadir

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store/sqlitestore"
)

// Files of the default data folder that are about the folder itself.
const (
	LocationFile = "location.json"
	MoveFile     = "move.json"
	ResetFile    = "reset.json"
)

// Location is location.json.
type Location struct {
	// Dir is the data folder in use; "" = the default one.
	Dir string `json:"dir,omitempty"`
	// Previous is the folder the data was moved from, kept until the
	// user removes it (Settings → Storage).
	Previous string    `json:"previous,omitempty"`
	MovedAt  time.Time `json:"movedAt,omitzero"`
	// Failed says why the last move did not happen.
	Failed string `json:"failed,omitempty"`
}

// PendingMove is move.json.
type PendingMove struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// PendingReset is reset.json.
type PendingReset struct {
	Everything bool     `json:"everything"`
	Also       []string `json:"also,omitempty"` // other folders to empty: caches, logs
}

func readJSON(path string, v any) bool {
	data, err := os.ReadFile(path)
	return err == nil && json.Unmarshal(data, v) == nil
}

// WriteJSON writes v to path as JSON, through a temporary file.
func WriteJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ReadLocation reads location.json of the default folder def.
func ReadLocation(def string) Location {
	var loc Location
	readJSON(filepath.Join(def, LocationFile), &loc)
	return loc
}

// Resolve finishes a pending erase or move, then returns the data
// folder to open. A folder location.json names that is missing (an
// external drive not connected) is an error: the app explains and quits
// rather than start an empty library.
func Resolve(def string) (string, error) {
	var rs PendingReset
	if readJSON(filepath.Join(def, ResetFile), &rs) {
		eraseEverything(def, rs)
	}
	var mv PendingMove
	if readJSON(filepath.Join(def, MoveFile), &mv) {
		finishMove(context.Background(), def, mv)
	}
	loc := ReadLocation(def)
	if loc.Dir == "" {
		return def, nil
	}
	if fi, err := os.Stat(loc.Dir); err != nil || !fi.IsDir() {
		return "", fmt.Errorf("the data folder %s (chosen in Settings → Storage) is not available: connect its drive, or remove %s to start over in %s",
			loc.Dir, filepath.Join(def, LocationFile), def)
	}
	return loc.Dir, nil
}

// SkipData reports the files a move or a size count leaves out: the
// database (moved at the start), the location files, transient ones.
func SkipData(rel string, db bool) bool {
	top := strings.SplitN(filepath.ToSlash(rel), "/", 2)[0]
	switch top {
	case LocationFile, MoveFile, ResetFile, LocationFile + ".tmp", MoveFile + ".tmp":
		return true
	}
	return db && strings.HasPrefix(top, sqlitestore.FileName)
}

// CopyTree copies src into dst, which must not hold anything yet, but
// the files skip names; it hashes what it reads and checks what it
// wrote. progress gets the bytes copied and the total.
func CopyTree(ctx context.Context, src, dst string, skip func(rel string) bool, progress func(done, total int64)) error {
	total, err := TreeSize(ctx, src, skip)
	if err != nil {
		return err
	}
	var done int64
	err = filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		rel, _ := filepath.Rel(src, path)
		if rel == "." {
			return os.MkdirAll(dst, 0o755)
		}
		if skip(rel) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		target := filepath.Join(dst, rel)
		switch {
		case d.IsDir():
			return os.MkdirAll(target, 0o755)
		case d.Type()&fs.ModeSymlink != 0:
			link, err := os.Readlink(path)
			if err != nil {
				return err
			}
			return os.Symlink(link, target)
		case !d.Type().IsRegular():
			return nil
		}
		n, err := copyVerified(path, target)
		done += n
		if progress != nil {
			progress(done, total)
		}
		return err
	})
	return err
}

// CopyVerified copies one file, keeping its mode and time, and checks the
// copy against the original's SHA-256; it reports the bytes copied (the
// History settings' recordings move and export use it too).
func CopyVerified(src, dst string) (int64, error) { return copyVerified(src, dst) }

func copyVerified(src, dst string) (int64, error) {
	in, err := os.Open(src)
	if err != nil {
		return 0, err
	}
	defer in.Close()
	fi, err := in.Stat()
	if err != nil {
		return 0, err
	}
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, fi.Mode().Perm())
	if err != nil {
		return 0, err
	}
	h := sha256.New()
	n, err := io.Copy(out, io.TeeReader(in, h))
	if err == nil {
		err = out.Sync()
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return n, err
	}
	os.Chtimes(dst, fi.ModTime(), fi.ModTime())
	want := h.Sum(nil)
	got, err := fileHash(dst)
	if err != nil {
		return n, err
	}
	if string(got) != string(want) {
		return n, fmt.Errorf("%s: the copy differs from the original", dst)
	}
	return n, nil
}

func fileHash(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return nil, err
	}
	return h.Sum(nil), nil
}

// syncTree copies the files of src that dst lacks or has with another
// size or time (what changed since CopyTree), and removes what dst has
// that src no longer does.
func syncTree(src, dst string, skip func(rel string) bool) error {
	err := filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, path)
		if rel == "." {
			return nil
		}
		if skip(rel) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		if !d.Type().IsRegular() {
			return nil
		}
		fi, err := d.Info()
		if err != nil {
			return err
		}
		if ti, err := os.Stat(target); err == nil && ti.Size() == fi.Size() && ti.ModTime().Equal(fi.ModTime()) {
			return nil
		}
		_, err = copyVerified(path, target)
		return err
	})
	if err != nil {
		return err
	}
	// Files deleted since the copy.
	var gone []string
	filepath.WalkDir(dst, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(dst, path)
		if skip(rel) {
			return nil
		}
		if _, err := os.Lstat(filepath.Join(src, rel)); errors.Is(err, fs.ErrNotExist) {
			gone = append(gone, path)
		}
		return nil
	})
	for _, p := range gone {
		os.Remove(p)
	}
	return nil
}

// TreeSize adds up the regular files under dir (skip may be nil).
func TreeSize(ctx context.Context, dir string, skip func(rel string) bool) (int64, error) {
	var n int64
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) || errors.Is(err, fs.ErrPermission) {
				return nil
			}
			return err
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if skip != nil {
			if rel, _ := filepath.Rel(dir, path); rel != "." && skip(rel) {
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
		}
		if d.Type().IsRegular() {
			if fi, err := d.Info(); err == nil {
				n += fi.Size()
			}
		}
		return nil
	})
	return n, err
}

// CheckMoveTarget refuses a destination that is the folder itself, inside
// it, or holds files already.
func CheckMoveTarget(from, to string) error {
	from, to = filepath.Clean(from), filepath.Clean(to)
	if from == to {
		return errors.New("that is the folder in use")
	}
	if rel, err := filepath.Rel(from, to); err == nil && !strings.HasPrefix(rel, "..") {
		return errors.New("the new folder cannot be inside the current one")
	}
	if ents, err := os.ReadDir(to); err == nil {
		for _, e := range ents {
			if !SkipData(e.Name(), false) && e.Name() != ".DS_Store" {
				return fmt.Errorf("%s is not empty", to)
			}
		}
	}
	return nil
}

// finishMove completes a move at the start (the database is closed).
func finishMove(ctx context.Context, def string, mv PendingMove) {
	os.Remove(filepath.Join(def, MoveFile))
	loc := ReadLocation(def)
	fail := func(err error) {
		log.Printf("data folder: the move to %s failed, keeping %s: %v", mv.To, mv.From, err)
		if mv.To != def {
			os.RemoveAll(mv.To)
		} else {
			RemoveExcept(def)
		}
		loc.Failed = err.Error()
		WriteJSON(filepath.Join(def, LocationFile), loc)
	}
	dbOnly := func(rel string) bool { return SkipData(rel, true) }
	if err := syncTree(mv.From, mv.To, dbOnly); err != nil {
		fail(err)
		return
	}
	for _, suffix := range []string{"", "-wal"} {
		src := filepath.Join(mv.From, sqlitestore.FileName+suffix)
		if _, err := os.Stat(src); err != nil {
			continue
		}
		if _, err := copyVerified(src, filepath.Join(mv.To, sqlitestore.FileName+suffix)); err != nil {
			fail(err)
			return
		}
	}
	db := filepath.Join(mv.To, sqlitestore.FileName)
	if _, err := sqlitestore.RewritePaths(ctx, db, mv.From, mv.To); err != nil {
		fail(fmt.Errorf("updating the library's paths: %w", err))
		return
	}
	if _, err := sqlitestore.CheckFile(ctx, db); err != nil {
		fail(err)
		return
	}
	os.Remove(filepath.Join(mv.To, sqlitestore.FileName+"-wal"))
	loc = Location{Dir: mv.To, Previous: mv.From, MovedAt: time.Now()}
	if mv.To == def {
		loc.Dir = ""
	}
	if err := WriteJSON(filepath.Join(def, LocationFile), loc); err != nil {
		fail(err)
		return
	}
	log.Printf("data folder: moved from %s to %s", mv.From, mv.To)
}

// RemoveExcept empties a data folder but for its location files.
func RemoveExcept(dir string) error {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	var first error
	for _, e := range ents {
		switch e.Name() {
		case LocationFile, MoveFile, ResetFile:
			continue
		}
		if err := os.RemoveAll(filepath.Join(dir, e.Name())); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// eraseEverything deletes the data (wherever it is), and the folders rs
// names (caches, logs), at the start.
func eraseEverything(def string, rs PendingReset) {
	loc := ReadLocation(def)
	if loc.Dir != "" {
		os.RemoveAll(loc.Dir)
	}
	if loc.Previous != "" && loc.Previous != def {
		os.RemoveAll(loc.Previous)
	}
	for _, d := range rs.Also {
		if d != "" && d != "/" {
			os.RemoveAll(d)
		}
	}
	RemoveExcept(def)
	os.Remove(filepath.Join(def, LocationFile))
	os.Remove(filepath.Join(def, ResetFile))
	log.Printf("data: erased everything, as asked in Settings → Storage")
}
