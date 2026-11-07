package mygo

import (
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"time"
)

// fileStamp tells whether a file changed: its modification time and size
// as the document last read or wrote it, as NSDocument and VS Code
// compare them.
type fileStamp struct {
	ModTime int64 `json:"modTime"` // Unix nanoseconds
	Size    int64 `json:"size"`
}

func stampOf(info os.FileInfo) fileStamp {
	return fileStamp{ModTime: info.ModTime().UnixNano(), Size: info.Size()}
}

// statStamp returns the stamp of the file at path; ok is false when it
// doesn't exist.
func statStamp(path string) (s fileStamp, ok bool, err error) {
	info, err := os.Stat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return fileStamp{}, false, nil
	}
	if err != nil {
		return fileStamp{}, false, err
	}
	return stampOf(info), true, nil
}

// writeOp is why a document is written.
type writeOp int

const (
	writeSave      writeOp = iota // to its file, unless another app changed it
	writeAutosave                 // as writeSave, without asking anything
	writeOverwrite                // to its file, whatever is there
	writeSaveAs                   // to another file, which the document becomes
)

// write writes the document to target. It reports whether it wrote
// anything: a document whose file another app changed is not written
// (ErrDocumentChanged), but marked in conflict.
func (b *docBase) write(target string, op writeOp) (bool, error) {
	snap := b.ops.capture()
	b.io.Lock()
	defer b.io.Unlock()
	b.mu.Lock()
	if b.closed && op == writeAutosave {
		b.mu.Unlock()
		return false, nil
	}
	known, deleted := b.stamp, b.deleted
	b.mu.Unlock()
	if op == writeSave || op == writeAutosave {
		now, exists, err := statStamp(target)
		if err != nil {
			return false, err
		}
		if exists && (now != known || deleted) {
			b.mu.Lock()
			was := b.conflict
			b.conflict, b.deleted = true, false
			b.mu.Unlock()
			if !was {
				b.notify()
			}
			return false, ErrDocumentChanged
		}
	}
	stamp, err := writeFileSafely(target, snap.write)
	if err != nil {
		return false, err
	}
	b.mu.Lock()
	b.ops.markSaved(snap)
	b.savedAt = snap.at
	b.stamp = stamp
	b.conflict, b.deleted = false, false
	moved := target != b.path
	if moved {
		b.path = target
		b.readOnly = false
	}
	b.dirtySince = time.Time{}
	if b.edited {
		b.dirtySince = time.Now() // edited while it was written
	}
	b.scheduleLocked()
	drop := !b.edited && b.backedUp != 0
	b.mu.Unlock()
	if drop {
		b.kind.recovery.remove(b)
	}
	if moved {
		addRecent(target)
		b.watch()
	}
	b.notify()
	return true, nil
}

// reload reads path again and makes it the content.
func (b *docBase) reload(path string) error {
	b.io.Lock()
	v, stamp, err := b.kind.readFile(path)
	if err != nil {
		b.io.Unlock()
		return err
	}
	b.mu.Lock()
	b.ops.replace(v)
	b.stamp = stamp
	b.conflict, b.deleted = false, false
	b.readOnly = !fileWritable(path)
	b.dirtySince = time.Time{}
	b.scheduleLocked()
	backedUp := b.backedUp != 0
	b.mu.Unlock()
	b.io.Unlock()
	if backedUp {
		b.kind.recovery.remove(b)
	}
	postMain(func() {
		fire(&b.onReload)
		if w := b.parent(); w != nil {
			emitDocumentReload(w)
		}
	})
	b.notify()
	return nil
}

// writeFileSafely writes a file as NSDocument and GIO do: into a new file
// next to it, flushed to the disk, which then replaces it, keeping its
// metadata (Backend.ReplaceFile), so that a crash or a full disk never
// leaves half a file. A symlink's target is replaced, and a file with
// other hard links, or in a directory the user can't create files in, is
// written in place, as GIO does.
func writeFileSafely(path string, write func(io.Writer) error) (fileStamp, error) {
	if real, err := filepath.EvalSymlinks(path); err == nil {
		path = real
	}
	info, err := os.Stat(path)
	exists := err == nil
	if exists && fileLinks(info) > 1 {
		return writeFileInPlace(path, write)
	}
	tmp, err := createSibling(path)
	if err != nil {
		if exists && errors.Is(err, fs.ErrPermission) {
			return writeFileInPlace(path, write)
		}
		return fileStamp{}, err
	}
	name := tmp.Name()
	err = writeAndSync(tmp, write)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = backend().ReplaceFile(path, name)
	}
	if err != nil {
		os.Remove(name)
		return fileStamp{}, err
	}
	s, _, err := statStamp(path)
	return s, err
}

// createSibling creates a hidden file next to path, with the permissions
// new files get (the umask applies).
func createSibling(path string) (*os.File, error) {
	dir, base := filepath.Split(path)
	for range 10 {
		f, err := os.OpenFile(filepath.Join(dir, "."+base+"."+randomText()[:8]+".tmp"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o666)
		if !errors.Is(err, fs.ErrExist) {
			return f, err
		}
	}
	return nil, errors.New("mygo: cannot create a temporary file next to " + path)
}

func writeFileInPlace(path string, write func(io.Writer) error) (fileStamp, error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_TRUNC, 0)
	if err != nil {
		return fileStamp{}, err
	}
	err = writeAndSync(f, write)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fileStamp{}, err
	}
	s, _, err := statStamp(path)
	return s, err
}

func writeAndSync(f *os.File, write func(io.Writer) error) error {
	if err := write(f); err != nil {
		return err
	}
	return f.Sync()
}
