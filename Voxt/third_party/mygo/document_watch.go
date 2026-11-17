package mygo

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Documents notice what other apps do to their files: one WatchFiles per
// open document's file, and a look when a document's window is focused,
// since watches miss changes (VS Code and Notepad++ look then too). An
// unedited document reads its file again, an edited one is in conflict
// and asks at its next save, a file renamed in its directory is followed,
// and a deleted one is written again by the next save. A watch of the
// file rather than of its directory costs the same whatever else the
// directory holds: a directory is one FSEvents stream on macOS, whatever
// is in it.

var docWatches struct {
	sync.Mutex
	files map[string]*docFileWatch
}

type docFileWatch struct {
	refs   int
	cancel context.CancelFunc
}

// watch watches b's file, after it opened or moved.
func (b *docBase) watch() {
	b.mu.Lock()
	path := ""
	if !b.closed {
		path = b.path
	}
	old := b.watched
	b.watched = path
	b.mu.Unlock()
	if path == old {
		return
	}
	docWatches.Lock()
	defer docWatches.Unlock()
	if old != "" {
		if w := docWatches.files[old]; w != nil {
			if w.refs--; w.refs == 0 {
				w.cancel()
				delete(docWatches.files, old)
			}
		}
	}
	if path == "" {
		return
	}
	if w := docWatches.files[path]; w != nil {
		w.refs++
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	if docWatches.files == nil {
		docWatches.files = map[string]*docFileWatch{}
	}
	docWatches.files[path] = &docFileWatch{refs: 1, cancel: cancel}
	go watchDocumentFile(ctx, path)
}

// watchDocumentFile follows the changes of the file at path until ctx is
// done, or its directory goes away.
func watchDocumentFile(ctx context.Context, path string) {
	w, err := WatchFiles(ctx, path, FileWatchOptions{Debounce: 100 * time.Millisecond})
	if err != nil {
		return // the file is gone: focus checks remain
	}
	defer w.Close()
	// What changed before the watch started, as after a rename.
	for _, b := range documentsAt(path) {
		go b.checkDisk()
	}
	for {
		r := w.next(ctx)
		if r.err != nil {
			return
		}
		// The file the watch saw go away, by what it was when the watch
		// last saw it: a look now may come after the next rename.
		moved := ""
		if r.events[len(r.events)-1].Op == FileRemove && r.gone.ok {
			moved = findRenamed(filepath.Dir(w.Root()), r.gone.id)
		}
		for _, b := range documentsAt(path) {
			if moved != "" {
				b.moved(moved)
			} else {
				go b.checkDisk()
			}
		}
		if moved != "" {
			return // moved watches the new path
		}
	}
}

// findRenamed returns the path of the file in dir that is the file id
// identifies, or "".
func findRenamed(dir string, id fileWatchID) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if other, ok := fileWatchIdentity(StatIdentity(info)); ok && other == id {
			return filepath.Join(dir, entry.Name())
		}
	}
	return ""
}

// documentsAt returns the open documents of the file at path.
func documentsAt(path string) []*docBase {
	var at []*docBase
	for _, b := range allDocuments() {
		if b.Path() == path {
			at = append(at, b)
		}
	}
	return at
}

// checkDisk compares b's file with what b last read or wrote.
func (b *docBase) checkDisk() {
	b.io.Lock()
	b.mu.Lock()
	path, known, edited, closed := b.path, b.stamp, b.edited, b.closed
	wasDeleted := b.deleted
	b.mu.Unlock()
	if path == "" || closed {
		b.io.Unlock()
		return
	}
	now, exists, err := statStamp(path)
	if err == nil && !exists && !wasDeleted {
		// Network shares report files gone that are not (VS Code).
		time.Sleep(100 * time.Millisecond)
		now, exists, err = statStamp(path)
	}
	if err != nil {
		b.io.Unlock()
		return
	}
	switch {
	case !exists:
		b.mu.Lock()
		changed := !b.deleted
		b.deleted = true
		b.mu.Unlock()
		b.io.Unlock()
		if changed {
			b.notify()
		}
	case now == known && !wasDeleted:
		readOnly := !fileWritable(path)
		b.mu.Lock()
		changed := b.readOnly != readOnly
		b.readOnly = readOnly
		b.mu.Unlock()
		b.io.Unlock()
		if changed {
			b.notify()
		}
	case !edited:
		b.io.Unlock()
		_ = b.reload(path)
	default:
		b.mu.Lock()
		changed := !b.conflict || b.deleted
		b.conflict, b.deleted = true, false
		b.mu.Unlock()
		b.io.Unlock()
		if changed {
			b.notify()
		}
	}
}

// moved follows b's file, which another app renamed to path.
func (b *docBase) moved(path string) {
	b.mu.Lock()
	b.path = path
	b.mu.Unlock()
	addRecent(path)
	b.watch()
	b.notify()
}
