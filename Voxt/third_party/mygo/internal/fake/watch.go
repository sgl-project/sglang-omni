package fake

import (
	"os"
	"sync"

	"github.com/egoist/mygo/internal/platform"
)

// FileWatch records the targets the core enrolls; tests report changes
// with Notify.
type FileWatch struct {
	mu      sync.Mutex
	targets map[uint64]platform.FileWatchTarget
	notify  func(platform.FileWatchNotice)
	closed  bool
	// Subtree makes Add accept Subtree targets, as Windows does.
	Subtree bool
	// NoFiles makes Add refuse targets that aren't directories, as every
	// system backend does.
	NoFiles bool
	// AddHook, when set, runs in Add before the target is enrolled; an
	// error it returns is Add's.
	AddHook func(platform.FileWatchTarget) error
}

func (b *Backend) NewFileWatch(notify func(platform.FileWatchNotice)) (platform.FileWatch, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.FileWatchError != nil {
		return nil, b.FileWatchError
	}
	w := &FileWatch{targets: map[uint64]platform.FileWatchTarget{}, notify: notify, Subtree: b.FileWatchSubtree, NoFiles: b.FileWatchNoFiles, AddHook: b.FileWatchAddHook}
	b.FileWatches = append(b.FileWatches, w)
	return w, nil
}

// LastFileWatch returns the FileWatch created last.
func (b *Backend) LastFileWatch() *FileWatch {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.FileWatches[len(b.FileWatches)-1]
}

func (w *FileWatch) Add(t platform.FileWatchTarget) error {
	if t.Subtree && !w.Subtree || !t.Directory && w.NoFiles {
		return platform.ErrUnsupported
	}
	if hook := w.AddHook; hook != nil {
		if err := hook(t); err != nil {
			return err
		}
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.targets[t.ID] = t
	return nil
}

func (w *FileWatch) Remove(id uint64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.targets, id)
}

func (w *FileWatch) Close() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.closed = true
	clear(w.targets)
}

// Targets returns the enrolled targets.
func (w *FileWatch) Targets() []platform.FileWatchTarget {
	w.mu.Lock()
	defer w.mu.Unlock()
	targets := make([]platform.FileWatchTarget, 0, len(w.targets))
	for _, t := range w.targets {
		targets = append(targets, t)
	}
	return targets
}

// Notify reports n, as a backend's goroutine would, unless the watch is
// closed.
func (w *FileWatch) Notify(n platform.FileWatchNotice) {
	w.mu.Lock()
	closed := w.closed
	w.mu.Unlock()
	if !closed {
		w.notify(n)
	}
}

// ReplaceFile renames replacement over target, with the target's
// permissions.
func (b *Backend) ReplaceFile(target, replacement string) error {
	if info, err := os.Stat(target); err == nil {
		if err := os.Chmod(replacement, info.Mode().Perm()); err != nil {
			return err
		}
	}
	return os.Rename(replacement, target)
}
