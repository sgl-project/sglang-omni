package mygo

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"sync"
	"time"

	"github.com/egoist/mygo/internal/platform"
)

// FileOp is the kind of a FileEvent.
type FileOp string

const (
	// FileCreate is a file or directory that appeared.
	FileCreate FileOp = "create"
	// FileWrite is a file whose content may have changed: its size or
	// modification time changed, or the system said it was written.
	// Changes of permissions and other metadata are not reported, nor
	// are directories written, whose changes are their entries'.
	FileWrite FileOp = "write"
	// FileRemove is a file or directory that went away.
	FileRemove FileOp = "remove"
	// FileRename is a file or directory that moved within the watched
	// directory.
	FileRename FileOp = "rename"
	// FileHint says only that something in or under the directory at
	// Path changed: which file, and how, is unknown. A watch with
	// FileWatchOptions.HintsOnly reports nothing else, but a FileRemove
	// of "." when its directory goes away.
	FileHint FileOp = "hint"
)

// FileEvent is a change in a watched file or directory. A watch compares
// what it sees with what it saw before, so events say what is different,
// not every step that made it so: a file written twice in a row is one
// write, and one created and removed in between is nothing. Treat them as
// hints to read the files again.
type FileEvent struct {
	Op FileOp `json:"op"`
	// Path is the path relative to the watched directory, separated by
	// slashes; "." is the watched directory or file itself.
	Path string `json:"path"`
	// OldPath is where a FileRename moved from.
	OldPath string `json:"oldPath,omitzero"`
	// IsDir reports whether it is a directory, as it was before a
	// FileRemove.
	IsDir bool `json:"isDir"`
}

// FileWatchOptions configures WatchFiles. The zero value watches a file,
// or a directory without the directories in it.
type FileWatchOptions struct {
	// Recursive watches the directories in the directory too.
	Recursive bool
	// Ignore, when set, is asked about every file and directory the
	// watch finds, by its path relative to the watched directory,
	// separated by slashes, and whether it is a directory. What it
	// ignores is not watched, reported or counted in MaxEntries, nor is
	// anything in a directory it ignores, which is never opened: return
	// true for node_modules and .git. It runs on a goroutine of the
	// watch, and must be quick.
	Ignore func(path string, isDir bool) bool
	// Debounce is how long a watch waits for changes to stop before it
	// reports them: 50 ms when zero, right away when negative.
	Debounce time.Duration
	// MaxDelay is the longest changes wait while more keep coming: 500 ms
	// when zero, and never less than Debounce.
	MaxDelay time.Duration
	// MaxEntries is how many files and directories a watch keeps a
	// snapshot of before it stops with ErrWatchLimit: 65536 on macOS,
	// where a watch covers a whole tree with one stream, and 8192
	// elsewhere, where a backend may take a handle per entry, when zero.
	// With HintsOnly it counts the directories the watch enrolls.
	MaxEntries int
	// HintsOnly makes the watch cheap and coarse: it keeps no snapshot of
	// the tree and stats no file, so it reports only FileHint events that
	// name a directory in or under which the system said something
	// changed, never which file, how, or that it was renamed. After the
	// system lost changes it hints "." (look at everything). It suits a
	// dev reloader, which rebuilds whatever changed; never use it where
	// what changed matters, such as following a document's file.
	//
	// It still doesn't follow symlinks and doesn't leave the directory.
	// Ignore is asked only about directories: what is in one it ignores
	// is not watched or hinted, but a change to a file it would ignore
	// still hints at that file's directory. On macOS and Windows a
	// recursive watch enrolls only its directory, which the system
	// watches as a tree; on Linux it enrolls every directory, without
	// stat'ing what is in them.
	HintsOnly bool
}

// ErrWatchLimit is what a watch stops with when its directory has more
// than FileWatchOptions.MaxEntries files and directories.
var ErrWatchLimit = errors.New("mygo: file watch entry limit exceeded")

// FileWatcher is a watch WatchFiles started. Its methods are safe from any
// goroutine.
type FileWatcher struct {
	mu sync.Mutex
	// batch is the changes waiting for Next, nil when none do: the watch
	// doesn't look again until Next takes them, which taken tells. gone
	// is what a watched file that the batch removes was.
	batch   []FileEvent
	gone    fileWatchGone
	waiters []chan fileResult
	stopped bool
	err     error
	taken   chan struct{}
	done    chan struct{}
	ctx     context.Context
	cancel  context.CancelFunc
	// native is the backend's watch, which only the goroutine of the
	// watch uses; what it reports goes through raw. What comes while raw
	// is full, as during a burst while the watch looks, waits in spill
	// (spilled tells), up to MaxEntries notices; past that, overflow says
	// some were dropped.
	native   platform.FileWatch
	raw      chan platform.FileWatchNotice
	spillMu  sync.Mutex
	spill    []platform.FileWatchNotice
	spilled  chan struct{}
	overflow chan struct{}
	root     string
	opts     FileWatchOptions
}
type fileResult struct {
	events []FileEvent
	gone   fileWatchGone
	err    error
}

// fileWatchGone is the identity of a watched file that went away, as the
// watch last saw it, so that a document finds the file again where
// another app renamed it (findRenamed): a look at the file after the
// batch came may be too late, once it moved.
type fileWatchGone struct {
	id fileWatchID
	ok bool
}

var fileWatches = struct {
	sync.Mutex
	stopped bool
	active  map[*FileWatcher]bool
}{active: make(map[*FileWatcher]bool)}

// WatchFiles watches the file or directory at path, and with
// opts.Recursive the directories in it, but not those that symlinks or
// Windows junctions point to. It returns once everything is watched; what
// is already there is not reported. Canceling ctx stops the watch, as
// Close does.
//
// A file is watched through its directory, so that a file replaced, as
// editors save, is reported as written, one removed as removed, and one
// that appears again as created. When the directory itself (or the
// directory of the file) is removed, renamed or replaced, the watch
// reports a FileRemove of "." and ends: Next returns io.EOF.
//
// It can be called from any goroutine, and on the main thread once the app
// runs. The app stops its watches when it quits.
func WatchFiles(ctx context.Context, path string, opts FileWatchOptions) (*FileWatcher, error) {
	needsApp("WatchFiles")
	if ctx == nil {
		return nil, errors.New("mygo: nil watch context")
	}
	if path == "" {
		return nil, errors.New("mygo: empty watch path")
	}
	if opts.MaxEntries < 0 {
		return nil, errors.New("mygo: negative watch limit")
	}
	if opts.MaxEntries == 0 {
		opts.MaxEntries = defaultWatchMaxEntries
	}
	if opts.Debounce == 0 {
		opts.Debounce = 50 * time.Millisecond
	}
	opts.Debounce = max(opts.Debounce, 0)
	if opts.MaxDelay == 0 {
		opts.MaxDelay = 500 * time.Millisecond
	}
	opts.MaxDelay = max(opts.MaxDelay, opts.Debounce)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	root, err := filepath.EvalSymlinks(absolute)
	if err != nil {
		return nil, err
	}
	life, cancel := context.WithCancel(ctx)
	w := &FileWatcher{ctx: life, cancel: cancel, root: root, opts: opts, taken: make(chan struct{}, 1), done: make(chan struct{}), raw: make(chan platform.FileWatchNotice, 4096), spilled: make(chan struct{}, 1), overflow: make(chan struct{}, 1)}
	fileWatches.Lock()
	if fileWatches.stopped {
		fileWatches.Unlock()
		cancel()
		return nil, errLoopStopped
	}
	fileWatches.active[w] = true
	fileWatches.Unlock()
	if w.native, err = backend().NewFileWatch(w.notice); err != nil {
		fileWatches.Lock()
		delete(fileWatches.active, w)
		fileWatches.Unlock()
		cancel()
		return nil, fmt.Errorf("mygo: watch: %w", err)
	}
	ready := make(chan error, 1)
	go w.run(ready)
	if err := await(ready); err != nil {
		return nil, err
	}
	return w, nil
}

// notice is what the backend reports to, on its goroutine. When raw is
// full the notice is spilled, and when the spill is full too the watch
// looks at everything again instead, as after any overflow: either way no
// change goes unseen.
func (w *FileWatcher) notice(n platform.FileWatchNotice) {
	select {
	case <-w.ctx.Done():
		return
	default:
	}
	if n.Err != nil && !errors.Is(n.Err, platform.ErrFileWatchOverflow) {
		// An error stops the watch, so it waits its turn.
		select {
		case w.raw <- n:
		case <-w.ctx.Done():
		}
		return
	}
	select {
	case w.raw <- n:
		return
	default:
	}
	w.spillMu.Lock()
	signal := w.spilled
	if len(w.spill) < w.opts.MaxEntries {
		w.spill = append(w.spill, n)
	} else {
		// More than a look at everything would cost to follow.
		w.spill = nil
		signal = w.overflow
	}
	w.spillMu.Unlock()
	select {
	case signal <- struct{}{}:
	default:
	}
}

// unspill takes the notices spilled.
func (w *FileWatcher) unspill() []platform.FileWatchNotice {
	w.spillMu.Lock()
	defer w.spillMu.Unlock()
	spill := w.spill
	w.spill = nil
	return spill
}

// Root returns the path the watch watches, absolute and with its symlinks
// resolved (macOS's /tmp is /private/tmp), to which the paths of events
// are relative.
func (w *FileWatcher) Root() string { return w.root }

// Next waits for the next changes and returns them, a batch of the events
// that one look found once changes stopped for Debounce. The watch doesn't
// look again until Next has taken a batch, so nothing is lost when Next is
// called late: what changed meanwhile comes in the next batch.
//
// It returns io.EOF once the watch is closed, or ended because its
// directory went away, and the error that stopped it otherwise. Canceling
// ctx cancels the wait, not the watch. On the main thread it keeps
// processing events while it waits.
func (w *FileWatcher) Next(ctx context.Context) ([]FileEvent, error) {
	if ctx == nil {
		return nil, errors.New("mygo: nil wait context")
	}
	r := w.next(ctx)
	return r.events, r.err
}

// next is Next, with what a watched file that the batch removes was.
func (w *FileWatcher) next(ctx context.Context) fileResult {
	if err := ctx.Err(); err != nil {
		return fileResult{err: err}
	}
	w.mu.Lock()
	if w.batch != nil {
		r := fileResult{events: w.batch, gone: w.gone}
		w.batch, w.gone = nil, fileWatchGone{}
		w.mu.Unlock()
		w.take()
		return r
	}
	if w.stopped {
		err := w.err
		if err == nil {
			err = io.EOF
		}
		w.mu.Unlock()
		return fileResult{err: err}
	}
	waiter := make(chan fileResult, 1)
	w.waiters = append(w.waiters, waiter)
	w.mu.Unlock()
	stop := context.AfterFunc(ctx, func() {
		w.mu.Lock()
		found := false
		for i, v := range w.waiters {
			if v == waiter {
				w.waiters = append(w.waiters[:i], w.waiters[i+1:]...)
				found = true
				break
			}
		}
		w.mu.Unlock()
		if found {
			deliver(waiter, fileResult{err: ctx.Err()})
		}
	})
	defer stop()
	return await(waiter)
}

// take tells the goroutine of the watch that a batch was taken.
func (w *FileWatcher) take() {
	select {
	case w.taken <- struct{}{}:
	default:
	}
}

// Close stops the watch and drops the changes not yet returned by Next,
// which then returns io.EOF. It doesn't wait for the watch to release
// what it holds; Done does. It returns nil.
func (w *FileWatcher) Close() error {
	w.stop(nil, true)
	return nil
}

// Done is closed once the watch has stopped and released its handles.
func (w *FileWatcher) Done() <-chan struct{} { return w.done }

// Err returns the error that stopped the watch: nil while it runs, and
// once it was closed, its context canceled, or its directory went away.
func (w *FileWatcher) Err() error { w.mu.Lock(); defer w.mu.Unlock(); return w.err }

func (w *FileWatcher) stop(err error, discard bool) {
	w.mu.Lock()
	if w.stopped {
		if discard {
			w.batch = nil
		}
		w.mu.Unlock()
		return
	}
	w.stopped = true
	w.err = err
	if discard {
		w.batch = nil
	}
	waiters := w.waiters
	w.waiters = nil
	w.mu.Unlock()
	if err == nil {
		err = io.EOF
	}
	for _, waiter := range waiters {
		deliver(waiter, fileResult{err: err})
	}
	w.cancel()
}

// emit hands events to a waiting Next, or keeps them for the next one.
func (w *FileWatcher) emit(events []FileEvent, gone fileWatchGone) {
	if len(events) == 0 {
		return
	}
	w.mu.Lock()
	if w.stopped {
		w.mu.Unlock()
		return
	}
	if len(w.waiters) == 0 {
		w.batch = append(w.batch, events...)
		if gone.ok {
			w.gone = gone
		}
		w.mu.Unlock()
		return
	}
	waiter := w.waiters[0]
	w.waiters = w.waiters[1:]
	w.mu.Unlock()
	deliver(waiter, fileResult{events: events, gone: gone})
}

// waiting reports whether a batch waits for Next.
func (w *FileWatcher) waiting() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.batch != nil
}

func stopFileWatches() {
	fileWatches.Lock()
	fileWatches.stopped = true
	all := make([]*FileWatcher, 0, len(fileWatches.active))
	for w := range fileWatches.active {
		all = append(all, w)
	}
	fileWatches.Unlock()
	for _, w := range all {
		w.Close()
	}
	for _, w := range all {
		<-w.Done()
	}
}

// run is the goroutine of the watch: it enrolls the tree, then waits for
// what the backend reports and, once nothing more came for opts.Debounce
// (or opts.MaxDelay passed) and Next took the last batch, looks at it.
func (w *FileWatcher) run(ready chan error) {
	var terminal error
	discard := true
	var scan fileWatchLooker = newFileWatchScan(w)
	if w.opts.HintsOnly {
		scan = newFileWatchHints(w)
	}
	defer func() {
		w.stop(terminal, discard)
		scan.close()
		w.native.Close()
		fileWatches.Lock()
		delete(fileWatches.active, w)
		fileWatches.Unlock()
		close(w.done)
	}()
	events, err := scan.start()
	if err != nil {
		terminal = err
		deliver(ready, err)
		return
	}
	w.emit(events, fileWatchGone{})
	deliver(ready, nil)
	var timer *time.Timer
	var timerC <-chan time.Time
	var first time.Time
	defer func() {
		if timer != nil {
			timer.Stop()
		}
	}()
	schedule := func() {
		if timerC == nil {
			first = time.Now()
		}
		delay := min(w.opts.Debounce, time.Until(first.Add(w.opts.MaxDelay)))
		if timer == nil {
			timer = time.NewTimer(delay)
		} else {
			timer.Reset(delay)
		}
		timerC = timer.C
	}
	// due says changes settled while a batch waited for Next.
	due := false
	// look looks at what changed and reports it; false ends the watch.
	look := func() bool {
		due = false
		events, err := scan.update()
		if err == nil {
			w.emit(events, scan.removedFile())
			return true
		}
		if w.ctx.Err() != nil {
			return false
		}
		if scan.gone() {
			// The directory went away: say so, then end as closed.
			w.emit(scan.goneEvents(), fileWatchGone{})
			discard = false
			return false
		}
		terminal = err
		return false
	}
	for {
		select {
		case <-w.ctx.Done():
			return
		case <-w.overflow:
			scan.lookAll()
			schedule()
		case <-w.spilled:
			for _, n := range w.unspill() {
				if err := scan.mark(n); err != nil {
					terminal = err
					return
				}
			}
			if scan.pending() {
				schedule()
			}
		case n := <-w.raw:
			if err := scan.mark(n); err != nil {
				terminal = err
				return
			}
			if scan.pending() {
				schedule()
			}
		case <-timerC:
			timerC = nil
			if w.waiting() {
				due = true
			} else if !look() {
				return
			}
		case <-w.taken:
			if due && timerC == nil && !look() {
				return
			}
		}
	}
}

var errWatchRootChanged = errors.New("watched root identity changed")
