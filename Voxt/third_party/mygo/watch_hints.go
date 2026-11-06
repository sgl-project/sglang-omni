package mygo

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/egoist/mygo/internal/platform"
)

// fileWatchLooker is what the goroutine of a watch looks with: the
// snapshot (fileWatchScan), or with HintsOnly the directories alone
// (fileWatchHints).
type fileWatchLooker interface {
	// start enrolls the tree and returns the events of what changed while
	// it looked.
	start() ([]FileEvent, error)
	// mark records what a notice says changed, for the next update; an
	// error ends the watch.
	mark(platform.FileWatchNotice) error
	pending() bool
	update() ([]FileEvent, error)
	// removedFile is what the watched file was, when the last update
	// found it gone.
	removedFile() fileWatchGone
	// lookAll makes the next update look at everything, after the system
	// lost changes.
	lookAll()
	// gone reports whether the watched directory went away, and
	// goneEvents what a watch reports then.
	gone() bool
	goneEvents() []FileEvent
	close()
}

func (s *fileWatchScan) lookAll() { s.all = true }

// fileWatchHints is what a watch with HintsOnly keeps: no snapshot, only
// the directories it enrolls, and where to look next. Only the goroutine
// of the watch uses it.
type fileWatchHints struct {
	w *FileWatcher
	// root is the watched directory, or the directory of the watched file
	// (file), at dir; identity is what it is, and name the file's name.
	root     *os.Root
	identity os.FileInfo
	dir      string
	file     bool
	name     string
	// targets are the enrolled directories, by slash-separated path
	// relative to the root ("." for the root, ".." for its parent).
	targets            map[string]fileWatchTargetRef
	ids                map[uint64]string
	next               uint64
	subtree, noSubtree bool
	// What the next update does: the directories it hints at, the paths
	// it enrolls if they are directories (walks), the enrolled
	// directories it checks are still there (checks), whether it checks
	// the root (check) and looks at everything (all).
	hints, walks, checks map[string]bool
	check, all           bool
}

func newFileWatchHints(w *FileWatcher) *fileWatchHints {
	return &fileWatchHints{
		w:       w,
		targets: map[string]fileWatchTargetRef{},
		ids:     map[uint64]string{},
		hints:   map[string]bool{},
		walks:   map[string]bool{},
		checks:  map[string]bool{},
	}
}

func (s *fileWatchHints) close() {
	if s.root != nil {
		s.root.Close()
	}
}

func (s *fileWatchHints) lookAll() { s.all = true }

func (s *fileWatchHints) start() ([]FileEvent, error) {
	info, err := os.Lstat(s.w.root)
	if err != nil {
		return nil, err
	}
	info = StatIdentity(info)
	s.dir = s.w.root
	if !info.IsDir() {
		s.file = true
		s.dir, s.name = filepath.Dir(s.w.root), filepath.Base(s.w.root)
	}
	if s.root, err = fileWatchOpenRoot(s.dir); err != nil {
		return nil, err
	}
	if s.identity, err = s.root.Stat("."); err != nil {
		return nil, err
	}
	s.identity = StatIdentity(s.identity)
	if s.file {
		if info, err = os.Lstat(s.dir); err != nil {
			return nil, err
		}
		info = StatIdentity(info)
	}
	if !os.SameFile(info, s.identity) {
		return nil, &os.PathError{Op: "watch", Path: s.dir, Err: errWatchRootChanged}
	}
	if err := s.watchParent(); err != nil {
		return nil, err
	}
	if s.file {
		// The file is watched through its directory, enrolled as "..".
		return nil, nil
	}
	return nil, s.enrollTree(s.root, ".", ".", s.identity, nil)
}

// watchParent enrolls the directory of a watched file, and the parent of
// the watched directory on Windows, as fileWatchScan.watchParent does.
func (s *fileWatchHints) watchParent() error {
	if !s.file && runtime.GOOS != "windows" {
		return nil
	}
	parent := filepath.Dir(s.w.root)
	if parent == s.w.root {
		return nil
	}
	if _, ok := s.targets[".."]; ok {
		return nil
	}
	info, err := os.Lstat(parent)
	if err != nil {
		return err
	}
	return s.add("..", StatIdentity(info))
}

// Hints watch no file.
func (s *fileWatchHints) removedFile() fileWatchGone { return fileWatchGone{} }

func (s *fileWatchHints) gone() bool {
	info, err := os.Lstat(s.dir)
	return errors.Is(err, fs.ErrNotExist) || err == nil && (!info.IsDir() || !os.SameFile(StatIdentity(info), s.identity))
}

func (s *fileWatchHints) goneEvents() []FileEvent {
	if s.file {
		// The file's directory went away, and the file with it.
		return []FileEvent{{Op: FileRemove, Path: "."}}
	}
	return []FileEvent{{Op: FileRemove, Path: ".", IsDir: true}}
}

// ignores reports whether directory p or one above it is one Ignore
// ignores.
func (s *fileWatchHints) ignores(p string) bool {
	if s.w.opts.Ignore == nil {
		return false
	}
	for ; p != "." && p != ""; p = path.Dir(p) {
		if s.w.opts.Ignore(p, true) {
			return true
		}
	}
	return false
}

// enrollTree enrolls directory p, at name in dir (the root for "."), and
// with Recursive the directories beneath it, but no file: what is beneath
// p is watched through it. want, when set, is what p must still be; a
// directory replaced in between is left to the hint of its parent.
// Directories are added to seen, when set.
func (s *fileWatchHints) enrollTree(dir *os.Root, name, p string, want os.FileInfo, seen map[string]bool) error {
	if err := s.w.ctx.Err(); err != nil {
		return err
	}
	d := s.root
	if p != "." {
		var err error
		if d, err = dir.OpenRoot(name); err != nil {
			if errors.Is(err, fs.ErrNotExist) || fileWatchInaccessible(err) {
				return nil // gone or unreadable: its parent's hint covers it
			}
			if checked, statErr := dir.Lstat(name); statErr != nil || !checked.IsDir() {
				return nil // replaced by a file or a symlink, which aren't enrolled
			}
			return err
		}
		defer d.Close()
	}
	info, err := d.Stat(".")
	if err != nil {
		if p != "." && (errors.Is(err, fs.ErrNotExist) || fileWatchInaccessible(err)) {
			return nil
		}
		return err
	}
	info = StatIdentity(info)
	if want != nil && !os.SameFile(want, info) {
		return nil
	}
	if seen != nil {
		seen[p] = true
	}
	if err := s.enroll(p, info); err != nil {
		return err
	}
	if !s.w.opts.Recursive || s.subtree {
		// Only the directory, or a tree its target watches whole.
		return nil
	}
	names, err := fileWatchSubdirs(d)
	if err != nil {
		if p != "." && (errors.Is(err, fs.ErrNotExist) || fileWatchInaccessible(err)) {
			return nil
		}
		return err
	}
	for _, child := range names {
		q := path.Join(p, child)
		if s.w.opts.Ignore != nil && s.w.opts.Ignore(q, true) {
			continue
		}
		if err := s.enrollTree(d, child, q, nil, seen); err != nil {
			return err
		}
	}
	return nil
}

// enroll has the backend watch directory p, which is info; one replaced
// as it is enrolled is left to the hint of its parent.
func (s *fileWatchHints) enroll(p string, info os.FileInfo) error {
	if t, ok := s.targets[p]; ok {
		if t.identity == fileWatchEntryOf(info).id {
			return nil
		}
		s.unenroll(p)
	}
	if len(s.targets) >= s.w.opts.MaxEntries {
		return ErrWatchLimit
	}
	err := s.add(p, info)
	if err == nil || errors.Is(err, context.Canceled) {
		return err
	}
	checked, statErr := s.root.Lstat(filepath.FromSlash(p))
	if errors.Is(statErr, fs.ErrNotExist) || statErr == nil && !os.SameFile(StatIdentity(checked), info) || fileWatchInaccessible(err) {
		if p == "." {
			return &os.PathError{Op: "watch", Path: s.dir, Err: errWatchRootChanged}
		}
		s.hints[path.Dir(p)] = true
		return nil
	}
	return err
}

func (s *fileWatchHints) add(p string, info os.FileInfo) error {
	s.next++
	t := platform.FileWatchTarget{ID: s.next, Path: filepath.Join(s.w.root, filepath.FromSlash(p)), Directory: true, Info: info}
	if p == "." && s.w.opts.Recursive && !s.noSubtree {
		t.Subtree = true
		err := s.w.native.Add(t)
		if errors.Is(err, platform.ErrUnsupported) {
			s.noSubtree = true
			t.Subtree = false
		} else if err != nil {
			return &os.PathError{Op: "watch", Path: t.Path, Err: err}
		} else {
			s.subtree = true
			s.record(p, t)
			return nil
		}
	}
	if err := s.w.native.Add(t); err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		return &os.PathError{Op: "watch", Path: t.Path, Err: err}
	}
	s.record(p, t)
	return nil
}

func (s *fileWatchHints) record(p string, t platform.FileWatchTarget) {
	s.targets[p] = fileWatchTargetRef{id: t.ID, identity: fileWatchEntryOf(t.Info).id, directory: true, subtree: t.Subtree}
	s.ids[t.ID] = p
}

func (s *fileWatchHints) unenroll(p string) {
	if t, ok := s.targets[p]; ok {
		delete(s.targets, p)
		delete(s.ids, t.id)
		s.w.native.Remove(t.id)
		if p == "." {
			s.subtree = false
		}
	}
}

// unenrollTree unenrolls p and the directories beneath it.
func (s *fileWatchHints) unenrollTree(p string) {
	for q := range s.targets {
		if q == p || strings.HasPrefix(q, p+"/") {
			s.unenroll(q)
		}
	}
}

func (s *fileWatchHints) mark(n platform.FileWatchNotice) error {
	if n.Err != nil {
		if errors.Is(n.Err, platform.ErrFileWatchOverflow) {
			s.all = true
			return nil
		}
		if _, ok := s.ids[n.Target]; !ok && n.Target != 0 {
			return nil // a target removed since
		}
		return n.Err
	}
	p, ok := s.ids[n.Target]
	if !ok {
		return nil
	}
	if p == ".." {
		if n.Kind == platform.FileWatchLost {
			s.unenroll("..") // enrolled again, or the watch ends
		} else if n.Name != filepath.Base(s.w.root) && (n.Name != "" || !s.file) {
			return nil // another entry of the directory
		}
		if s.file {
			s.hints["."] = true
		}
		s.check = true
		return nil
	}
	if n.Kind == platform.FileWatchLost {
		if p == "." && s.subtree {
			s.all = true
		}
		s.unenroll(p)
		s.walks[p] = true
		s.hints[path.Dir(p)] = true
		return nil
	}
	if n.Name == "" {
		// The directory itself, which may have moved.
		s.checks[p] = true
		s.hints[p] = true
		return nil
	}
	if !validWatchName(n.Name, s.subtree && p == ".") {
		return errors.New("mygo: invalid name in a file watch notice")
	}
	child := path.Join(p, n.Name)
	dir := path.Dir(child)
	if !s.w.opts.Recursive && dir != "." || s.ignores(dir) {
		return nil
	}
	s.hints[dir] = true
	if s.w.opts.Recursive && !s.subtree {
		// It may be a directory to enroll, or one enrolled that moved.
		if _, ok := s.targets[child]; ok {
			s.checks[child] = true
		}
		s.walks[child] = true
	}
	if len(s.walks)+len(s.checks) > s.w.opts.MaxEntries {
		s.all = true
		clear(s.walks)
		clear(s.checks)
	}
	return nil
}

func (s *fileWatchHints) pending() bool {
	return s.all || s.check || len(s.hints) > 0 || len(s.walks) > 0 || len(s.checks) > 0
}

func (s *fileWatchHints) update() ([]FileEvent, error) {
	if err := s.w.ctx.Err(); err != nil {
		return nil, err
	}
	info, err := os.Lstat(s.dir)
	if err != nil {
		return nil, err
	}
	if info = StatIdentity(info); !info.IsDir() || !os.SameFile(info, s.identity) {
		return nil, &os.PathError{Op: "watch", Path: s.dir, Err: errWatchRootChanged}
	}
	if err := s.watchParent(); err != nil {
		return nil, err
	}
	s.check = false
	if s.file {
		all := s.all
		s.all = false
		hint := all || len(s.hints) > 0
		clear(s.hints)
		if hint {
			return []FileEvent{{Op: FileHint, Path: "."}}, nil
		}
		return nil, nil
	}
	if s.all {
		// Everything may have changed: enroll the tree again, and drop the
		// directories gone.
		s.all = false
		clear(s.walks)
		clear(s.checks)
		clear(s.hints)
		s.hints["."] = true
		if _, ok := s.targets["."]; !ok {
			s.subtree = false
		}
		seen := map[string]bool{}
		if err := s.enrollTree(s.root, ".", ".", s.identity, seen); err != nil {
			return nil, err
		}
		for q := range s.targets {
			if q != ".." && !seen[q] {
				s.unenroll(q)
			}
		}
	} else {
		if _, ok := s.targets["."]; !ok {
			// The root's target was lost: enroll the tree again.
			if err := s.enrollTree(s.root, ".", ".", s.identity, nil); err != nil {
				return nil, err
			}
		}
		for _, p := range fileWatchShallowFirst(s.checks) {
			t, ok := s.targets[p]
			if !ok {
				continue
			}
			info, err := s.root.Lstat(filepath.FromSlash(p))
			if err == nil && info.IsDir() && t.identity == fileWatchEntryOf(info).id {
				continue
			}
			// Moved, removed or replaced: what was enrolled there is stale.
			s.unenrollTree(p)
			s.walks[p] = true
		}
		clear(s.checks)
		for _, p := range fileWatchShallowFirst(s.walks) {
			if _, ok := s.targets[p]; ok || s.ignores(p) {
				continue
			}
			info, err := s.root.Lstat(filepath.FromSlash(p))
			if err != nil || !info.IsDir() {
				continue // gone, or not a directory: watched through its directory
			}
			if err := s.enrollTree(s.root, filepath.FromSlash(p), p, StatIdentity(info), nil); err != nil {
				return nil, err
			}
		}
		clear(s.walks)
	}
	events := fileWatchHintEvents(s.hints)
	clear(s.hints)
	return events, nil
}

// fileWatchShallowFirst returns the paths of m, parents first.
func fileWatchShallowFirst(m map[string]bool) []string {
	paths := make([]string, 0, len(m))
	for p := range m {
		paths = append(paths, p)
	}
	sort.Slice(paths, func(i, j int) bool {
		if a, b := strings.Count(paths[i], "/"), strings.Count(paths[j], "/"); a != b {
			return a < b
		}
		return paths[i] < paths[j]
	})
	return paths
}

// fileWatchHintEvents turns the directories hinted at into events, sorted,
// leaving out those under another one: a hint covers what is beneath.
func fileWatchHintEvents(hints map[string]bool) []FileEvent {
	if hints["."] {
		return []FileEvent{{Op: FileHint, Path: ".", IsDir: true}}
	}
	var events []FileEvent
	for _, p := range fileWatchShallowFirst(hints) {
		covered := false
		for q := path.Dir(p); q != "."; q = path.Dir(q) {
			if hints[q] {
				covered = true
				break
			}
		}
		if !covered {
			events = append(events, FileEvent{Op: FileHint, Path: p, IsDir: true})
		}
	}
	sort.Slice(events, func(i, j int) bool { return events[i].Path < events[j].Path })
	return events
}
