package mygo

import (
	"context"
	"errors"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync/atomic"

	"github.com/egoist/mygo/internal/platform"
)

// fileWatchScan keeps what a watch saw of its tree, and which entries the
// backend watches. The system only says where something changed, so the
// watch lists or stats those places again and compares what it finds with
// what it saw: the events are the differences. Only the goroutine of the
// watch uses it.
type fileWatchScan struct {
	w *FileWatcher
	// root is the watched directory, or the directory of the watched file
	// (file), at dir; identity is what it is, and name the file's name.
	root     *os.Root
	identity os.FileInfo
	dir      string
	file     bool
	name     string
	// entries are what the scan saw, by slash-separated path relative to
	// the root ("." for the root), and children the names in each listed
	// directory; targets what it enrolled.
	entries  map[string]fileWatchEntry
	children map[string]map[string]bool
	targets  map[string]fileWatchTargetRef
	ids      map[uint64]string
	next     uint64
	// subtree says the root's target watches the whole tree (Windows), so
	// no other directory is enrolled; noSubtree that the backend can't.
	// noFiles says the backend watches files only through their
	// directories, so no file is enrolled.
	subtree, noSubtree, noFiles bool
	// What to look at again in the next update: directories to list,
	// entries to stat, entries the system said were written, and whether
	// to list every directory, after the system lost changes.
	lists, stats, writes map[string]bool
	all                  bool
	// races are entries that changed while the scan looked at them, which
	// are reported as written.
	races map[string]bool
	// ignored holds the directories Ignore ignored, by name in each
	// listed directory that has some, so that notices about what is in
	// them are dropped.
	ignored map[string]map[string]bool
	// before holds what an update changed, as it was before; removed is
	// the watched file the last update found gone.
	before  map[string]fileWatchBefore
	removed fileWatchGone
}

// fileWatchEntry is what the scan keeps of a file or directory: its
// identity, and what says its content changed. It holds no pointer, so a
// snapshot of many costs little memory and nothing for the collector to
// scan.
type fileWatchEntry struct {
	id    fileWatchID
	size  int64
	mtime int64
	mode  fs.FileMode
}

func (e fileWatchEntry) isDir() bool { return e.mode.IsDir() }

// fileWatchUnknown numbers the entries whose identity can't be read, none
// on the systems that watch, so that each is a file of its own.
var fileWatchUnknown atomic.Uint64

// fileWatchEntryOf returns what the scan keeps of info, a stat taken now:
// its identity is read now (StatIdentity), not when it is compared.
func fileWatchEntryOf(info os.FileInfo) fileWatchEntry {
	id, ok := fileWatchIdentity(StatIdentity(info))
	if !ok {
		id = fileWatchID{^uint64(0), fileWatchUnknown.Add(1)}
	}
	return fileWatchEntry{id: id, size: info.Size(), mtime: info.ModTime().UnixNano(), mode: info.Mode()}
}

// fileWatchBefore is an entry as it was before an update; ok is false for
// one that didn't exist.
type fileWatchBefore struct {
	entry fileWatchEntry
	ok    bool
}

// fileWatchTargetRef is what the scan keeps of a target it enrolled.
type fileWatchTargetRef struct {
	id                 uint64
	identity           fileWatchID
	directory, subtree bool
}

func newFileWatchScan(w *FileWatcher) *fileWatchScan {
	return &fileWatchScan{
		w:        w,
		entries:  map[string]fileWatchEntry{},
		children: map[string]map[string]bool{},
		targets:  map[string]fileWatchTargetRef{},
		ids:      map[uint64]string{},
		lists:    map[string]bool{},
		stats:    map[string]bool{},
		writes:   map[string]bool{},
		races:    map[string]bool{},
		ignored:  map[string]map[string]bool{},
	}
}

func (s *fileWatchScan) close() {
	if s.root != nil {
		s.root.Close()
	}
}

// start enrolls the root and, below it, what the watch covers. The events
// it returns are for entries that changed while it looked at them.
func (s *fileWatchScan) start() ([]FileEvent, error) {
	info, err := os.Lstat(s.w.root)
	if err != nil {
		return nil, err
	}
	info = StatIdentity(info)
	s.dir = s.w.root
	if !info.IsDir() {
		// A file is watched through its directory, which names it.
		s.file = true
		s.dir, s.name = filepath.Dir(s.w.root), filepath.Base(s.w.root)
	}
	if s.root, err = fileWatchOpenRoot(s.dir); err != nil {
		return nil, err
	}
	// The identity through the handle, as Windows loads that of a path
	// only when it is first compared.
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
	if err = s.watchParent(); err != nil {
		return nil, err
	}
	info = s.identity
	if s.file {
		if info, err = s.root.Lstat(s.name); err != nil {
			return nil, err
		}
	}
	// The initial contents are not events, so what the scan sees is not
	// kept as it was before (touch).
	s.before = nil
	if err = s.visit(s.root, ".", info); err != nil {
		return nil, err
	}
	s.before = map[string]fileWatchBefore{}
	return s.events(), nil
}

// watchParent enrolls the directory of a watched file, and the parent of
// the watched directory on Windows, where the directory's own handle
// reports nothing when it is renamed or replaced; "..", whose notices
// about other entries are ignored. A drive's root, which can't be renamed,
// has no parent.
func (s *fileWatchScan) watchParent() error {
	if !s.file && runtime.GOOS != "windows" {
		return nil
	}
	parent := filepath.Dir(s.w.root)
	if parent == s.w.root {
		return nil
	}
	info, err := os.Lstat(parent)
	if err != nil {
		return err
	}
	return s.add("..", StatIdentity(info))
}

// gone reports whether the watched directory, or the directory of the
// watched file, went away or was replaced.
func (s *fileWatchScan) removedFile() fileWatchGone { return s.removed }

func (s *fileWatchScan) gone() bool {
	info, err := os.Lstat(s.dir)
	return errors.Is(err, fs.ErrNotExist) || err == nil && (!info.IsDir() || !os.SameFile(StatIdentity(info), s.identity))
}

// goneEvents are the last events of a watch whose directory went away.
func (s *fileWatchScan) goneEvents() []FileEvent {
	e, ok := s.entries["."]
	if !ok {
		return nil // a watched file that was already gone
	}
	return []FileEvent{{Op: FileRemove, Path: ".", IsDir: e.isDir()}}
}

// osPath is the path of p for root.
func (s *fileWatchScan) osPath(p string) string {
	if s.file {
		return s.name
	}
	return filepath.FromSlash(p)
}

// base is the name of p in its directory: "." for the watched directory,
// and the watched file's name.
func (s *fileWatchScan) base(p string) string {
	if s.file {
		return s.name
	}
	return path.Base(p)
}

// lstat stats p, without following a symlink, in dir, the directory that
// has it: one call by its name, where the root would open every directory
// above it in turn.
func (s *fileWatchScan) lstat(dir *os.Root, p string) (os.FileInfo, error) {
	info, err := dir.Lstat(s.base(p))
	return info, s.pathError(err, p)
}

// entryInfo returns what entry, child read from d, is, as lstat would.
func (s *fileWatchScan) entryInfo(d *fileWatchDir, entry fs.DirEntry, child string) (os.FileInfo, error) {
	info, err := fileWatchEntryInfo(d.root, entry)
	return info, s.pathError(err, child)
}

// pathError names p, from the root, in err, which names it relative to the
// directory that has it.
func (s *fileWatchScan) pathError(err error, p string) error {
	if e, ok := err.(*os.PathError); ok {
		e.Path = s.osPath(p)
	}
	return err
}

// ignores reports whether p is in a directory Ignore ignored.
func (s *fileWatchScan) ignores(p string) bool {
	for p != "." {
		if s.ignored[path.Dir(p)][path.Base(p)] {
			return true
		}
		p = path.Dir(p)
	}
	return false
}

// skip reports whether Ignore ignores child, and remembers in ignored the
// directories it does.
func (s *fileWatchScan) skip(child string, isDir bool) bool {
	if s.w.opts.Ignore == nil || !s.w.opts.Ignore(child, isDir) {
		return false
	}
	if isDir {
		s.ignoredIn(path.Dir(child))[path.Base(child)] = true
	}
	return true
}

// mark records what n says changed, for the next update. It returns an
// error when n ends the watch.
func (s *fileWatchScan) mark(n platform.FileWatchNotice) error {
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
		name := filepath.Base(s.w.root)
		if n.Kind == platform.FileWatchLost {
			// The next update enrolls it again, or ends the watch.
			s.unenroll("..")
		} else if n.Name != name && (n.Name != "" || !s.file) {
			return nil
		}
		if s.file {
			s.stats["."] = true
			if n.Kind == platform.FileWatchWrite {
				s.writes["."] = true
			}
		} else {
			s.lists["."] = true
		}
		return nil
	}
	if s.file {
		// The file itself (macOS).
		if n.Kind == platform.FileWatchLost {
			s.unenroll(".")
		}
		s.stats["."] = true
		if n.Kind == platform.FileWatchWrite {
			s.writes["."] = true
		}
		return nil
	}
	t := s.targets[p]
	listable := t.directory && (p == "." || s.w.opts.Recursive)
	if n.Kind == platform.FileWatchLost {
		// The next update enrolls it again if it still exists, and looks
		// at everything if it watched the whole tree.
		if p == "." && s.subtree {
			s.all = true
		}
		s.unenroll(p)
		n = platform.FileWatchNotice{Kind: platform.FileWatchDirty}
	}
	if n.Name == "" || !listable {
		if listable {
			s.lists[p] = true
		} else {
			s.stats[p] = true
		}
		if n.Kind == platform.FileWatchWrite {
			s.writes[p] = true
		}
		return nil
	}
	if !validWatchName(n.Name, t.subtree) {
		return errors.New("mygo: invalid name in a file watch notice")
	}
	// Statting a child lists its directory if it appeared, went away or
	// was replaced.
	child := path.Join(p, n.Name)
	if s.ignores(child) {
		return nil
	}
	s.stats[child] = true
	if n.Kind == platform.FileWatchWrite {
		s.writes[child] = true
	}
	if len(s.lists)+len(s.stats)+len(s.writes) > s.w.opts.MaxEntries {
		s.all = true
		clear(s.lists)
		clear(s.stats)
		clear(s.writes)
	}
	return nil
}

// validWatchName reports whether name names a child, or for a subtree a
// path beneath it.
func validWatchName(name string, subtree bool) bool {
	if strings.ContainsRune(name, 0) || strings.ContainsRune(name, filepath.Separator) && filepath.Separator != '/' {
		return false
	}
	if !subtree && strings.Contains(name, "/") {
		return false
	}
	for part := range strings.SplitSeq(name, "/") {
		if part == "" || part == "." || part == ".." {
			return false
		}
	}
	return true
}

// pending reports whether an update has anything to do.
func (s *fileWatchScan) pending() bool {
	return s.all || len(s.lists) > 0 || len(s.stats) > 0 || len(s.writes) > 0
}

// update looks again at what changed and returns the events.
func (s *fileWatchScan) update() ([]FileEvent, error) {
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
	if err = s.watchParent(); err != nil {
		return nil, err
	}
	s.before = map[string]fileWatchBefore{}
	if s.file {
		all := s.all
		s.all = false
		if all || len(s.stats) > 0 {
			if err := s.statFile(); err != nil {
				return nil, err
			}
		}
		clear(s.stats)
		return s.events(), nil
	}
	if s.all {
		s.all = false
		for p, e := range s.entries {
			if e.isDir() && (p == "." || s.w.opts.Recursive) {
				s.lists[p] = true
			}
		}
	}
	for p := range s.stats {
		if err := s.stat(p); err != nil {
			return nil, err
		}
	}
	// Parents first: listing a directory again can replace those below it.
	// Listing a directory lists those to list in it, from it.
	dirs := make([]string, 0, len(s.lists))
	for p := range s.lists {
		dirs = append(dirs, p)
	}
	sort.Slice(dirs, func(i, j int) bool {
		if a, b := strings.Count(dirs[i], "/"), strings.Count(dirs[j], "/"); a != b {
			return a < b
		}
		return dirs[i] < dirs[j]
	})
	for _, p := range dirs {
		if !s.lists[p] || !s.listable(p) {
			continue // listed, removed or replaced with its parent
		}
		delete(s.lists, p)
		if err := s.list(nil, p); err != nil {
			return nil, err
		}
	}
	clear(s.lists)
	clear(s.stats)
	return s.events(), nil
}

// events compares what the last update changed with what it was, then
// forgets both.
func (s *fileWatchScan) events() []FileEvent {
	writes := s.writes
	for p := range s.races {
		writes[p] = true
	}
	for p := range writes {
		if _, ok := s.entries[p]; ok {
			s.touch(p)
		}
	}
	events := fileWatchDiff(s.before, s.entries, writes)
	a := s.before["."]
	b, ok := s.entries["."]
	if s.file && a.ok && ok && !a.entry.isDir() && !b.isDir() && a.entry.id != b.id {
		// A file replaced, as editors save, is written.
		events = []FileEvent{{Op: FileWrite, Path: "."}}
	}
	s.removed = fileWatchGone{}
	if s.file && a.ok && !ok {
		s.removed = fileWatchGone{a.entry.id, true}
	}
	clear(s.before)
	clear(s.writes)
	clear(s.races)
	return events
}

// statFile looks at the watched file again.
func (s *fileWatchScan) statFile() error {
	old, known := s.entries["."]
	info, err := s.root.Lstat(s.name)
	if errors.Is(err, fs.ErrNotExist) {
		if known {
			s.removeTree(".")
		}
		return nil
	}
	if s.inaccessible(".", err) {
		return nil
	}
	if err != nil {
		return err
	}
	if known && old.id == fileWatchEntryOf(info).id {
		s.set(".", info)
		return s.enroll(".", info)
	}
	if known {
		s.removeTree(".")
	}
	return s.visit(s.root, ".", info)
}

// stat looks at p again. For an entry it didn't know, or one that is gone
// or was replaced, it lists the directory that has it instead.
func (s *fileWatchScan) stat(p string) error {
	known := p
	for known != "." && !s.has(known) {
		known = path.Dir(known)
	}
	if known != p && path.Dir(p) == known && s.w.opts.Ignore != nil && s.listable(known) {
		// Something new that Ignore may ignore, which needs no listing.
		info, err := s.root.Lstat(filepath.FromSlash(p))
		if err == nil && s.skip(p, info.IsDir()) {
			return nil
		}
	}
	if known == "." || known != p && s.listable(known) {
		s.lists[known] = true
		return nil
	}
	p = known
	old := s.entries[p]
	info, err := s.root.Lstat(filepath.FromSlash(p))
	if err == nil && old.id == fileWatchEntryOf(info).id {
		s.set(p, info)
		return s.enroll(p, info)
	}
	if s.inaccessible(p, err) {
		return nil
	}
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	s.lists[path.Dir(p)] = true
	return nil
}

// inaccessible reports whether err says p can't be looked at, as a
// directory without permission or, on Windows, a file the system holds
// (pagefile.sys): it is left as it was seen, if at all, rather than
// ending the watch. The watched directory itself can't be.
func (s *fileWatchScan) inaccessible(p string, err error) bool {
	return err != nil && (p != "." || s.file) && fileWatchInaccessible(err)
}

// ignoredIn returns the directories ignored in directory p.
func (s *fileWatchScan) ignoredIn(p string) map[string]bool {
	if s.ignored[p] == nil {
		s.ignored[p] = map[string]bool{}
	}
	return s.ignored[p]
}

// has reports whether the scan knows p.
func (s *fileWatchScan) has(p string) bool {
	_, ok := s.entries[p]
	return ok
}

// listable reports whether p is a directory the watch lists.
func (s *fileWatchScan) listable(p string) bool {
	e, ok := s.entries[p]
	return ok && s.descends(p, e.isDir())
}

// list lists directory p again, in dir, the directory that has it, or
// from the root when dir is nil: what appeared or was replaced in it is
// visited, what is gone removed, and the rest stat'ed again. The
// directories in it still to list are listed from it.
func (s *fileWatchScan) list(dir *os.Root, p string) error {
	name := s.base(p)
	if dir == nil {
		dir, name = s.root, filepath.FromSlash(p)
	}
	d, err := s.openDir(dir, name, p, s.entries[p])
	if err != nil {
		if p == "." {
			return err
		}
		if errors.Is(err, fs.ErrNotExist) || errors.Is(err, errWatchRootChanged) {
			// Gone or replaced: what has it knows.
			return s.list(nil, path.Dir(p))
		}
		if s.inaccessible(p, err) {
			return nil
		}
		return err
	}
	defer d.close()
	if err := s.enroll(p, d.info); err != nil {
		return err
	}
	s.set(p, d.info)
	seen := map[string]bool{}
	delete(s.ignored, p)
	for {
		entries, err := d.file.ReadDir(128)
		for _, entry := range entries {
			name := entry.Name()
			child := path.Join(p, name)
			if s.skip(child, entry.IsDir()) {
				continue
			}
			seen[name] = true
			info, err := s.entryInfo(d, entry, child)
			if errors.Is(err, fs.ErrNotExist) {
				s.races[p] = true
				delete(seen, name)
				continue
			}
			if s.inaccessible(child, err) {
				continue // as it was, if it was known
			}
			if err != nil {
				return err
			}
			if old, ok := s.entries[child]; ok {
				if old.id == fileWatchEntryOf(info).id {
					s.set(child, info)
					if err := s.enroll(child, info); err != nil {
						return err
					}
					continue
				}
				s.removeTree(child)
			}
			if err := s.visit(d.root, child, info); err != nil {
				return err
			}
		}
		if err == io.EOF {
			break
		}
		if s.inaccessible(p, err) {
			return nil // what is gone is not known
		}
		if err != nil {
			return err
		}
	}
	for name := range s.children[p] {
		if !seen[name] {
			s.removeTree(path.Join(p, name))
		}
	}
	for name := range s.children[p] {
		if child := path.Join(p, name); s.lists[child] {
			delete(s.lists, child)
			if s.listable(child) {
				if err := s.list(d.root, child); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// fileWatchDir is a directory of the tree, open to be read: root holds
// what is in it, which is looked at by name, relative to it, and file
// reads it.
type fileWatchDir struct {
	root *os.Root
	file *os.File
	info os.FileInfo
	// owned says root is the directory's own, which close closes, rather
	// than the scan's.
	owned bool
}

func (d *fileWatchDir) close() {
	d.file.Close()
	if d.owned {
		d.root.Close()
	}
}

// openDir opens directory p, at name in dir, which must still be the
// directory want.
func (s *fileWatchScan) openDir(dir *os.Root, name, p string, want fileWatchEntry) (*fileWatchDir, error) {
	d := &fileWatchDir{root: s.root}
	if p != "." {
		root, err := dir.OpenRoot(name)
		if err != nil {
			return nil, s.openError(err, dir, name, p, want)
		}
		d.root, d.owned = root, true
	}
	var err error
	if d.file, err = d.root.Open("."); err != nil {
		if d.owned {
			d.root.Close()
		}
		return nil, s.pathError(err, p)
	}
	if d.info, err = d.file.Stat(); err != nil {
		d.close()
		return nil, err
	}
	if !d.info.IsDir() || want.id != fileWatchEntryOf(d.info).id {
		d.close()
		return nil, &os.PathError{Op: "watch", Path: s.osPath(p), Err: errWatchRootChanged}
	}
	return d, nil
}

// openError is what opening directory p, at name in dir, failing with err
// says: errWatchRootChanged when it is no longer the directory want, as
// when a symlink replaced it, which the root follows only within dir.
func (s *fileWatchScan) openError(err error, dir *os.Root, name, p string, want fileWatchEntry) error {
	if errors.Is(err, fs.ErrNotExist) || s.inaccessible(p, err) {
		return s.pathError(err, p)
	}
	checked, statErr := dir.Lstat(name)
	if statErr != nil {
		return s.pathError(statErr, p)
	}
	if !checked.IsDir() || want.id != fileWatchEntryOf(checked).id {
		return &os.PathError{Op: "watch", Path: s.osPath(p), Err: errWatchRootChanged}
	}
	return s.pathError(err, p)
}

// visit records p, which wasn't known, in dir, the directory that has it,
// and when the watch covers it what is beneath it. An entry removed or
// replaced while it looks is skipped, and its directory reported as
// written.
func (s *fileWatchScan) visit(dir *os.Root, p string, info os.FileInfo) error {
	if err := s.w.ctx.Err(); err != nil {
		return err
	}
	if len(s.entries) >= s.w.opts.MaxEntries || len(s.races) > s.w.opts.MaxEntries {
		return ErrWatchLimit
	}
	gone := func() error {
		s.unenroll(p)
		s.races[path.Dir(p)] = true
		return nil
	}
	// Enroll first, then check the entry is still what was enrolled, so
	// that nothing changes unseen in between. Replacements, as editors'
	// atomic saves do, are retried a few times. What isn't enrolled is
	// watched through its directory, enrolled before it was listed, or
	// not at all, and needs no check.
	var d *fileWatchDir
	for attempt := 0; ; attempt++ {
		if attempt == 3 {
			return gone()
		}
		if !s.enrolls(p, info) {
			s.unenroll(p)
		} else {
			err := s.enroll(p, info)
			if errors.Is(err, fs.ErrPermission) || errors.Is(err, platform.ErrUnsupported) || errors.Is(err, context.Canceled) {
				return err
			}
			checked, statErr := s.lstat(dir, p)
			if (p != "." || s.file) && (errors.Is(statErr, fs.ErrNotExist) || errors.Is(err, fs.ErrNotExist)) {
				return gone()
			}
			if statErr != nil {
				return statErr
			}
			if !os.SameFile(info, checked) {
				if p == "." && !s.file {
					return errWatchRootChanged
				}
				s.races[p] = true
				info = checked
				continue
			}
			if err != nil {
				return err
			}
			info = checked
		}
		if !s.descends(p, info.IsDir()) {
			break
		}
		opened, err := s.openDir(dir, s.base(p), p, fileWatchEntryOf(info))
		if err == nil {
			d, info = opened, opened.info
			break
		}
		if s.inaccessible(p, err) {
			break // known, but not what is in it
		}
		if p == "." {
			return err
		}
		if errors.Is(err, fs.ErrNotExist) {
			return gone()
		}
		if !errors.Is(err, errWatchRootChanged) {
			return err
		}
		s.races[p] = true
		if info, err = s.lstat(dir, p); errors.Is(err, fs.ErrNotExist) {
			return gone()
		} else if err != nil {
			return err
		}
	}
	s.set(p, info)
	if d == nil {
		return nil
	}
	defer d.close()
	for {
		entries, err := d.file.ReadDir(128)
		for _, entry := range entries {
			child := path.Join(p, entry.Name())
			if s.skip(child, entry.IsDir()) {
				continue
			}
			info, err := s.entryInfo(d, entry, child)
			if errors.Is(err, fs.ErrNotExist) {
				s.races[p] = true
				continue
			}
			if s.inaccessible(child, err) {
				continue
			}
			if err != nil {
				return err
			}
			if err := s.visit(d.root, child, info); err != nil {
				return err
			}
		}
		if err == io.EOF || s.inaccessible(p, err) {
			return nil // what it could list
		}
		if err != nil {
			return err
		}
	}
}

// descends reports whether the watch covers what is in p, a directory
// if isDir.
func (s *fileWatchScan) descends(p string, isDir bool) bool {
	// Symlinks and Windows junctions are not directories to Lstat.
	return isDir && !s.file && (p == "." || s.w.opts.Recursive)
}

// touch remembers p as it is before an update changes it; nothing during
// the first scan.
func (s *fileWatchScan) touch(p string) {
	if s.before == nil {
		return
	}
	if _, ok := s.before[p]; !ok {
		e, ok := s.entries[p]
		s.before[p] = fileWatchBefore{e, ok}
	}
}

func (s *fileWatchScan) set(p string, info os.FileInfo) {
	s.touch(p)
	// What an update compares with what came before it, so the identity
	// doesn't follow a later replacement of the entry.
	s.entries[p] = fileWatchEntryOf(info)
	if p != "." {
		dir := path.Dir(p)
		if s.children[dir] == nil {
			s.children[dir] = map[string]bool{}
		}
		s.children[dir][path.Base(p)] = true
	}
}

// removeTree forgets p and what is beneath it.
func (s *fileWatchScan) removeTree(p string) {
	for name := range s.children[p] {
		s.removeTree(path.Join(p, name))
	}
	delete(s.children, p)
	delete(s.ignored, p)
	s.touch(p)
	delete(s.entries, p)
	if p != "." {
		delete(s.children[path.Dir(p)], path.Base(p))
	}
	s.unenroll(p)
}

// enrolls reports whether enroll has the backend watch p, which the scan
// saw as info: not when the root's target covers it, it is a file the
// backend watches through its directory, or a symlink or a special file.
// A watched file is enrolled only while it is a file.
func (s *fileWatchScan) enrolls(p string, info os.FileInfo) bool {
	return (info.Mode().IsRegular() && !s.noFiles || info.IsDir() && !s.file) && (!s.subtree || p == ".")
}

// enroll has the backend watch p, which the scan saw as info, if it
// enrolls it.
func (s *fileWatchScan) enroll(p string, info os.FileInfo) error {
	if !s.enrolls(p, info) {
		s.unenroll(p)
		return nil
	}
	if err := s.add(p, info); !s.inaccessible(p, err) {
		return err
	}
	return nil // seen, but not watched
}

func (s *fileWatchScan) add(p string, info os.FileInfo) error {
	if old, ok := s.targets[p]; ok {
		if old.identity == fileWatchEntryOf(info).id {
			return nil
		}
		s.unenroll(p)
	}
	s.next++
	t := platform.FileWatchTarget{ID: s.next, Path: filepath.Join(s.w.root, filepath.FromSlash(p)), Directory: info.IsDir(), Info: info}
	if p == "." && s.w.opts.Recursive && !s.noSubtree {
		t.Subtree = true
		err := s.w.native.Add(t)
		if errors.Is(err, platform.ErrUnsupported) {
			s.noSubtree = true
			t.Subtree = false
		} else if err != nil {
			return &os.PathError{Op: "watch", Path: t.Path, Err: err}
		} else {
			// Directories below enrolled while the root had no target are
			// covered by it now.
			for q := range s.targets {
				if q != "." && q != ".." {
					s.unenroll(q)
				}
			}
			s.subtree = true
			s.record(p, t)
			return nil
		}
	}
	if err := s.w.native.Add(t); err != nil {
		if !t.Directory && errors.Is(err, platform.ErrUnsupported) {
			// Its directory, enrolled before it was listed, watches it.
			s.noFiles = true
			return nil
		}
		if errors.Is(err, context.Canceled) {
			return err
		}
		return &os.PathError{Op: "watch", Path: t.Path, Err: err}
	}
	s.record(p, t)
	return nil
}

func (s *fileWatchScan) record(p string, t platform.FileWatchTarget) {
	s.targets[p] = fileWatchTargetRef{id: t.ID, identity: fileWatchEntryOf(t.Info).id, directory: t.Directory, subtree: t.Subtree}
	s.ids[t.ID] = p
}

func (s *fileWatchScan) unenroll(p string) {
	if t, ok := s.targets[p]; ok {
		delete(s.targets, p)
		delete(s.ids, t.id)
		s.w.native.Remove(t.id)
		if p == "." {
			s.subtree = false
		}
	}
}

// fileWatchDiff turns what an update changed into events: changed holds
// what it looked at as it was before, entries what is now. An entry whose
// identity moved to another path is a rename, when nothing else has that
// identity.
func fileWatchDiff(changed map[string]fileWatchBefore, entries map[string]fileWatchEntry, writes map[string]bool) []FileEvent {
	removed := make(map[string]fileWatchEntry)
	created := make(map[string]fileWatchEntry)
	for p, was := range changed {
		now, ok := entries[p]
		if was.ok && (!ok || was.entry.id != now.id) {
			removed[p] = was.entry
		}
		if ok && (!was.ok || was.entry.id != now.id) {
			created[p] = now
		}
	}
	type group struct {
		path   string
		events []FileEvent
	}
	var groups []group
	paired := make(map[string]string)
	keys := make([]string, 0, len(removed))
	for p := range removed {
		keys = append(keys, p)
	}
	sort.Strings(keys)
	// Pairing looks each removed entry up by identity: comparing it with
	// every entry instead would make a directory of n files renamed n²
	// comparisons.
	var ids *fileWatchIDs
	if len(removed) > 0 && len(created) > 0 {
		ids = newFileWatchIDs(changed, entries)
	}
	for _, old := range keys {
		a := removed[old]
		// A hard link makes pairing by identity ambiguous: the identity
		// must be the only one of its kind before, after and among what
		// was created.
		match := ""
		if ids != nil {
			if match = ids.pair(a); match != "" {
				if _, ok := created[match]; !ok {
					match = ""
				}
			}
		}
		if match == "" {
			continue
		}
		// A replacement is a removal and a creation even if the old name
		// survives elsewhere.
		if _, exists := entries[old]; exists {
			continue
		}
		if changed[match].ok {
			continue
		}
		// Moved with a directory renamed above it, which says it all.
		descendant := false
		for parent := path.Dir(old); parent != "."; parent = path.Dir(parent) {
			if dest, ok := paired[parent]; ok && match == dest+strings.TrimPrefix(old, parent) {
				descendant = true
				break
			}
		}
		paired[old] = match
		delete(removed, old)
		b := created[match]
		delete(created, match)
		var events []FileEvent
		if !descendant {
			events = append(events, FileEvent{Op: FileRename, OldPath: old, Path: match, IsDir: a.isDir()})
		}
		if !b.isDir() && (writes[old] || writes[match] || fileWatchChanged(a, b)) {
			events = append(events, FileEvent{Op: FileWrite, Path: match})
		}
		groups = append(groups, group{match, events})
	}
	for p, was := range changed {
		var events []FileEvent
		if a, ok := removed[p]; ok {
			events = append(events, FileEvent{Op: FileRemove, Path: p, IsDir: a.isDir()})
		}
		if b, ok := created[p]; ok {
			events = append(events, FileEvent{Op: FileCreate, Path: p, IsDir: b.isDir()})
		}
		if now, ok := entries[p]; was.ok && ok && !now.isDir() && was.entry.id == now.id && (writes[p] || fileWatchChanged(was.entry, now)) {
			events = append(events, FileEvent{Op: FileWrite, Path: p})
		}
		if len(events) > 0 {
			groups = append(groups, group{p, events})
		}
	}
	sort.SliceStable(groups, func(i, j int) bool { return groups[i].path < groups[j].path })
	var result []FileEvent
	for _, g := range groups {
		result = append(result, g.events...)
	}
	return result
}

// fileWatchID is a file's identity as os.SameFile compares it (device and
// inode; volume and file index on Windows), usable as a map key.
type fileWatchID struct{ volume, index uint64 }

// fileWatchIDs counts the entries of an update's before and after by
// identity, keeping the path of one, for pairing renames.
type fileWatchIDs struct {
	before, after map[fileWatchID]fileWatchIDSeen
}

type fileWatchIDSeen struct {
	path  string
	count int
}

// newFileWatchIDs counts what an update looked at, as it was (changed)
// and is (entries).
func newFileWatchIDs(changed map[string]fileWatchBefore, entries map[string]fileWatchEntry) *fileWatchIDs {
	ids := &fileWatchIDs{before: make(map[fileWatchID]fileWatchIDSeen, len(changed)), after: make(map[fileWatchID]fileWatchIDSeen, len(changed))}
	count := func(m map[fileWatchID]fileWatchIDSeen, p string, id fileWatchID) {
		seen := m[id]
		seen.path = p
		seen.count++
		m[id] = seen
	}
	for p, was := range changed {
		if was.ok {
			count(ids.before, p, was.entry.id)
		}
		if now, ok := entries[p]; ok {
			count(ids.after, p, now.id)
		}
	}
	return ids
}

// pair returns the path after has for a, an entry before, when a's
// identity is the only one of its kind before and after; "" otherwise.
func (ids *fileWatchIDs) pair(a fileWatchEntry) string {
	if ids.before[a.id].count != 1 {
		return ""
	}
	if seen := ids.after[a.id]; seen.count == 1 {
		return seen.path
	}
	return ""
}

// fileWatchChanged reports whether a file's content may have changed, as
// its size or modification time says; its permissions and other metadata,
// which backup tools, indexers and antivirus touch, don't count.
func fileWatchChanged(a, b fileWatchEntry) bool {
	return a.size != b.size || a.mtime != b.mtime
}
