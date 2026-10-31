package mygo

import (
	"encoding/json/v2"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// The recovery copies of documents: what a document's unsaved changes
// are, written a second after the user stops editing into the app's
// data, as VS Code's backups, Zed's database and gnome-text-editor's
// drafts keep them, so that a crash, a forced quit or a quit that keeps
// documents (DocumentsOptions.AskOnQuit unset) loses nothing. The next
// launch opens them again, edited, and a file that changed meanwhile is
// in conflict, as its stamp says.
//
// Each kind of documents keeps them in a directory of PathUserData,
// "Recovery/<kind>", in a session of the running instance, which a lock
// file tells is alive: another instance of the app leaves it alone, and
// the next launch takes the sessions no instance holds.

type recoveryEntry struct {
	ID string `json:"id"`
	// Path is the document's file; Name the name of an untitled one.
	Path  string    `json:"path,omitzero"`
	Name  string    `json:"name,omitzero"`
	Stamp fileStamp `json:"stamp"`
}

type recoveryStore struct {
	k *docKind

	mu      sync.Mutex
	dir     string // the session, "" until the first copy
	lock    *os.File
	entries map[string]recoveryEntry
}

func newRecoveryStore(k *docKind) *recoveryStore {
	return &recoveryStore{k: k, entries: map[string]recoveryEntry{}}
}

// root is the directory of the kind's sessions.
func (s *recoveryStore) root() (string, error) {
	data, err := App.Path(PathUserData)
	if err != nil {
		return "", err
	}
	t := s.k.types[0]
	name := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' {
			return r
		}
		return '-'
	}, t.Name+"-"+t.Extensions[0])
	return filepath.Join(data, "Recovery", name), nil
}

// session returns the directory of this instance's session, which it
// creates and locks the first time. s.mu is held.
func (s *recoveryStore) session() (string, error) {
	if s.dir != "" {
		return s.dir, nil
	}
	root, err := s.root()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(root, randomText()[:12])
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	lock, err := os.OpenFile(filepath.Join(dir, "lock"), os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return "", err
	}
	if err := lockFile(lock); err != nil {
		lock.Close()
		return "", err
	}
	s.dir, s.lock = dir, lock
	return dir, nil
}

// save writes the copy of a document and its entry. b.io is held.
func (s *recoveryStore) save(e recoveryEntry, write func(io.Writer) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	dir, err := s.session()
	if err != nil {
		return err
	}
	if err := writeAppFile(filepath.Join(dir, e.ID), write); err != nil {
		return err
	}
	s.entries[e.ID] = e
	return s.writeIndex()
}

// remove forgets the copy of b.
func (s *recoveryStore) remove(b *docBase) {
	b.mu.Lock()
	id := b.id
	b.backedUp = 0
	b.mu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.entries[id]; !ok {
		return
	}
	delete(s.entries, id)
	if err := s.writeIndex(); err != nil {
		log.Printf("mygo: documents: %v", err)
	}
	os.Remove(filepath.Join(s.dir, id))
}

// writeIndex writes the entries of the session. s.mu is held.
func (s *recoveryStore) writeIndex() error {
	list := make([]recoveryEntry, 0, len(s.entries))
	for _, e := range s.entries {
		list = append(list, e)
	}
	return writeAppFile(filepath.Join(s.dir, "index.json"), func(w io.Writer) error {
		return json.MarshalWrite(w, list)
	})
}

// close removes the session when it keeps no copy, at quit.
func (s *recoveryStore) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lock == nil {
		return
	}
	s.lock.Close()
	s.lock = nil
	if len(s.entries) == 0 {
		os.RemoveAll(s.dir)
	}
	s.dir = ""
	clear(s.entries)
}

// orphan is a copy that no running instance holds.
type orphan struct {
	entry recoveryEntry
	file  string
}

// claim takes the sessions that no instance holds: those of an instance
// that crashed or quit keeping documents. It returns their copies and a
// function removing the sessions, once the documents are restored.
func (s *recoveryStore) claim() (orphans []orphan, done func()) {
	root, err := s.root()
	if err != nil {
		return nil, func() {}
	}
	s.mu.Lock()
	own := s.dir
	s.mu.Unlock()
	dirs, _ := os.ReadDir(root)
	var claimed []string
	var locks []*os.File
	for _, d := range dirs {
		dir := filepath.Join(root, d.Name())
		if !d.IsDir() || dir == own {
			continue
		}
		lock, err := os.OpenFile(filepath.Join(dir, "lock"), os.O_RDWR|os.O_CREATE, 0o600)
		if err != nil {
			continue
		}
		if lockFile(lock) != nil {
			lock.Close() // alive
			continue
		}
		locks = append(locks, lock)
		claimed = append(claimed, dir)
		var entries []recoveryEntry
		if data, err := readAppFile(filepath.Join(dir, "index.json")); err == nil {
			if err := json.Unmarshal(data, &entries); err != nil {
				log.Printf("mygo: documents: %s: %v", dir, err)
			}
		}
		for _, e := range entries {
			if e.ID != "" && !strings.ContainsAny(e.ID, `/\.`) {
				orphans = append(orphans, orphan{e, filepath.Join(dir, e.ID)})
			}
		}
	}
	return orphans, func() {
		for i, dir := range claimed {
			locks[i].Close()
			os.RemoveAll(dir)
		}
	}
}

// backup writes the recovery copy of b, if it changed since the last.
func (b *docBase) backup() {
	snap := b.ops.capture()
	b.io.Lock()
	defer b.io.Unlock()
	b.mu.Lock()
	if !b.edited || snap.at == b.backedUp {
		b.mu.Unlock()
		return
	}
	e := recoveryEntry{ID: b.id, Path: b.path, Stamp: b.stamp}
	if b.path == "" {
		e.Name = b.name
	}
	b.mu.Unlock()
	if err := b.kind.recovery.save(e, snap.write); err != nil {
		log.Printf("mygo: documents: cannot write the recovery copy of %s: %v", e.Name+e.Path, err)
		return
	}
	b.mu.Lock()
	b.backedUp = snap.at
	b.dirtySince = time.Time{}
	if b.changes != snap.at {
		b.dirtySince = time.Now() // edited while it was written
	}
	b.mu.Unlock()
}

// restore opens the documents of the sessions no instance holds, edited,
// and keeps their copies in this session.
func (k *docKind) restore() int {
	orphans, done := k.recovery.claim()
	n := 0
	for _, o := range orphans {
		e := o.entry
		if e.Path != "" && openDocument(e.Path) != nil {
			continue // a file the app was started with, say
		}
		f, err := os.Open(o.file)
		if err != nil {
			continue
		}
		v, err := k.read(f)
		f.Close()
		if err != nil {
			log.Printf("mygo: documents: cannot read the recovery copy of %s: %v", e.Name+e.Path, err)
			continue
		}
		b, err := k.create(v, e.Name)
		if err != nil {
			continue
		}
		b.id = e.ID
		b.ops.restore(v)
		if e.Path != "" {
			b.path, b.stamp = e.Path, e.Stamp
			switch now, exists, _ := statStamp(e.Path); {
			case !exists:
				b.deleted = true
			case now != e.Stamp:
				b.conflict = true
			}
			b.readOnly = !b.deleted && !fileWritable(e.Path)
		} else if n := untitledNumber(e.Name); n > 0 {
			b.untitled = n
		}
		b.dirtySince = time.Now()
		b.backup()
		k.show(b)
		n++
	}
	done()
	return n
}

// writeAppFile writes a file of the app's own: a new file renamed over
// it, so that it is whole or not there.
func writeAppFile(path string, write func(io.Writer) error) error {
	f, err := createSibling(path)
	if err != nil {
		return err
	}
	err = writeAndSync(f, write)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = rename(f.Name(), path)
	}
	if err != nil {
		os.Remove(f.Name())
	}
	return err
}
