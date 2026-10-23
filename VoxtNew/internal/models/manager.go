package models

// Manager: which models are installed (in memory, read when it opens and
// on Refresh), and their downloads, each on a goroutine of its own,
// through actnowkit's hfhub into the family's model library.

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/hehehai/actnowkit/hfhub"
	fammodels "github.com/hehehai/actnowkit/models"
)

// ErrUnknown: no model of the catalogue has that id.
var ErrUnknown = errors.New("models: unknown model")

// ErrBorrowed: a borrowed model is not downloaded by VoxtNew.
var ErrBorrowed = errors.New("models: another app's model")

// Options configure a Manager.
type Options struct {
	// Store is VoxtNew's view of the family's model library: its $HUB,
	// its claims, the other apps' (required).
	Store *fammodels.Store
	// Network is how downloads reach the Hub (SetNetwork changes it).
	Network Network
	// ReserveBytes is the room a download leaves free on the disk
	// (0: hfhub's 2 GB; negative: no check).
	ReserveBytes int64
	// Catalog replaces the catalogue (tests: conformance's repos).
	Catalog []Model
	// OnChange is called after a status changed, on any goroutine; at
	// most every 200 ms while bytes arrive.
	OnChange func()
}

// Network is the downloads' way to the Hub: the endpoints raced (nil:
// hfhub's, $HF_ENDPOINT or huggingface.co and hf-mirror.com) and the HTTP
// client (nil: hfhub's, through the system's proxy).
type Network struct {
	Endpoints []string
	Client    *http.Client
}

// Status is a model's state.
type Status struct {
	// Installed: on disk at a commit VoxtNew runs, and VoxtNew's to use:
	// VoxtNew claims it, or it is one of VoxtNew's own models no other app
	// claims. OnDisk: on disk, whether or not (another app's model VoxtNew
	// may use: "Use them", family spec §7.10.1).
	Installed, OnDisk bool
	// Revision is the commit on disk ("" when none is).
	Revision    string
	Downloading bool
	// Done and Total are a download's bytes (of the model and the models
	// it needs); Done is also what a stopped download kept.
	Done, Total int64
	// WaitingFor is the bundle id of the app downloading what this
	// download waits for (hfhub.UnknownOwner when unknown).
	WaitingFor string
	// Elsewhere is the bundle id of another app downloading the model
	// while VoxtNew does not (family spec §7.10.4).
	Elsewhere string
	// Err is the last download's failure, until the next starts.
	Err error
	// Mine: VoxtNew claims it. Users are the other apps that claim it
	// (installed ones, names for "Shared with …").
	Mine  bool
	Users []string
	// DeletedBy is the bundle id of the app that deleted it for all apps,
	// while that is why it is gone.
	DeletedBy string
}

// Manager keeps the catalogue's models in the family's library.
// Goroutine-safe.
type Manager struct {
	store   *fammodels.Store
	opt     Options
	catalog []Model

	refreshMu sync.Mutex // one Refresh at a time: a slow one never writes over a later one
	mu        sync.Mutex
	client    *hfhub.Client
	status    map[string]Status // by id; downloads' bytes live in jobs
	jobs      map[string]*job
	busy      map[string]bool // ids a job is installing (a speech model's VAD: one job)
	lastNote  time.Time
	closed    bool
}

// job is one download: a model and the models it needs.
type job struct {
	cancel  context.CancelFunc
	done    chan struct{}
	total   int64
	base    int64 // bytes of the entries finished
	current int64 // of the entry downloading
	waiting string
}

// Open reads which models are on disk and which VoxtNew claims (I/O:
// call it while the app starts, not while a view builds).
func Open(opt Options) *Manager {
	if opt.Store == nil {
		panic("models: Open without a Store")
	}
	m := &Manager{store: opt.Store, opt: opt, catalog: opt.Catalog,
		status: map[string]Status{}, jobs: map[string]*job{}, busy: map[string]bool{}}
	if m.catalog == nil {
		m.catalog = catalog
	}
	m.client = m.newClient(opt.Network)
	m.refresh()
	return m
}

func (m *Manager) newClient(n Network) *hfhub.Client {
	return hfhub.New(hfhub.Options{Store: m.store, Endpoints: n.Endpoints, HTTPClient: n.Client,
		ReserveBytes: m.opt.ReserveBytes, UserAgent: "VoxtNew"})
}

// SetNetwork changes the way to the Hub; downloads already running keep
// theirs.
func (m *Manager) SetNetwork(n Network) {
	c := m.newClient(n)
	m.mu.Lock()
	m.client = c
	m.mu.Unlock()
}

// Hub is the family's model library VoxtNew uses ($HUB).
func (m *Manager) Hub() string { return m.store.Hub }

// Store is the family's view of it.
func (m *Manager) Store() *fammodels.Store { return m.store }

// Lookup is the model of id in the Manager's catalogue.
func (m *Manager) Lookup(id string) (Model, bool) {
	for _, mod := range m.catalog {
		if mod.ID == id {
			return mod, true
		}
	}
	return Model{}, false
}

// Catalog is the Manager's catalogue.
func (m *Manager) Catalog() []Model { return slices.Clone(m.catalog) }

// Installed says whether model id is VoxtNew's to use (memory only).
func (m *Manager) Installed(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.status[id].Installed
}

// Dir is the snapshot folder of model id, "" when it is not installed.
func (m *Manager) Dir(id string) string {
	mod, ok := m.Lookup(id)
	m.mu.Lock()
	s := m.status[id]
	m.mu.Unlock()
	if !ok || !s.Installed {
		return ""
	}
	return SnapshotDir(m.store.Hub, mod, s.Revision)
}

// Status is model id's state (memory only).
func (m *Manager) Status(id string) Status {
	m.mu.Lock()
	defer m.mu.Unlock()
	s := m.status[id]
	s.Users = slices.Clone(s.Users)
	if j := m.jobs[id]; j != nil {
		s.Downloading, s.Done, s.Total, s.WaitingFor = true, j.base+j.current, j.total, j.waiting
	}
	return s
}

// Refresh reads the disk and the family's claims again (another app
// downloaded or deleted a model, or the user came back to VoxtNew): it
// drops VoxtNew's claims on models deleted for all apps (tombstones). It
// blocks on I/O: call it off the main thread.
func (m *Manager) Refresh() {
	m.refresh()
	m.note(true)
}

func (m *Manager) refresh() {
	m.refreshMu.Lock()
	defer m.refreshMu.Unlock()
	tombs, _ := m.store.ApplyTombstones()
	all, _ := m.store.Claims()
	survey := map[fammodels.Ref][]string{}
	var want []fammodels.Claim
	for _, mod := range m.catalog {
		want = append(want, mod.Claim("", ""))
		for _, rev := range mod.Accept {
			want = append(want, mod.Claim(rev, ""))
		}
	}
	for _, f := range m.store.Survey(want...) {
		for _, u := range f.By {
			survey[f.Want.Ref()] = append(survey[f.Want.Ref()], u.Name)
		}
	}
	next := map[string]Status{}
	for _, mod := range m.catalog {
		var s Status
		rev, _ := find(m.store.Hub, mod)
		ref := mod.Ref(rev)
		if rev != "" {
			s.OnDisk, s.Revision, s.Users = true, rev, survey[ref]
			_, s.Mine = all.Find(ref)
			// VoxtNew's to use: claimed by it, or one of its own models no
			// other app claims (a borrowed model, or one another app
			// installed, is VoxtNew's once the user chose to use it).
			s.Installed = s.Mine || !mod.Borrowed && len(s.Users) == 0
		} else if !mod.Borrowed {
			if l, err := hfhub.Status(m.store.Hub, mod.Entry("")); err == nil && l.State == hfhub.Partial {
				s.Done = l.Bytes
			}
			if o, ok := fammodels.Activity(m.store.Hub, mod.Repo, m.store.BundleID); ok && o.Op == fammodels.OpDownload {
				s.Elsewhere = o.BundleID
			}
			for _, t := range tombs {
				if t.Ref() == mod.Ref("") {
					s.DeletedBy = t.RemovedBy
				}
			}
		}
		next[mod.ID] = s
	}
	m.mu.Lock()
	for id, s := range next {
		old := m.status[id]
		if s.DeletedBy == "" && !s.OnDisk {
			s.DeletedBy = old.DeletedBy // a tombstone is applied once: keep saying it
		}
		s.Err = old.Err
		m.status[id] = s
	}
	m.mu.Unlock()
}

// Download starts installing model id and the models it needs, unless
// it is installed or downloading. Failures are the Status's Err.
func (m *Manager) Download(id string) error {
	mod, ok := m.Lookup(id)
	if !ok {
		return fmt.Errorf("%w: %q", ErrUnknown, id)
	}
	if mod.Borrowed {
		return fmt.Errorf("%w: %q", ErrBorrowed, id)
	}
	m.mu.Lock()
	if m.closed || m.status[id].OnDisk || m.jobs[id] != nil {
		m.mu.Unlock()
		return nil
	}
	var list []Model
	for _, need := range mod.Needs {
		if n, ok := m.Lookup(need); ok && !m.status[need].OnDisk && !m.busy[need] {
			list = append(list, n)
		}
	}
	list = append(list, mod)
	ctx, cancel := context.WithCancel(context.Background())
	j := &job{cancel: cancel, done: make(chan struct{})}
	for _, l := range list {
		j.total += l.Size()
		m.busy[l.ID] = true
	}
	m.jobs[id] = j
	s := m.status[id]
	s.Err, s.DeletedBy = nil, ""
	m.status[id] = s
	client := m.client
	m.mu.Unlock()
	m.note(true)
	go m.run(ctx, client, id, j, list)
	return nil
}

// Cancel stops model id's download; what came is kept for the next (this
// app's or another's).
func (m *Manager) Cancel(id string) {
	m.mu.Lock()
	j := m.jobs[id]
	m.mu.Unlock()
	if j != nil {
		j.cancel()
		<-j.done
	}
}

// Close stops every download (what came is kept).
func (m *Manager) Close() {
	m.mu.Lock()
	m.closed = true
	var ids []string
	for id := range m.jobs {
		ids = append(ids, id)
	}
	m.mu.Unlock()
	for _, id := range ids {
		m.Cancel(id)
	}
}

func (m *Manager) run(ctx context.Context, client *hfhub.Client, id string, j *job, list []Model) {
	var err error
	for _, mod := range list {
		err = client.Download(ctx, mod.Entry(""), func(p hfhub.Progress) {
			m.mu.Lock()
			j.current, j.waiting = min(p.Bytes, mod.Size()), p.WaitingFor
			m.mu.Unlock()
			m.note(false)
		})
		if err != nil {
			break
		}
		m.mu.Lock()
		j.base, j.current, j.waiting = j.base+mod.Size(), 0, ""
		m.mu.Unlock()
	}
	m.refresh() // before the job ends: never "not downloading" and "not installed" in between
	m.mu.Lock()
	for _, mod := range list {
		delete(m.busy, mod.ID)
	}
	delete(m.jobs, id)
	if err != nil && !errors.Is(err, context.Canceled) {
		s := m.status[id]
		s.Err = err
		m.status[id] = s
	}
	m.mu.Unlock()
	close(j.done)
	m.note(true)
}

// note tells OnChange; a note that is not forced waits 200 ms after the
// last.
func (m *Manager) note(force bool) {
	if m.opt.OnChange == nil {
		return
	}
	m.mu.Lock()
	now := time.Now()
	if !force && now.Sub(m.lastNote) < 200*time.Millisecond {
		m.mu.Unlock()
		return
	}
	m.lastNote = now
	m.mu.Unlock()
	m.opt.OnChange()
}
