package actnow

// VoxtNew's runtime file (runtime/<bundle id>.json, spec §11.1): the
// models its runtimes hold (the omni servers', voxtnew-llm's), each with
// its repo, commit, worker, bytes and since when it is idle; its memory
// budget against the other apps'; its paste in progress. The runtimes
// report through Loaded, Busy, Idle and Unloaded; the file is rewritten
// on each change and removed when VoxtNew quits.

import (
	"cmp"
	"context"
	"maps"
	"slices"
	"sync"
	"time"

	famruntime "github.com/hehehai/actnowkit/runtime"
)

// Runtime is VoxtNew's runtime file. Goroutine-safe; each change writes
// the file (I/O: off the main thread).
type Runtime struct {
	f *Family

	// wmu orders the writes: each writes the state of its time, the last
	// one the latest.
	wmu     sync.Mutex
	mu      sync.Mutex
	loaded  map[string]famruntime.Loaded // by model id
	pasting bool
	closed  bool // removed: VoxtNew quits, nothing is written any more
	// unload lets a runtime go of an idle model on a memory request
	// (OnUnload), by model id.
	unload func(id string)
	now    func() time.Time
	// physical is the Mac's memory (tests set it).
	physical func() uint64
	// ownWait is how long Reserve waits for VoxtNew's own idle models to
	// leave, othersWait how long it gives the other apps.
	ownWait, othersWait time.Duration
}

func newRuntime(f *Family) *Runtime {
	return &Runtime{f: f, loaded: map[string]famruntime.Loaded{}, now: time.Now,
		physical: famruntime.PhysicalMemory, ownWait: 3 * time.Second, othersWait: time.Second}
}

// OnUnload sets what lets go of model id when another app asks for
// memory (spec §11.1: idle models only).
func (r *Runtime) OnUnload(fn func(id string)) {
	r.mu.Lock()
	r.unload = fn
	r.mu.Unlock()
}

// Loaded says a runtime (worker "omni", "llm") holds model id, from
// repo@revision, taking about bytes; it is busy until Idle.
func (r *Runtime) Loaded(id, worker, repo, revision string, bytes int64) {
	r.change(func() {
		r.loaded[id] = famruntime.Loaded{Repo: repo, Revision: revision, Worker: worker, Bytes: bytes}
	})
}

// Busy says model id works (a session uses it).
func (r *Runtime) Busy(id string) { r.setIdle(id, false) }

// Idle says model id finished its work.
func (r *Runtime) Idle(id string) { r.setIdle(id, true) }

func (r *Runtime) setIdle(id string, idle bool) {
	r.change(func() {
		l, ok := r.loaded[id]
		if !ok || (l.IdleSince != nil) == idle {
			return
		}
		l.IdleSince = nil
		if idle {
			t := r.now().UTC().Truncate(time.Second)
			l.IdleSince = &t
		}
		r.loaded[id] = l
	})
}

// Unloaded says model id left memory.
func (r *Runtime) Unloaded(id string) {
	r.change(func() { delete(r.loaded, id) })
}

// change applies fn and writes the file when it changed something.
func (r *Runtime) change(fn func()) {
	r.mu.Lock()
	before := maps.Clone(r.loaded)
	fn()
	same := maps.EqualFunc(before, r.loaded, sameLoaded)
	r.mu.Unlock()
	if !same {
		r.write()
	}
}

func sameLoaded(a, b famruntime.Loaded) bool {
	return a.Repo == b.Repo && a.Revision == b.Revision && a.Worker == b.Worker && a.Bytes == b.Bytes &&
		(a.IdleSince == nil) == (b.IdleSince == nil) && (a.IdleSince == nil || a.IdleSince.Equal(*b.IdleSince))
}

// LoadedBytes is the memory VoxtNew's models take.
func (r *Runtime) LoadedBytes() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	var n int64
	for _, l := range r.loaded {
		n += l.Bytes
	}
	return n
}

// Budget is VoxtNew's memory budget for models (spec §11.1): the family's
// share of the physical memory less what the other running apps hold
// (VoxtNew sets no limit of its own).
func (r *Runtime) Budget() int64 {
	return famruntime.Budget(0, famruntime.FamilyCap(r.physical()), famruntime.OthersLoadedBytes(r.f.dir, r.f.bundleID))
}

// Reserve makes room before a runtime loads model id of bytes (spec
// §11.1): when it does not fit the budget, VoxtNew's own idle models go
// first (the largest first, until it fits; Reserve waits for them to
// leave), then the other apps are asked to unload theirs and are given a
// moment; the model loads anyway afterwards (it never blocks the user).
// Idle is the family's: for famruntime.UnloadIdleAfter, so that a
// session's models (the speech model idle between its live stream and
// its Final, the language model warming up meanwhile) never unload each
// other.
func (r *Runtime) Reserve(ctx context.Context, id string, bytes int64) {
	need := bytes - (r.Budget() - r.othersOf(id))
	if need <= 0 {
		return
	}
	r.mu.Lock()
	unload := r.unload
	r.mu.Unlock()
	if unload != nil {
		idle := r.idle(famruntime.UnloadIdleAfter, id)
		slices.SortStableFunc(idle, func(a, b idleModel) int { return cmp.Compare(b.bytes, a.bytes) })
		var gone []string
		for _, m := range idle {
			if need <= 0 {
				break
			}
			r.f.logf("family: unloading %s (idle) to load %s", m.id, id)
			unload(m.id)
			gone = append(gone, m.id)
			need -= m.bytes
		}
		r.waitGone(ctx, gone)
	}
	if need <= 0 || r.f.readOnly {
		return
	}
	r.f.logf("family: asking the other apps for memory to load %s", id)
	if err := famruntime.RequestMemory(r.f.dir); err != nil {
		r.f.logf("family: memory request: %v", err)
	}
	select {
	case <-ctx.Done():
	case <-time.After(r.othersWait):
	}
}

// waitGone waits, at most ownWait, until models ids left the file (their
// runtimes said Unloaded).
func (r *Runtime) waitGone(ctx context.Context, ids []string) {
	deadline := time.Now().Add(r.ownWait)
	for len(ids) > 0 && time.Now().Before(deadline) && ctx.Err() == nil {
		r.mu.Lock()
		left := slices.ContainsFunc(ids, func(id string) bool { _, ok := r.loaded[id]; return ok })
		r.mu.Unlock()
		if !left {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// othersOf is what VoxtNew's models but id take.
func (r *Runtime) othersOf(id string) int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	var n int64
	for k, l := range r.loaded {
		if k != id {
			n += l.Bytes
		}
	}
	return n
}

// OnMemory answers another app's memory request (family.TopicMemory):
// the models idle for famruntime.UnloadIdleAfter go; a model in use
// stays.
func (r *Runtime) OnMemory() {
	r.mu.Lock()
	unload := r.unload
	r.mu.Unlock()
	if unload == nil {
		return
	}
	for _, m := range r.idle(famruntime.UnloadIdleAfter, "") {
		r.f.logf("family: unloading %s (idle) for another app", m.id)
		unload(m.id)
	}
}

// idleModel is a model idle long enough to go.
type idleModel struct {
	id    string
	bytes int64
}

// idle lists the models idle for at least after, but keep, by id.
func (r *Runtime) idle(after time.Duration, keep string) []idleModel {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.now()
	var out []idleModel
	for _, id := range slices.Sorted(maps.Keys(r.loaded)) {
		l := r.loaded[id]
		if id != keep && l.IdleSince != nil && now.Sub(*l.IdleSince) >= after {
			out = append(out, idleModel{id, l.Bytes})
		}
	}
	return out
}

// Paste runs fn, a paste through the clipboard (write it, ⌘V, restore
// it), after another app's paste ends (at most famruntime.PasteWait), and
// says so in the file while it runs, so that the other voice-input apps
// wait for it (spec §11.1).
func (r *Runtime) Paste(ctx context.Context, fn func() error) error {
	famruntime.WaitPasting(ctx, r.f.dir, r.f.bundleID)
	r.mu.Lock()
	r.pasting = true
	r.mu.Unlock()
	r.write()
	defer func() {
		r.mu.Lock()
		r.pasting = false
		r.mu.Unlock()
		r.write()
	}()
	return fn()
}

// write writes the runtime file.
func (r *Runtime) write() {
	if r.f.readOnly {
		return
	}
	r.wmu.Lock()
	defer r.wmu.Unlock()
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return
	}
	ids := slices.Sorted(maps.Keys(r.loaded))
	loaded := make([]famruntime.Loaded, 0, len(ids))
	for _, id := range ids {
		loaded = append(loaded, r.loaded[id])
	}
	file := famruntime.File{BundleID: r.f.bundleID, Loaded: loaded, Pasting: r.pasting}
	r.mu.Unlock()
	file.BudgetBytes = r.Budget()
	if err := famruntime.Write(r.f.dir, file); err != nil {
		r.f.logf("family: runtime: %v", err)
	}
}

// remove deletes the runtime file (VoxtNew quits).
func (r *Runtime) remove() {
	r.wmu.Lock()
	defer r.wmu.Unlock()
	r.mu.Lock()
	r.closed = true
	r.mu.Unlock()
	if r.f.readOnly {
		return
	}
	if err := famruntime.Remove(r.f.dir, r.f.bundleID); err != nil {
		r.f.logf("family: runtime: %v", err)
	}
}
