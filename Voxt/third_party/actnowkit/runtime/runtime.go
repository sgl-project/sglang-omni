// Package runtime is the family's running-process files and memory
// coordination (spec §11.1): each app that loads local models writes
// runtime/<bundle id>.json (pid, budget, loaded models, the "pasting"
// flag), and budgets its memory against what the other running apps hold.
//
//	cap := runtime.FamilyCap(runtime.PhysicalMemory())
//	budget := runtime.Budget(ownSetting, cap, runtime.OthersLoadedBytes(d, self))
//	runtime.Write(d, runtime.File{BundleID: self, BudgetBytes: budget, Loaded: loaded})
//
// The package is named after the spec's directory; import it under
// another name (famruntime) beside the standard library's runtime.
package runtime

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/hehehai/actnowkit/family"
)

// Kind is the schema kind of runtime/<bundle id>.json.
const Kind = "runtime"

// Loaded is a model a process holds in memory.
type Loaded struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
	Worker   string `json:"worker,omitempty"` // the app's engine worker: "listen", "gpu"…
	Bytes    int64  `json:"bytes"`
	// IdleSince is when the model last finished work; nil = busy.
	IdleSince *time.Time   `json:"idleSince"`
	Extra     family.Extra `json:"-"`
}

func (l *Loaded) UnmarshalJSON(b []byte) error {
	type plain Loaded
	return family.UnmarshalKnown(b, (*plain)(l), &l.Extra)
}

func (l Loaded) MarshalJSON() ([]byte, error) {
	type plain Loaded
	return family.MarshalKnown(plain(l), l.Extra)
}

// File is runtime/<bundle id>.json. Only its app writes it; it is
// removed when the app quits, and ignored when its process is gone.
type File struct {
	Schema   string `json:"schema"`
	BundleID string `json:"bundleId"`
	PID      int    `json:"pid"`
	// Exe is the writer's executable (v0.8.0): a reader takes the file
	// for a live app only when the process with PID runs this executable,
	// so a reused pid does not keep a crashed app's models counted
	// (spec §6.3, §11.1). Absent in files of older writers.
	Exe         string    `json:"exe,omitempty"`
	Updated     time.Time `json:"updated"`
	BudgetBytes int64     `json:"budgetBytes"`
	Loaded      []Loaded  `json:"loaded"`
	// Pasting is set while the app writes the clipboard, sends ⌘V and
	// restores the clipboard (voice-input apps); others wait for it.
	Pasting bool         `json:"pasting,omitempty"`
	Extra   family.Extra `json:"-"`
}

func (f *File) UnmarshalJSON(b []byte) error {
	type plain File
	return family.UnmarshalKnown(b, (*plain)(f), &f.Extra)
}

func (f File) MarshalJSON() ([]byte, error) {
	type plain File
	return family.MarshalKnown(plain(f), f.Extra)
}

// LoadedBytes is the memory the file says its models take.
func (f File) LoadedBytes() int64 {
	var n int64
	for _, l := range f.Loaded {
		n += l.Bytes
	}
	return n
}

// Write writes the app's runtime file (schema, pid and time filled in)
// and posts TopicRuntime. Call it after loading or unloading a model, and
// around pasting.
func Write(d family.Dir, f File) error {
	if f.BundleID == "" {
		return errors.New("runtime: no bundle id")
	}
	f.Schema = family.SchemaOf(Kind, 1)
	if f.PID == 0 {
		f.PID = os.Getpid()
		if f.Exe == "" {
			f.Exe, _ = os.Executable()
		}
	}
	f.Updated = family.Now()
	if f.Loaded == nil {
		f.Loaded = []Loaded{}
	}
	if err := family.WriteJSON(d.RuntimeFile(f.BundleID), f); err != nil {
		return err
	}
	_ = d.Post(family.TopicRuntime)
	return nil
}

// Remove deletes the app's runtime file (at quit).
func Remove(d family.Dir, bundleID string) error {
	err := os.Remove(d.RuntimeFile(bundleID))
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	_ = d.Post(family.TopicRuntime)
	return nil
}

// Read reads one app's runtime file.
func Read(d family.Dir, bundleID string) (File, error) {
	var f File
	err := family.Load(d.RuntimeFile(bundleID), Kind, 1, &f)
	if errors.Is(err, family.ErrNewerSchema) {
		err = nil
	}
	return f, err
}

// ReadAll reads the runtime files of processes that still exist. Files of
// gone processes (a crash) are skipped, not deleted. A file is gone when
// its pid does not run, when it was written before the system started,
// or when its pid now runs another executable than the file's exe (a
// reused pid).
func ReadAll(d family.Dir) ([]File, []error) {
	entries, err := os.ReadDir(d.RuntimeDir())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, []error{err}
	}
	var out []File
	var errs []error
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || strings.HasPrefix(name, ".") || !strings.HasSuffix(name, ".json") {
			continue
		}
		f, err := Read(d, strings.TrimSuffix(name, ".json"))
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if !Live(f) {
			continue
		}
		out = append(out, f)
	}
	return out, errs
}

// Live reports whether a runtime file belongs to a running process (see
// ReadAll).
func Live(f File) bool {
	if !family.ProcessAlive(f.PID) {
		return false
	}
	if bt := bootTime(); !bt.IsZero() && !f.Updated.IsZero() && f.Updated.Before(bt.Add(-time.Minute)) {
		return false
	}
	if f.Exe != "" {
		if exe := procExe(f.PID); exe != "" && !sameExe(exe, f.Exe) {
			return false
		}
	}
	return true
}

func sameExe(a, b string) bool {
	if a == b {
		return true
	}
	ra, err1 := filepath.EvalSymlinks(a)
	rb, err2 := filepath.EvalSymlinks(b)
	return err1 == nil && err2 == nil && ra == rb
}

// Others is ReadAll without self (a bundle id).
func Others(d family.Dir, self string) []File {
	all, _ := ReadAll(d)
	out := all[:0]
	for _, f := range all {
		if f.BundleID != self {
			out = append(out, f)
		}
	}
	return out
}

// OthersLoadedBytes is the memory the other running apps' models take.
func OthersLoadedBytes(d family.Dir, self string) int64 {
	var n int64
	for _, f := range Others(d, self) {
		n += f.LoadedBytes()
	}
	return n
}

// LoadedBy lists the other running apps that hold repo@revision loaded
// ("Also loaded in Dicta", TextAlsoLoadedIn).
func LoadedBy(d family.Dir, self, repo, revision string) []File {
	var out []File
	for _, f := range Others(d, self) {
		for _, l := range f.Loaded {
			if l.Repo == repo && (revision == "" || l.Revision == revision) {
				out = append(out, f)
				break
			}
		}
	}
	return out
}

// DefaultFamilyShare is the part of physical memory the family's models
// may take together (spec §11.1).
const DefaultFamilyShare = 0.6

// UnloadIdleAfter: on TopicMemory, apps unload models idle this long.
const UnloadIdleAfter = 60 * time.Second

// FamilyCap is DefaultFamilyShare of physical memory.
func FamilyCap(physical uint64) int64 { return int64(float64(physical) * DefaultFamilyShare) }

// Budget is an app's model memory budget (spec §11.1):
// min(own, familyCap − othersLoaded), never negative. own ≤ 0 means the
// app sets no limit of its own.
func Budget(own, familyCap, othersLoaded int64) int64 {
	b := familyCap - othersLoaded
	if own > 0 && own < b {
		b = own
	}
	return max(b, 0)
}

// RequestMemory asks the other apps to unload models idle for
// UnloadIdleAfter (spec §11.1). Retry the load once afterwards (after a
// short wait, or on TopicRuntime), and then load anyway.
func RequestMemory(d family.Dir) error { return d.Post(family.TopicMemory) }

// Idle reports whether a loaded model may be unloaded on a memory request.
func Idle(l Loaded, now time.Time) bool {
	return l.IdleSince != nil && now.Sub(*l.IdleSince) >= UnloadIdleAfter
}

// PasteWait is how long an app waits for another one's paste.
const PasteWait = 500 * time.Millisecond

// WaitPasting waits while another running app has Pasting set, at most
// PasteWait (or until ctx ends), polling every 20 ms. It reports whether
// the way is clear.
func WaitPasting(ctx context.Context, d family.Dir, self string) bool {
	deadline := time.Now().Add(PasteWait)
	for {
		busy := false
		for _, f := range Others(d, self) {
			if f.Pasting {
				busy = true
				break
			}
		}
		if !busy {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(20 * time.Millisecond):
		}
	}
}

// Reference texts.
var (
	// TextAlsoLoadedIn: on a loaded model's row (spec §11.1).
	TextAlsoLoadedIn = family.Text{EN: "Also loaded in {app}", ZH: "{app} 也已加载"}
)
