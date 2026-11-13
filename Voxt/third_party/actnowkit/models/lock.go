package models

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/hehehai/actnowkit/family"
)

// Operations named in lock owner files.
const (
	OpDownload = "download"
	OpVerify   = "verify"
	OpDelete   = "delete"
	OpClaim    = "claim"
	OpFinish   = "finish" // completing a snapshot (rename, refs)
)

// Owner says who holds a lock: written beside it as <lock file>.owner so
// that other apps can show "Downloading in AudioSmith…" (spec §7.4). Only
// a hint: flock decides.
type Owner struct {
	BundleID string       `json:"bundleId"`
	PID      int          `json:"pid"`
	Since    time.Time    `json:"since"`
	Op       string       `json:"op"`
	Extra    family.Extra `json:"-"`
}

func (o *Owner) UnmarshalJSON(b []byte) error {
	type plain Owner
	return family.UnmarshalKnown(b, (*plain)(o), &o.Extra)
}

func (o Owner) MarshalJSON() ([]byte, error) {
	type plain Owner
	return family.MarshalKnown(plain(o), o.Extra)
}

// BusyError: a lock is held by another process (or another open in this
// one). Owner is what its owner file says, nil when unknown. It matches
// family.ErrLocked.
type BusyError struct {
	Lock  string
	Owner *Owner
}

func (e *BusyError) Error() string {
	if e.Owner != nil && e.Owner.BundleID != "" {
		return fmt.Sprintf("%s is held by %s (%s)", filepath.Base(e.Lock), e.Owner.BundleID, e.Owner.Op)
	}
	return filepath.Base(e.Lock) + " is held by another process"
}

func (e *BusyError) Is(target error) bool { return target == family.ErrLocked }

// HeldLock is a repo or blob lock this process holds.
type HeldLock struct {
	lock  *family.Lock
	path  string
	owner Owner
}

// Path is the lock file.
func (l *HeldLock) Path() string { return l.path }

// Mode is the lock's current mode.
func (l *HeldLock) Mode() family.LockMode { return l.lock.Mode }

// ReadOwner reads the owner file of a lock; false when there is none or
// its process is gone.
func ReadOwner(lockPath string) (Owner, bool) {
	data, err := family.ReadLimited(lockPath + ".owner")
	if err != nil {
		return Owner{}, false
	}
	var o Owner
	if json.Unmarshal(data, &o) != nil || !family.ProcessAlive(o.PID) {
		return Owner{}, false
	}
	return o, true
}

func lockPerm(path string) error { return os.MkdirAll(filepath.Dir(path), 0o755) }

func take(ctx context.Context, path string, mode family.LockMode, who Owner, wait bool) (*HeldLock, error) {
	if err := lockPerm(path); err != nil {
		return nil, err
	}
	var l *family.Lock
	var err error
	if wait {
		l, err = family.WaitLock(ctx, path, mode, 0o644)
	} else {
		l, err = family.TryLock(path, mode, 0o644)
	}
	if errors.Is(err, family.ErrLocked) {
		be := &BusyError{Lock: path}
		if o, ok := ReadOwner(path); ok {
			be.Owner = &o
		}
		return nil, be
	}
	if err != nil {
		return nil, err
	}
	h := &HeldLock{lock: l, path: path, owner: fillOwner(who)}
	h.writeOwner()
	return h, nil
}

func fillOwner(o Owner) Owner {
	if o.PID == 0 {
		o.PID = os.Getpid()
	}
	if o.Since.IsZero() {
		o.Since = family.Now()
	}
	return o
}

func (l *HeldLock) writeOwner() {
	if data, err := json.Marshal(l.owner); err == nil {
		_ = family.WriteFileAtomic(l.path+".owner", data, 0o644)
	}
}

// Release drops the lock (and its owner file, when it is still this
// process's). Safe on nil and twice.
func (l *HeldLock) Release() error {
	if l == nil || l.lock == nil {
		return nil
	}
	if o, ok := ReadOwner(l.path); ok && o.PID == l.owner.PID && o.BundleID == l.owner.BundleID && o.Op == l.owner.Op && o.Since.Equal(l.owner.Since) {
		os.Remove(l.path + ".owner")
	}
	err := l.lock.Unlock()
	l.lock = nil
	return err
}

// Upgrade turns a shared repo lock into an exclusive one, waiting for the
// other holders to finish (spec §7.4: release SH, then take EX; the caller
// re-checks the state afterwards, since others may have acted in between).
func (l *HeldLock) Upgrade(ctx context.Context, op string) error {
	if l.lock == nil {
		return errors.New("models: upgrade of a released lock")
	}
	if l.lock.Mode == family.Exclusive {
		return nil
	}
	l.lock.Unlock()
	nl, err := family.WaitLock(ctx, l.path, family.Exclusive, 0o644)
	if err != nil {
		l.lock = nil
		return err
	}
	l.lock = nl
	l.owner.Op, l.owner.Since = op, family.Now()
	l.writeOwner()
	return nil
}

// Downgrade turns an exclusive repo lock into a shared one, e.g. to keep
// holding a repo (against deletion) while waiting for others, after
// finishing its snapshot. Not atomic either (family.Lock.Convert): re-check
// afterwards. On error the lock is no longer held; Release it.
func (l *HeldLock) Downgrade(ctx context.Context, op string) error {
	if l.lock == nil {
		return errors.New("models: downgrade of a released lock")
	}
	if l.lock.Mode == family.Shared {
		return nil
	}
	if err := l.lock.Convert(ctx, family.Shared); err != nil {
		return err
	}
	l.owner.Op, l.owner.Since = op, family.Now()
	l.writeOwner()
	return nil
}

// TryLockRepo takes the repo lock without waiting: *BusyError when held.
// Downloads and claims take it Shared; finishing a snapshot and deleting
// take it Exclusive (spec §7.4). Deleting never waits: tell the user to
// try again (TextBusy).
func TryLockRepo(hub, repo string, mode family.LockMode, who Owner) (*HeldLock, error) {
	if !ValidRepo(repo) {
		return nil, fmt.Errorf("%q: %w", repo, errBadRepo)
	}
	return take(context.Background(), RepoLockFile(hub, repo), mode, who, false)
}

// LockRepo is TryLockRepo waiting until the lock is free or ctx ends.
func LockRepo(ctx context.Context, hub, repo string, mode family.LockMode, who Owner) (*HeldLock, error) {
	if !ValidRepo(repo) {
		return nil, fmt.Errorf("%q: %w", repo, errBadRepo)
	}
	return take(ctx, RepoLockFile(hub, repo), mode, who, true)
}

// TryLockBlob takes a blob's lock (exclusive) without waiting. A
// *BusyError means another app (or a Python tool) is downloading the same
// blob: wait for it (LockBlob), then check whether the blob is complete.
func TryLockBlob(hub, repo, blob string, who Owner) (*HeldLock, error) {
	if !ValidRepo(repo) || !validBlob(blob) {
		return nil, fmt.Errorf("%s %q: %w", repo, blob, errBadRepo)
	}
	return take(context.Background(), BlobLockFile(hub, repo, blob), family.Exclusive, who, false)
}

// LockBlob is TryLockBlob waiting until the lock is free or ctx ends.
func LockBlob(ctx context.Context, hub, repo, blob string, who Owner) (*HeldLock, error) {
	if !ValidRepo(repo) || !validBlob(blob) {
		return nil, fmt.Errorf("%s %q: %w", repo, blob, errBadRepo)
	}
	return take(ctx, BlobLockFile(hub, repo, blob), family.Exclusive, who, true)
}

var errBadRepo = errors.New("not a Hugging Face repo id (owner/name) or blob id")

func validBlob(b string) bool {
	if b == "" || len(b) > 128 {
		return false
	}
	for _, c := range b {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F' || c == '-' || c == '_') {
			return false
		}
	}
	return true
}

// Activity is what another app is doing with a repo, from the owner
// files of its locks (the repo lock's, else any blob lock's), skipping
// self (a bundle id). ok is false when nobody is, or the owners are gone.
// Use it for "Downloading in AudioSmith… 43%" (TextDownloadingIn); the
// progress can be estimated from the size of the .incomplete blobs.
func Activity(hub, repo, self string) (Owner, bool) {
	if o, ok := ReadOwner(RepoLockFile(hub, repo)); ok && o.BundleID != self {
		return o, true
	}
	matches, _ := filepath.Glob(filepath.Join(LocksDir(hub, repo), "*.lock.owner"))
	for _, m := range matches {
		lock := m[:len(m)-len(".owner")]
		if o, ok := ReadOwner(lock); ok && o.BundleID != self {
			return o, true
		}
	}
	return Owner{}, false
}
