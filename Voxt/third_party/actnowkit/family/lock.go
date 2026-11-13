package family

import (
	"context"
	"errors"
	"os"
	"time"
)

// LockMode is a flock(2) mode.
type LockMode int

const (
	Shared    LockMode = iota // LOCK_SH
	Exclusive                 // LOCK_EX
)

func (m LockMode) String() string {
	if m == Exclusive {
		return "exclusive"
	}
	return "shared"
}

// ErrLocked: another process (or another open of the file in this one)
// holds the lock in a conflicting mode.
var ErrLocked = errors.New("locked by another process")

// ErrLockUnsupported: this system has no flock(2).
var ErrLockUnsupported = errors.New("file locks are not supported on this system")

// Lock is a held flock(2) lock (spec §5.3). The kernel releases it when
// the process exits. Lock files are never deleted: a new file would be a
// new inode, and two processes could each "hold" the lock.
//
// flock locks belong to an open file description, so two Locks taken in
// one process on the same path exclude each other like two processes do.
type Lock struct {
	f    *os.File
	Path string
	Mode LockMode
}

// TryLock takes the lock at path without waiting; ErrLocked when it is
// held in a conflicting mode. The file is created (perm) when missing; its
// directory must exist.
func TryLock(path string, mode LockMode, perm os.FileMode) (*Lock, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, perm)
	if err != nil {
		// A lock file another user or tool made read-only still locks.
		if f, err = os.Open(path); err != nil {
			return nil, err
		}
	}
	if err := flock(f, mode, false); err != nil {
		f.Close()
		return nil, err
	}
	return &Lock{f: f, Path: path, Mode: mode}, nil
}

// WaitLock takes the lock at path, waiting until it is free or ctx ends.
// It polls (5 ms, growing to 100 ms) so that waiting can be cancelled.
func WaitLock(ctx context.Context, path string, mode LockMode, perm os.FileMode) (*Lock, error) {
	delay := 5 * time.Millisecond
	for {
		l, err := TryLock(path, mode, perm)
		if !errors.Is(err, ErrLocked) {
			return l, err
		}
		t := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			t.Stop()
			return nil, ctx.Err()
		case <-t.C:
		}
		delay = min(delay*2, 100*time.Millisecond)
	}
}

// Unlock releases the lock. Safe to call twice and on nil.
func (l *Lock) Unlock() error {
	if l == nil || l.f == nil {
		return nil
	}
	err := funlock(l.f)
	if cerr := l.f.Close(); err == nil {
		err = cerr
	}
	l.f = nil
	return err
}

// Convert changes the mode of the held lock in place (flock(2) on the
// same descriptor), waiting while another holder's lock conflicts, until
// ctx ends. The kernel does not make this atomic: it may drop the old
// lock first and grant a waiting process its lock in between, so re-check
// what the lock protects afterwards. When it fails, the lock is no longer
// held in either mode: Unlock it.
func (l *Lock) Convert(ctx context.Context, mode LockMode) error {
	if l == nil || l.f == nil {
		return errors.New("family: convert of a released lock")
	}
	if l.Mode == mode {
		return nil
	}
	delay := 5 * time.Millisecond
	for {
		err := flock(l.f, mode, false)
		if err == nil {
			l.Mode = mode
			return nil
		}
		if !errors.Is(err, ErrLocked) {
			return err
		}
		t := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-t.C:
		}
		delay = min(delay*2, 100*time.Millisecond)
	}
}

// WithLock runs fn holding locks/<name>.lock exclusively: the
// read-modify-write of a multi-writer file (spec §5.3). Keep fn short (no
// network, no dialogs): the spec asks for under 100 ms.
func (d Dir) WithLock(ctx context.Context, name string, fn func() error) error {
	if err := os.MkdirAll(d.LocksDir(), 0o700); err != nil {
		return err
	}
	l, err := WaitLock(ctx, d.LockFile(name), Exclusive, 0o600)
	if err != nil {
		return err
	}
	defer l.Unlock()
	return fn()
}

// Update is the read-modify-write of the multi-writer file at path
// (spec §5.3) under locks/<lock>.lock: v (a pointer to the file's zero
// value) is loaded from path, fn changes it, and it is written back
// atomically. A missing file leaves v zero (exists false); a corrupt one
// is quarantined and treated as missing. A file of a newer major is not
// touched: ErrNewerSchema. fn may return ErrNoChange to skip the write.
func (d Dir) Update(ctx context.Context, lock, path, kind string, major int, v any, fn func(exists bool) error) error {
	return d.WithLock(ctx, lock, func() error {
		err := Load(path, kind, major, v)
		exists := err == nil
		var ce *CorruptError
		switch {
		case err == nil, IsNotExist(err):
		case errors.As(err, &ce):
			if err := Quarantine(path); err != nil {
				return err
			}
		default:
			return err
		}
		if err := fn(exists); err != nil {
			if errors.Is(err, ErrNoChange) {
				return nil
			}
			return err
		}
		return WriteJSON(path, v)
	})
}

// ErrNoChange, returned by an Update callback, skips the write.
var ErrNoChange = errors.New("no change")
