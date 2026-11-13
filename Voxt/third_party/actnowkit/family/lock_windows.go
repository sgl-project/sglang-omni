//go:build windows

package family

import (
	"errors"
	"os"
	"sync"

	"golang.org/x/sys/windows"
)

type lockHandle struct {
	path string
	mode LockMode
}

var lockModes = struct {
	sync.Mutex
	modes map[uintptr]lockHandle
}{modes: map[uintptr]lockHandle{}}

func flock(f *os.File, mode LockMode, wait bool) error {
	flags := uint32(0)
	if mode == Exclusive {
		flags = windows.LOCKFILE_EXCLUSIVE_LOCK
	}
	if !wait {
		flags |= windows.LOCKFILE_FAIL_IMMEDIATELY
	}
	_ = funlock(f) // Convert drops the old lock before changing its mode.
	lockModes.Lock()
	for handle, held := range lockModes.modes {
		if handle != f.Fd() && held.path == f.Name() && (held.mode == Exclusive || mode == Exclusive) {
			lockModes.Unlock()
			return ErrLocked
		}
	}
	lockModes.modes[f.Fd()] = lockHandle{path: f.Name(), mode: mode}
	lockModes.Unlock()
	overlapped := new(windows.Overlapped)
	if err := windows.LockFileEx(windows.Handle(f.Fd()), flags, 0, 1, 0, overlapped); err != nil {
		lockModes.Lock()
		delete(lockModes.modes, f.Fd())
		lockModes.Unlock()
		if errors.Is(err, windows.ERROR_LOCK_VIOLATION) || errors.Is(err, windows.ERROR_IO_PENDING) {
			return ErrLocked
		}
		return &os.PathError{Op: "flock", Path: f.Name(), Err: err}
	}
	return nil
}

func funlock(f *os.File) error {
	lockModes.Lock()
	delete(lockModes.modes, f.Fd())
	lockModes.Unlock()
	err := windows.UnlockFileEx(windows.Handle(f.Fd()), 0, 1, 0, new(windows.Overlapped))
	if errors.Is(err, windows.ERROR_NOT_LOCKED) {
		return nil
	}
	return err
}

// ProcessAlive reports whether a process with this pid exists; on this
// system it cannot tell and says true for any positive pid.
func ProcessAlive(pid int) bool { return pid > 0 }
