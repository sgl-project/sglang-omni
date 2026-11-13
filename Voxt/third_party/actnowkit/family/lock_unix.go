//go:build unix

package family

import (
	"errors"
	"os"
	"syscall"
)

func flock(f *os.File, mode LockMode, wait bool) error {
	how := syscall.LOCK_SH
	if mode == Exclusive {
		how = syscall.LOCK_EX
	}
	if !wait {
		how |= syscall.LOCK_NB
	}
	for {
		err := syscall.Flock(int(f.Fd()), how)
		switch {
		case err == nil:
			return nil
		case errors.Is(err, syscall.EINTR):
			continue
		case errors.Is(err, syscall.EWOULDBLOCK):
			return ErrLocked
		default:
			return &os.PathError{Op: "flock", Path: f.Name(), Err: err}
		}
	}
}

func funlock(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}

// ProcessAlive reports whether a process with this pid exists (it may be
// another user's). Pids are reused: use it only to ignore stale hints
// (lock owners, runtime files), never as proof an app runs.
func ProcessAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}
