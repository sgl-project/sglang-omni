//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package download

import (
	"os"
	"syscall"
)

// Lock takes an exclusive lock on the file path, which it creates, and
// returns what releases it. When another process (or another call) holds
// it, Lock calls wait, if set, then waits for it. The system releases the
// lock too when the process ends, so a process that dies downloading
// leaves no stale lock behind.
func Lock(path string, wait func()) (unlock func(), err error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	how := syscall.LOCK_EX | syscall.LOCK_NB
	for {
		err = syscall.Flock(int(f.Fd()), how)
		if err == syscall.EWOULDBLOCK && how&syscall.LOCK_NB != 0 {
			if wait != nil {
				wait()
			}
			how = syscall.LOCK_EX
			continue
		}
		if err != syscall.EINTR {
			break
		}
	}
	if err != nil {
		f.Close()
		return nil, &os.PathError{Op: "flock", Path: path, Err: err}
	}
	return func() {
		syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		f.Close()
	}, nil
}
