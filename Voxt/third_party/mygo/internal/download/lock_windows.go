package download

import (
	"os"
	"syscall"
	"unsafe"
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procLockFileEx   = kernel32.NewProc("LockFileEx")
	procUnlockFileEx = kernel32.NewProc("UnlockFileEx")
)

const (
	lockfileFailImmediately = 0x1
	lockfileExclusiveLock   = 0x2
	allBytes                = uintptr(^uint32(0))

	errorLockViolation syscall.Errno = 33 // ERROR_LOCK_VIOLATION
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
	// The whole file, as cmd/go's lockedfile locks it. The handle is
	// synchronous, so LockFileEx returns once it holds the lock.
	ol := new(syscall.Overlapped)
	lock := func(flags uintptr) error {
		if r, _, e := procLockFileEx.Call(f.Fd(), flags, 0, allBytes, allBytes, uintptr(unsafe.Pointer(ol))); r == 0 {
			return e
		}
		return nil
	}
	err = lock(lockfileExclusiveLock | lockfileFailImmediately)
	if err == errorLockViolation {
		if wait != nil {
			wait()
		}
		err = lock(lockfileExclusiveLock)
	}
	if err != nil {
		f.Close()
		return nil, &os.PathError{Op: "LockFileEx", Path: path, Err: err}
	}
	return func() {
		procUnlockFileEx.Call(f.Fd(), 0, allBytes, allBytes, uintptr(unsafe.Pointer(ol)))
		f.Close()
	}, nil
}
