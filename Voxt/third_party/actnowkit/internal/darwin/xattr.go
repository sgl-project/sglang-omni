//go:build darwin

package darwin

import (
	"runtime"
	"syscall"
)

// errno reads the calling thread's errno (the caller locks the thread).
func errno() syscall.Errno {
	if p := errnoLoc(); p != nil {
		return syscall.Errno(*p)
	}
	return syscall.EIO
}


// SetXattr sets an extended attribute of path (following symlinks).
func SetXattr(path, name string, value []byte) error {
	if err := loadLibSystem(); err != nil {
		return err
	}
	var p *byte
	if len(value) > 0 {
		p = &value[0]
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if setxattr(path, name, p, uintptr(len(value)), 0, 0) != 0 {
		return &pathError{"setxattr", path, errno()}
	}
	return nil
}

// GetXattr reads an extended attribute; ok is false when it is not set.
func GetXattr(path, name string) (value []byte, ok bool, err error) {
	if err := loadLibSystem(); err != nil {
		return nil, false, err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	n := getxattr(path, name, nil, 0, 0, 0)
	if n < 0 {
		if e := errno(); e != syscall.ENOATTR {
			return nil, false, &pathError{"getxattr", path, e}
		}
		return nil, false, nil
	}
	buf := make([]byte, n+1)
	n = getxattr(path, name, &buf[0], uintptr(len(buf)), 0, 0)
	if n < 0 {
		if e := errno(); e != syscall.ENOATTR {
			return nil, false, &pathError{"getxattr", path, e}
		}
		return nil, false, nil
	}
	return buf[:n], true, nil
}

// RemoveXattr removes an extended attribute; not set is not an error.
func RemoveXattr(path, name string) error {
	if err := loadLibSystem(); err != nil {
		return err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if removexattr(path, name, 0) != 0 {
		if e := errno(); e != syscall.ENOATTR {
			return &pathError{"removexattr", path, e}
		}
	}
	return nil
}

// ProcPath is the executable of a running process (proc_pidpath).
func ProcPath(pid int) (string, error) {
	if err := loadLibSystem(); err != nil {
		return "", err
	}
	buf := make([]byte, 4096) // PROC_PIDPATHINFO_MAXSIZE
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	n := procPidpath(int32(pid), &buf[0], uint32(len(buf)))
	if n <= 0 {
		return "", errno()
	}
	return string(buf[:n]), nil
}

type pathError struct {
	op, path string
	err      syscall.Errno
}

func (e *pathError) Error() string { return e.op + " " + e.path + ": " + e.err.Error() }
func (e *pathError) Unwrap() error { return e.err }
