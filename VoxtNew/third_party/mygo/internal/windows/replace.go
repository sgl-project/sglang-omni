//go:build windows && (amd64 || arm64)

package windows

import (
	"os"
	"syscall"
	"time"
	"unsafe"
)

var procReplaceFileW = kernel32.NewProc("ReplaceFileW")

const (
	replacefileIgnoreMergeErrors = 0x2
	replacefileIgnoreACLErrors   = 0x4

	errorUnableToRemoveReplaced  = syscall.Errno(1175)
	errorUnableToMoveReplacement = syscall.Errno(1176)
	errorSharingViolation        = syscall.Errno(32)
	errorLockViolation           = syscall.Errno(33)
)

// ReplaceFile replaces target with ReplaceFileW, which keeps the target's
// security descriptor, attributes, alternate data streams and creation
// time, as Office and NSDocument's replacement do. Another program holding
// the target open without sharing deletes (an indexer, an antivirus, a
// cloud client) gets a moment to let go.
func (*Backend) ReplaceFile(target, replacement string) error {
	if _, err := os.Lstat(target); os.IsNotExist(err) {
		return os.Rename(replacement, target)
	}
	t, err := syscall.UTF16PtrFromString(target)
	if err != nil {
		return err
	}
	r, err := syscall.UTF16PtrFromString(replacement)
	if err != nil {
		return err
	}
	for attempt := 0; ; attempt++ {
		ok, _, callErr := procReplaceFileW.Call(uintptr(unsafe.Pointer(t)), uintptr(unsafe.Pointer(r)), 0,
			replacefileIgnoreMergeErrors|replacefileIgnoreACLErrors, 0, 0)
		if ok != 0 {
			return nil
		}
		errno, _ := callErr.(syscall.Errno)
		retry := errno == errorSharingViolation || errno == errorLockViolation ||
			errno == syscall.ERROR_ACCESS_DENIED || errno == errorUnableToRemoveReplaced
		if !retry || attempt == 9 {
			if errno == errorUnableToMoveReplacement {
				// The target stays as it was; the replacement is left.
				return &os.LinkError{Op: "replace", Old: replacement, New: target, Err: errno}
			}
			return &os.LinkError{Op: "replace", Old: replacement, New: target, Err: callErr}
		}
		time.Sleep(50 * time.Millisecond)
	}
}
