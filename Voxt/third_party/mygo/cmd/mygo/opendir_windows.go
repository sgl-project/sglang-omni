//go:build windows

package main

import (
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

// openDir opens a directory to list it, sharing it for deletion, which
// os.Open does not: while mygo dev's watcher lists a directory, deleting
// or renaming it would fail ("being used by another process").
func openDir(name string) (*os.File, error) {
	path := name
	// Where Windows does not enable long paths, os lifts MAX_PATH with
	// this prefix.
	if abs, err := filepath.Abs(name); err == nil && len(abs) >= 248 && !strings.HasPrefix(abs, `\\`) {
		path = `\\?\` + abs
	}
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: name, Err: err}
	}
	h, err := syscall.CreateFile(p, syscall.GENERIC_READ,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		nil, syscall.OPEN_EXISTING, syscall.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: name, Err: err}
	}
	return os.NewFile(uintptr(h), name), nil
}
