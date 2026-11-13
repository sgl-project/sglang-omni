//go:build windows

package main

import (
	"path/filepath"

	"golang.org/x/sys/windows"
)

// diskFreeImpl is the space free for the user on path's disk.
func diskFreeImpl(path string) int64 {
	for p := path; ; p = filepath.Dir(p) {
		if free, ok := volumeFree(p); ok {
			return free
		}
		next := filepath.Dir(p)
		if next == p {
			return 0
		}
	}
}

// volumeFree is the bytes free for the caller on dir's volume.
func volumeFree(dir string) (int64, bool) {
	dirPtr, err := windows.UTF16PtrFromString(dir)
	if err != nil {
		return 0, false
	}
	var free, total, totalFree uint64
	if err := windows.GetDiskFreeSpaceEx(dirPtr, &free, &total, &totalFree); err != nil {
		return 0, false
	}
	return int64(free), true
}
