//go:build windows

package models

import (
	"os"
	"path/filepath"
	"syscall"

	"golang.org/x/sys/windows"
)

// freeSpace is the space available to the user on dir's volume (or its
// nearest existing parent's), -1 when unknown.
func freeSpace(dir string) int64 {
	for {
		path, err := windows.UTF16PtrFromString(dir)
		if err == nil {
			var available, total, free uint64
			if err := windows.GetDiskFreeSpaceEx(path, &available, &total, &free); err == nil {
				return int64(available)
			} else if !os.IsNotExist(err) {
				return -1
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return -1
		}
		dir = parent
	}
}

// linkedElsewhere reports a file with hard links beyond its own name and
// the snapshot entries that resolve to it.
func linkedElsewhere(path string, _ os.FileInfo, snapshotLinks uint32) bool {
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()
	var info syscall.ByHandleFileInformation
	if err := syscall.GetFileInformationByHandle(syscall.Handle(file.Fd()), &info); err != nil {
		return false
	}
	return info.NumberOfLinks > snapshotLinks+1
}
