//go:build windows

package hfhub

import (
	"os"
	"path/filepath"

	"golang.org/x/sys/windows"
)

// freeBytes is the space available to the user on the volume of dir (or
// of its nearest existing parent), -1 when unknown.
func freeBytes(dir string) int64 {
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
