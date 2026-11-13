//go:build darwin || linux

package hfhub

import (
	"os"
	"path/filepath"
	"syscall"
)

// freeBytes is the space available to the user on the volume of dir (or
// of its nearest existing parent), -1 when unknown.
func freeBytes(dir string) int64 {
	for {
		var st syscall.Statfs_t
		if err := syscall.Statfs(dir, &st); err == nil {
			return int64(uint64(st.Bavail) * uint64(st.Bsize))
		} else if !os.IsNotExist(err) {
			return -1
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return -1
		}
		dir = parent
	}
}
