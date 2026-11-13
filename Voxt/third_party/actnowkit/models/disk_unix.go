//go:build darwin || linux

package models

import (
	"os"
	"path/filepath"
	"syscall"
)

// freeSpace is the space available to the user on dir's volume (or its
// nearest existing parent's), -1 when unknown.
func freeSpace(dir string) int64 {
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

// linkedElsewhere reports a file with other hard links (an imported blob
// still in the Hugging Face cache): removing this name frees nothing.
func linkedElsewhere(fi os.FileInfo) bool {
	st, ok := fi.Sys().(*syscall.Stat_t)
	return ok && st.Nlink > 1
}
