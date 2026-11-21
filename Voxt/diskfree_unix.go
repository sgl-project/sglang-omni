//go:build !windows

package main

import (
	"path/filepath"
	"syscall"
)

// diskFreeImpl is the space free for the user on path's disk.
func diskFreeImpl(path string) int64 {
	var st syscall.Statfs_t
	for p := path; p != "" && p != "/" && p != "."; p = filepath.Dir(p) {
		if syscall.Statfs(p, &st) == nil {
			return int64(st.Bavail) * int64(st.Bsize)
		}
	}
	if syscall.Statfs("/", &st) == nil {
		return int64(st.Bavail) * int64(st.Bsize)
	}
	return 0
}
