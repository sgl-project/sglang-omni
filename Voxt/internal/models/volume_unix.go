//go:build !windows

package models

import (
	"os"
	"syscall"
)

// SameVolume reports whether a and b are folders on one volume, where
// a model is hard-linked instead of copied.
func SameVolume(a, b string) bool {
	sa, err := os.Stat(a)
	if err != nil {
		return false
	}
	sb, err := os.Stat(b)
	if err != nil {
		return false
	}
	da, ok := sa.Sys().(*syscall.Stat_t)
	db, ok2 := sb.Sys().(*syscall.Stat_t)
	return ok && ok2 && da.Dev == db.Dev
}
