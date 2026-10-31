//go:build unix

package mygo

import (
	"os"
	"syscall"
)

// fileLinks returns how many hard links a file has.
func fileLinks(info os.FileInfo) int {
	if st, ok := info.Sys().(*syscall.Stat_t); ok {
		return int(st.Nlink)
	}
	return 1
}

// openShared opens the file at path to read it, which other apps may
// rename or replace meanwhile, as everywhere on Unix.
func openShared(path string) (*os.File, error) { return os.Open(path) }

// fileWritable reports whether the user may write the file at path.
func fileWritable(path string) bool { return syscall.Access(path, 2 /* W_OK */) == nil }

// lockFile takes an exclusive lock on f, which fails while another
// process holds it, until f is closed.
func lockFile(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
}
