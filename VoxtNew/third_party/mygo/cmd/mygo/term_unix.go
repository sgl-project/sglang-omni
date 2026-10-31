//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package main

import (
	"os"
	"syscall"
	"unsafe"
)

// termWidth returns the number of columns of the terminal f, or 0.
func termWidth(f *os.File) int {
	if f == nil {
		return 0
	}
	var ws struct{ rows, cols, x, y uint16 }
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, f.Fd(), syscall.TIOCGWINSZ, uintptr(unsafe.Pointer(&ws))); errno != 0 {
		return 0
	}
	return int(ws.cols)
}

// enableVT makes the terminal f interpret escape sequences, which Unix
// terminals do.
func enableVT(f *os.File) bool { return true }
