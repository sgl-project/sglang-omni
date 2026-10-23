//go:build windows

package main

import (
	"os"
	"syscall"
	"unsafe"
)

var (
	kernel32                       = syscall.NewLazyDLL("kernel32.dll")
	procGetConsoleScreenBufferInfo = kernel32.NewProc("GetConsoleScreenBufferInfo")
	procSetConsoleMode             = kernel32.NewProc("SetConsoleMode")
)

// termWidth returns the number of columns of the console f, or 0.
func termWidth(f *os.File) int {
	if f == nil {
		return 0
	}
	var info struct {
		size, cursor             [2]int16
		attributes               uint16
		left, top, right, bottom int16
		maxSize                  [2]int16
	}
	if r, _, _ := procGetConsoleScreenBufferInfo.Call(f.Fd(), uintptr(unsafe.Pointer(&info))); r == 0 {
		return 0
	}
	return int(info.right - info.left + 1)
}

// enableVT makes the console f interpret escape sequences, as terminals
// do, and reports whether it does.
func enableVT(f *os.File) bool {
	const enableVirtualTerminalProcessing = 0x4
	var mode uint32
	if syscall.GetConsoleMode(syscall.Handle(f.Fd()), &mode) != nil {
		return false
	}
	if mode&enableVirtualTerminalProcessing != 0 {
		return true
	}
	r, _, _ := procSetConsoleMode.Call(f.Fd(), uintptr(mode|enableVirtualTerminalProcessing))
	return r != 0
}
