//go:build !windows

package main

import (
	"os"
	"syscall"
)

func lockSnapshot(file *os.File, nonBlocking bool) error {
	operation := syscall.LOCK_EX
	if nonBlocking {
		operation |= syscall.LOCK_NB
	}
	return syscall.Flock(int(file.Fd()), operation)
}

func unlockSnapshot(file *os.File) error {
	return syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
}
