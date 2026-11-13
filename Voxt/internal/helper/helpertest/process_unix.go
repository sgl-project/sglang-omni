//go:build !windows

package helpertest

import (
	"os"
	"syscall"
)

// processAlive reports whether a process with pid runs (a signal 0 to it).
func processAlive(pid int) bool {
	p, err := os.FindProcess(pid)
	return err == nil && p.Signal(syscall.Signal(0)) == nil
}
