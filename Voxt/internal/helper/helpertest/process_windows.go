//go:build windows

package helpertest

import "os"

// processAlive reports whether a process with pid runs. On Windows,
// FindProcess opens the process and fails when it does not exist (there is
// no signal 0, and os.Kill would terminate it).
func processAlive(pid int) bool {
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	_ = p.Release()
	return true
}
