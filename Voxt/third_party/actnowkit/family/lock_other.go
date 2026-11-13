//go:build !unix && !windows

package family

import "os"

func flock(*os.File, LockMode, bool) error { return ErrLockUnsupported }

func funlock(*os.File) error { return nil }

// ProcessAlive reports whether a process with this pid exists; on this
// system it cannot tell and says true for any positive pid.
func ProcessAlive(pid int) bool { return pid > 0 }
