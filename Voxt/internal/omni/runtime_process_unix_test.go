//go:build !windows

package omni

import "syscall"

func gone(pid int) bool { return syscall.Kill(pid, 0) != nil }
