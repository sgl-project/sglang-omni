//go:build !windows

package codexauth

import "syscall"

// errAddrInUse is the error a listen reports when the port is taken.
var errAddrInUse = syscall.EADDRINUSE
