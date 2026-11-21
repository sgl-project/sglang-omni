//go:build windows

package codexauth

import "syscall"

// errAddrInUse is the error a listen reports when the port is taken:
// WSAEADDRINUSE, which Windows' net package reports as errno 10048
// (syscall.EADDRINUSE is a different, unused errno on Windows).
var errAddrInUse = syscall.Errno(10048)
