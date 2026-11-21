//go:build windows

package conformance

import "os"

// permBitsOK: Windows has no POSIX permission bits to check.
func permBitsOK(os.FileInfo) bool { return true }
