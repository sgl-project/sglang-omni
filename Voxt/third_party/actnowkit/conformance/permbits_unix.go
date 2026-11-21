//go:build !windows

package conformance

import "os"

// permBitsOK is the family directory's privacy check (0600/0700).
func permBitsOK(fi os.FileInfo) bool { return fi.Mode().Perm()&0o077 == 0 }
