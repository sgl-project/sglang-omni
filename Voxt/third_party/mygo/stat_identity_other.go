//go:build !windows

package mygo

import "io/fs"

// statIdentity has nothing to do: elsewhere a stat carries the file's
// identity from the moment it is taken.
func statIdentity(fs.FileInfo) {}
