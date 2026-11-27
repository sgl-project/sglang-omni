package mygo

import "io/fs"

// StatIdentity returns info with its identity fixed as it is now, for
// comparing with os.SameFile later.
//
// A FileInfo from Stat, Lstat or a directory entry carries the file's
// identity, which os.SameFile compares. On Windows, Stat and Lstat defer
// reading it to the first comparison, which opens the path again: after a
// rename and replace, the stat of the directory before compares equal to
// the one after, and a file replaced as editors save is taken for the one
// before. Where identities of stats taken apart are compared, as the file
// watches do, pass them here first.
func StatIdentity(info fs.FileInfo) fs.FileInfo {
	if info != nil {
		statIdentity(info)
	}
	return info
}
