package mygo

import "os"

// fileWatchIdentity reads no identity on Plan 9, where watches are
// unsupported: pairing compares with os.SameFile.
func fileWatchIdentity(os.FileInfo) (fileWatchID, bool) { return fileWatchID{}, false }
