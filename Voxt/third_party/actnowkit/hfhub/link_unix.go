//go:build !windows

package hfhub

import "os"

// linkBlobFile links the snapshot entry to its blob (Windows falls back to
// a hard link, then a copy).
func linkBlobFile(target, link, _ string) error {
	return os.Symlink(target, link)
}
