//go:build !windows

package hfhub

import "path/filepath"

// resolveSnapshotLink is the snapshot entry's blob, by the link.
func resolveSnapshotLink(_, link string) (string, error) {
	return filepath.EvalSymlinks(link)
}
