//go:build windows

package hfhub

import "github.com/hehehai/actnowkit/models"

// resolveSnapshotLink is the snapshot entry's blob: a symlink's target, or
// the entry itself when it is a hard link or a copy (no blob to follow).
func resolveSnapshotLink(repoDir, link string) (string, error) {
	return models.ResolveBlob(repoDir, link)
}
