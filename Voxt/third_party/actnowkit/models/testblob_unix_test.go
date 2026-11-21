//go:build !windows

package models

import "path/filepath"

// testBlobPath is the snapshot entry's blob in tests (a link's target, or
// the entry itself).
func testBlobPath(repoDir, p string) (string, error) {
	return filepath.EvalSymlinks(p)
}
