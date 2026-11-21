//go:build windows

package models

// testBlobPath is the snapshot entry's blob in tests.
func testBlobPath(repoDir, p string) (string, error) {
	return ResolveBlob(repoDir, p)
}
