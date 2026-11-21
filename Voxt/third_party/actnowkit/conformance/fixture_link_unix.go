//go:build !windows

package conformance

import "os"

// linkFixtureBlob symlinks the fixture's snapshot entry (Windows: hard link).
func linkFixtureBlob(fx *Fixture, _, target, link string) {
	must(fx.T, os.Symlink(target, link))
}
