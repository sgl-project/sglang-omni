//go:build windows

package conformance

import "os"

// linkFixtureBlob links the fixture's snapshot entry: symlink, else hard link.
func linkFixtureBlob(fx *Fixture, blob, target, link string) {
	if err := os.Symlink(target, link); err != nil {
		must(fx.T, os.Link(blob, link))
	}
}
