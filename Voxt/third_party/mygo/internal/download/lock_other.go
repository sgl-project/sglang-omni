//go:build !(darwin || dragonfly || freebsd || linux || netbsd || openbsd || windows)

package download

// Lock takes no lock where MyGo has no lock to take: processes there each
// download a file, and Cached keeps the one renamed into place first.
func Lock(path string, wait func()) (unlock func(), err error) { return func() {}, nil }
