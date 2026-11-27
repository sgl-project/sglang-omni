//go:build darwin

package darwin

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/egoist/mygo/internal/platform"
)

// appDirs are the folders apps are installed in, as Spotlight and Launchpad
// find them: apps in a folder of their own one level down too.
func appDirs() []string {
	dirs := []string{
		"/Applications",
		"/System/Applications",
		"/System/Library/CoreServices/Applications",
	}
	if home, err := os.UserHomeDir(); err == nil {
		dirs = append(dirs, filepath.Join(home, "Applications"))
	}
	return dirs
}

// Applications finds the .app bundles in the folders apps are installed in
// and those one level down (Utilities, an app's folder), without looking
// inside bundles, and Finder (which is in CoreServices), each once: an app
// a symbolic link points to, as /Applications/Safari.app does, is listed
// at the link, which FileIcon draws without the badge of links. Each has the
// name Finder shows (localized, without .app) and its bundle identifier,
// or its path without one.
func (b *Backend) Applications() ([]platform.Application, error) {
	var paths []string
	seen := map[string]bool{}
	add := func(p string) {
		real := p
		if r, err := filepath.EvalSymlinks(p); err == nil {
			real = r
		}
		if !seen[real] {
			seen[real] = true
			paths = append(paths, p)
		}
	}
	for _, dir := range appDirs() {
		entries, _ := os.ReadDir(dir)
		for _, e := range entries {
			p := filepath.Join(dir, e.Name())
			if strings.HasSuffix(e.Name(), ".app") {
				add(p)
				continue
			}
			if info, err := os.Stat(p); err != nil || !info.IsDir() {
				continue
			}
			sub, _ := os.ReadDir(p)
			for _, s := range sub {
				if strings.HasSuffix(s.Name(), ".app") {
					add(filepath.Join(p, s.Name()))
				}
			}
		}
	}
	add("/System/Library/CoreServices/Finder.app")

	apps := make([]platform.Application, 0, len(paths))
	withPool(func() {
		fm := send(class("NSFileManager"), "defaultManager")
		for _, p := range paths {
			ns := nsString(p)
			bundle := send(class("NSBundle"), "bundleWithPath:", uintptr(ns))
			if bundle == 0 {
				continue
			}
			a := platform.Application{
				// Finder shows .app where the user has it show all
				// extensions; Launchpad and Spotlight never do.
				Name: strings.TrimSuffix(goString(send(fm, "displayNameAtPath:", uintptr(ns))), ".app"),
				ID:   goString(send(bundle, "bundleIdentifier")),
				Path: p,
			}
			if a.ID == "" {
				a.ID = p // as Steam's shortcuts to games, which have none
			}
			apps = append(apps, a)
		}
	})
	return apps, nil
}

// OpenApplication starts an app, or brings it to the front.
func (s shell) OpenApplication(path string) error { return s.OpenPath(path) }
