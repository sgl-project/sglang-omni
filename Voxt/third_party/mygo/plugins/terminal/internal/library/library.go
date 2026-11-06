// Package library finds libghostty-vt for the terminal plugin: next to
// the app, in the user's cache, or downloaded there, as mygo-plugin.json
// names it.
package library

import (
	"context"
	"encoding/json/v2"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/egoist/mygo"
	fetch "github.com/egoist/mygo/internal/download"
)

// nativeLibraries is the format of mygo-plugin.json.
type nativeLibraries struct {
	Libraries []struct {
		Name  string                `json:"name"`
		Files map[string]nativeFile `json:"files"`
	} `json:"libraries"`
}

type nativeFile struct {
	Name   string `json:"name"`
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
}

// library returns the build of libghostty-vt for this platform that the
// manifest names.
func library(manifest []byte) (nativeFile, error) {
	var n nativeLibraries
	if err := json.Unmarshal(manifest, &n); err != nil {
		return nativeFile{}, err
	}
	target := runtime.GOOS + "-" + runtime.GOARCH
	for _, l := range n.Libraries {
		if f, ok := l.Files[target]; ok && l.Name == "libghostty-vt" {
			return f, nil
		}
	}
	return nativeFile{}, fmt.Errorf("terminal: libghostty-vt has no build for %s", target)
}

// Find returns the libghostty-vt that manifest names for this platform,
// the first of:
//
//   - the file $MYGO_GHOSTTY_VT names;
//   - the one among the app's resources (mygo.PathResources), where `mygo
//     build` and `mygo dev` put it, signed with the app on macOS;
//   - the one next to the executable;
//   - the one in the user's cache, where the CLI downloads it
//     (<cache>/mygo/natives/<sha256>/), which programs that are not
//     packaged apps, as under `go run` and `go test`, download there when
//     it is missing, checking its SHA-256.
//
// Packaged apps never download it: build them with the CLI.
func Find(manifest []byte) (string, error) {
	if p := os.Getenv("MYGO_GHOSTTY_VT"); p != "" {
		return p, nil
	}
	f, err := library(manifest)
	if err != nil {
		return "", err
	}
	var dirs []string
	if dir, err := mygo.App.Path(mygo.PathResources); err == nil {
		dirs = append(dirs, dir)
	}
	if exe, err := os.Executable(); err == nil {
		dirs = append(dirs, filepath.Dir(exe))
	}
	cached, err := cachePath(f)
	if err == nil {
		dirs = append(dirs, filepath.Dir(cached))
	}
	for _, dir := range dirs {
		p := filepath.Join(dir, f.Name)
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	if mygo.App.IsPackaged() {
		return "", fmt.Errorf("terminal: %s is missing from the app: build it with mygo build", f.Name)
	}
	if cached == "" {
		return "", err
	}
	if err := download(f, cached); err != nil {
		return "", err
	}
	return cached, nil
}

// cachePath returns where the CLI and this package keep a downloaded
// library.
func cachePath(f nativeFile) (string, error) {
	dir, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "mygo", "natives", f.SHA256, f.Name), nil
}

// download downloads f to path, checking its SHA-256, once among the
// processes that need it at the same time.
func download(f nativeFile, path string) error {
	if f.URL == "" || f.SHA256 == "" {
		return fmt.Errorf("terminal: no download of %s for %s/%s", f.Name, runtime.GOOS, runtime.GOARCH)
	}
	log.Printf("terminal: downloading %s into %s", f.URL, filepath.Dir(path))
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if err := fetch.Cached(ctx, fetch.File{URL: f.URL, SHA256: f.SHA256}, path); err != nil {
		return fmt.Errorf("terminal: downloading %s: %w", f.Name, err)
	}
	return nil
}
