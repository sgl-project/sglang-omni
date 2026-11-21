// Package nsis pins the NSIS that mygo build downloads on Windows to make
// the installer, and installs it into the user's cache directory, where
// makensis runs as it does from an installation. CI downloads it with
// internal/prefetch before the tests run.
package nsis

import (
	"archive/zip"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/egoist/mygo/internal/download"
)

// Release is the NSIS that mygo build downloads: the official zip,
// checked against its SHA-256. The URLs are copies of that zip, tried in
// order: the asset of the nsis-<version> release of MyGo's repository
// first, since SourceForge, where NSIS publishes it, has been down for
// hours at a time, its mirrors redirecting to it.
var Release = struct {
	Version, SHA256 string
	URLs            []string
}{
	Version: "3.13",
	SHA256:  "ba63dffc4410ee89193e1cb5a41989991bd77c61068da17e3156d136b7b0b3d8",
	URLs: []string{
		"https://github.com/egoist/mygo/releases/download/nsis-3.13/nsis-3.13.zip",
		"https://downloads.sourceforge.net/project/nsis/NSIS%203/3.13/nsis-3.13.zip",
	},
}

// Dir is where mygo build keeps the NSIS it downloads, or "" without a
// cache directory.
func Dir() string {
	cache, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(cache, "mygo", "nsis-"+Release.Version)
}

// Executable returns makensis in the NSIS directory dir, or "".
func Executable(dir string) string {
	for _, p := range []string{filepath.Join(dir, "Bin", "makensis.exe"), filepath.Join(dir, "makensis.exe")} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// Download downloads Release into dir, unless it is there already, and
// returns its makensis; file makes the download of each URL. The archive
// is unpacked next to dir and renamed into place, so dir is complete
// whenever it exists, and processes that need it at the same time
// download it once: the others wait, then find it.
func Download(ctx context.Context, dir string, file func(url string) download.File) (string, error) {
	if p := Executable(dir); p != "" {
		return p, nil
	}
	parent := filepath.Dir(dir)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return "", err
	}
	unlock, err := download.Lock(dir+".lock", nil)
	if err != nil {
		return "", err
	}
	defer unlock()
	if p := Executable(dir); p != "" {
		return p, nil
	}
	work, err := os.MkdirTemp(parent, ".nsis-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(work)
	archive := filepath.Join(work, "nsis.zip")
	var errs []error
	for _, url := range Release.URLs {
		f := file(url)
		f.SHA256 = Release.SHA256
		err := download.Get(ctx, f, archive)
		if err == nil {
			errs = nil
			break
		}
		errs = append(errs, err)
	}
	if len(errs) > 0 {
		return "", errors.Join(errs...)
	}
	unpacked := filepath.Join(work, "nsis")
	if err := unzipTop(archive, unpacked); err != nil {
		return "", fmt.Errorf("unpacking NSIS: %w", err)
	}
	if Executable(unpacked) == "" {
		return "", errors.New("unpacking NSIS: the archive holds no makensis.exe")
	}
	if err := os.Rename(unpacked, dir); err != nil {
		// Anything there is broken, or it would have been found above.
		if err := os.RemoveAll(dir); err != nil {
			return "", err
		}
		if err := os.Rename(unpacked, dir); err != nil {
			return "", err
		}
	}
	return Executable(dir), nil
}

// unzipTop unpacks the zip archive at path, whose entries are all in one
// top directory, into dir without that directory.
func unzipTop(path, dir string) error {
	r, err := zip.OpenReader(path)
	if err != nil {
		return err
	}
	defer r.Close()
	top := ""
	for _, f := range r.File {
		first, rest, ok := strings.Cut(f.Name, "/")
		if top == "" {
			top = first
		}
		if !ok || first != top {
			return fmt.Errorf("%q is not in the top directory %q", f.Name, top)
		}
		if rest == "" {
			continue // the top directory
		}
		name := strings.TrimSuffix(rest, "/")
		if !fs.ValidPath(name) || !filepath.IsLocal(filepath.FromSlash(name)) || strings.Contains(name, `\`) {
			return fmt.Errorf("invalid path %q", f.Name)
		}
		target := filepath.Join(dir, filepath.FromSlash(name))
		switch mode := f.Mode(); {
		case mode.IsDir():
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case mode.IsRegular():
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			if err := unzipFile(f, target); err != nil {
				return err
			}
		default:
			return fmt.Errorf("%q is neither a file nor a directory", f.Name)
		}
	}
	return nil
}

func unzipFile(f *zip.File, target string) error {
	src, err := f.Open()
	if err != nil {
		return err
	}
	defer src.Close()
	dst, err := os.Create(target)
	if err != nil {
		return err
	}
	// The zip reader checks the CRC-32 of the file once it is read.
	_, err = io.Copy(dst, src)
	if cerr := dst.Close(); err == nil {
		err = cerr
	}
	return err
}
