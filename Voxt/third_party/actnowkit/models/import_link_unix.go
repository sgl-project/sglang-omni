//go:build !windows

package models

import (
	"os"
	"path/filepath"
	"strings"
)

// linkFiles makes <dir>/<rel> → ../../blobs/<blob> (relative, like
// huggingface_hub); fresh: dir is new.
func linkFiles(dir string, files []srcFile, fresh bool) error {
	for _, f := range files {
		link := filepath.Join(dir, filepath.FromSlash(f.rel))
		if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
			return err
		}
		target := strings.Repeat("../", strings.Count(f.rel, "/")+2) + "blobs/" + f.blob
		if !fresh {
			if cur, err := os.Readlink(link); err == nil && cur == target {
				continue
			}
			_ = os.Remove(link)
		}
		if err := os.Symlink(target, link); err != nil {
			return err
		}
	}
	return nil
}
