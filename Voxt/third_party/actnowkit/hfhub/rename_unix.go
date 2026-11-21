//go:build !windows

package hfhub

import "os"

// renameFile renames a file (Windows retries while another process holds it).
func renameFile(oldPath, newPath string) error {
	return os.Rename(oldPath, newPath)
}
