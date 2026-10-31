//go:build !unix && !windows

package mygo

import "os"

func fileLinks(os.FileInfo) int { return 1 }

func openShared(path string) (*os.File, error) { return os.Open(path) }

func fileWritable(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().Perm()&0o200 != 0
}

func lockFile(*os.File) error { return nil }
