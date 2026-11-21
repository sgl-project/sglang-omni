//go:build windows

package hfhub

import (
	"io"
	"os"
)

// linkBlobFile links the snapshot entry to its blob: a symlink when the
// system allows it, else a hard link, else a copy.
func linkBlobFile(target, link, blob string) error {
	if err := os.Symlink(target, link); err == nil {
		return nil
	}
	if err := os.Link(blob, link); err == nil {
		return nil
	}
	in, err := os.Open(blob)
	if err != nil {
		return err
	}
	defer in.Close()
	info, err := in.Stat()
	if err != nil {
		return err
	}
	out, err := os.OpenFile(link, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode().Perm())
	if err != nil {
		return err
	}
	_, err = io.Copy(out, in)
	if closeErr := out.Close(); err == nil {
		err = closeErr
	}
	return err
}
