//go:build !windows

package hfhub

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// finishFile checks a complete .incomplete blob, moves it in place and
// releases its lock.
func (c *Client) finishFile(ctx context.Context, fs *fileState) error {
	defer fs.close()
	if err := fs.f.Sync(); err != nil {
		return err
	}
	ok, err := checkBlob(ctx, fs.path, fs.remoteFile)
	if err != nil {
		return err
	}
	if !ok {
		os.Remove(fs.path)
		os.Remove(fs.path + ".parts")
		return fmt.Errorf("%s: %w", fs.Path, ErrChecksum)
	}
	if err := os.Rename(fs.path, strings.TrimSuffix(fs.path, ".incomplete")); err != nil {
		return err
	}
	os.Remove(fs.path + ".parts")
	return nil
}
