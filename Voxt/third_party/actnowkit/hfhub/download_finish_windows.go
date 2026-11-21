//go:build windows

package hfhub

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"
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
		if err := fs.f.Close(); err != nil {
			return err
		}
		os.Remove(fs.path)
		os.Remove(fs.path + ".parts")
		return fmt.Errorf("%s: %w", fs.Path, ErrChecksum)
	}
	if err := fs.f.Close(); err != nil {
		return err
	}
	if err := os.Rename(fs.path, strings.TrimSuffix(fs.path, ".incomplete")); err != nil {
		return err
	}
	for attempt := 0; ; attempt++ {
		err := os.Remove(fs.path + ".parts")
		if err == nil || os.IsNotExist(err) {
			break
		}
		if attempt == 99 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	return nil
}
