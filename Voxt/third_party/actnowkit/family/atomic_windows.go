//go:build windows

package family

import (
	"os"
	"time"
)

// renameFile retries while a reader or an antivirus scan holds the file.
func renameFile(oldPath, newPath string) error {
	for attempt := 0; ; attempt++ {
		err := os.Rename(oldPath, newPath)
		if err == nil || attempt == 99 {
			return err
		}
		time.Sleep(5 * time.Millisecond)
	}
}
