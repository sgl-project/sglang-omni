package mygo

import (
	"errors"
	"os"
	"runtime"
	"syscall"
	"time"
)

// Windows refuses to replace or delete a file that another program has
// open without sharing its deletion, as Go opens files, and antivirus
// software and indexers do while they scan new files; and it refuses to
// open a file while it is being replaced. The app's own files (window
// states, recovery copies, recent documents) are replaced in place, so
// their renames and reads retry for a few seconds while Windows reports
// them in use, rather than lose what they hold.

// rename renames a file of the app, retrying while Windows reports it in
// use.
func rename(from, to string) error {
	return retryInUse(func() error { return os.Rename(from, to) })
}

// readAppFile reads a file of the app, retrying while Windows reports it
// in use.
func readAppFile(path string) (data []byte, err error) {
	err = retryInUse(func() error {
		data, err = os.ReadFile(path)
		return err
	})
	return data, err
}

func retryInUse(f func() error) error {
	for i := 1; ; i++ {
		err := f()
		if err == nil || i == 10 || runtime.GOOS != "windows" ||
			// ERROR_SHARING_VIOLATION, or ERROR_ACCESS_DENIED for a file
			// being deleted or replaced, or a directory holding an open file.
			!errors.Is(err, syscall.Errno(32)) && !errors.Is(err, syscall.Errno(5)) {
			return err
		}
		time.Sleep(time.Duration(i) * 100 * time.Millisecond)
	}
}
