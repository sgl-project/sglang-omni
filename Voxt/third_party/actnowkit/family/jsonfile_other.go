//go:build !windows

package family

import "os"

func openForLimitedRead(path string) (*os.File, error) { return os.Open(path) }
