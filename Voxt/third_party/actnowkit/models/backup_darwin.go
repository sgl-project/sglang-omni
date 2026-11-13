//go:build darwin

package models

import "github.com/hehehai/actnowkit/internal/darwin"

func platformSetBackupExcluded(path string, excluded bool) error {
	if excluded {
		return darwin.SetXattr(path, BackupExcludeXattr, backupExcludeValue)
	}
	return darwin.RemoveXattr(path, BackupExcludeXattr)
}

func platformIsBackupExcluded(path string) (bool, error) {
	_, ok, err := darwin.GetXattr(path, BackupExcludeXattr)
	return ok, err
}

// backupExcludeValue is the attribute's value: a binary property list of
// the string "com.apple.backupd", as tmutil writes it.
var backupExcludeValue = []byte("bplist00_\x10\x11com.apple.backupd\x08\x00\x00\x00\x00\x00\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1c")
