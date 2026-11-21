//go:build !darwin

package models

func platformSetBackupExcluded(string, bool) error { return ErrBackupUnsupported }

func platformIsBackupExcluded(string) (bool, error) { return false, ErrBackupUnsupported }
