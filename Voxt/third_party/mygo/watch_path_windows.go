package mygo

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"syscall"
)

// OpenRoot uses an ordinary Windows handle that prevents deletion and rename.
// Root.OpenRoot shares deletion, so retain that anchored handle instead and
// release the temporary parent before returning. A drive's root has no
// parent, and can't be renamed anyway.
func fileWatchOpenRoot(path string) (*os.Root, error) {
	if filepath.Dir(path) == path {
		return os.OpenRoot(path)
	}
	parent, err := os.OpenRoot(filepath.Dir(path))
	if err != nil {
		return nil, err
	}
	defer parent.Close()
	return parent.OpenRoot(filepath.Base(path))
}

// fileWatchInaccessible reports whether err says a file can't be looked
// at: without permission, or held by the system with a sharing mode that
// leaves out even reading its attributes, as pagefile.sys and
// hiberfil.sys.
func fileWatchInaccessible(err error) bool {
	const errorSharingViolation, errorLockViolation = syscall.Errno(32), syscall.Errno(33)
	return errors.Is(err, fs.ErrPermission) || errors.Is(err, errorSharingViolation) || errors.Is(err, errorLockViolation)
}

// fileWatchEntryInfo returns what entry is, read from a directory opened in
// root, by its name in root. What reading a directory says of an entry
// lacks its identity, which os.SameFile would load by its path.
func fileWatchEntryInfo(root *os.Root, entry fs.DirEntry) (os.FileInfo, error) {
	return root.Lstat(entry.Name())
}
