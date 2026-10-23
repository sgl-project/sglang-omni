//go:build darwin

package darwin

import (
	"errors"
	"os"
	"unsafe"
)

// ReplaceFile replaces target with -[NSFileManager
// replaceItemAtURL:withItemAtURL:…], NSDocument's safe save: the target
// keeps its permissions, creation date, extended attributes (Finder tags,
// Finder info) and ACLs, and its identity, so that aliases still find it.
func (*Backend) ReplaceFile(target, replacement string) error {
	if _, err := os.Lstat(target); os.IsNotExist(err) {
		return os.Rename(replacement, target)
	}
	load()
	var err error
	withPool(func() {
		fm := send(class("NSFileManager"), "defaultManager")
		var nsErr id
		ok := sendBool(fm, "replaceItemAtURL:withItemAtURL:backupItemName:options:resultingItemURL:error:",
			uintptr(fileURL(target)), uintptr(fileURL(replacement)), 0, 0, 0, uintptr(unsafe.Pointer(&nsErr)))
		if !ok {
			msg := "cannot replace the file"
			if nsErr != 0 {
				msg = goString(send(nsErr, "localizedDescription"))
			}
			err = &os.LinkError{Op: "replace", Old: replacement, New: target, Err: errors.New(msg)}
		}
	})
	return err
}
