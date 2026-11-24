//go:build windows

package mygo

import (
	"io/fs"
	"os"
	"reflect"
	"unsafe"
)

// statIdentity makes the os package read the file ID its fileStat holds,
// which it otherwise defers to the first os.SameFile comparison: that one
// opens the path again and, after a rename and replace, reads the
// replacement's ID, taking two directories for one.
//
// os/types_windows.go orders the fields so the ID follows the path, and
// clears the path once the ID is read, so clearing it here leaves the ID
// read for every later comparison. A FileInfo of another kind, from
// os.Root or File.Stat, holds no fileStat: it IDs the file through a
// handle, read when it was taken, and is left as it is.
func statIdentity(info fs.FileInfo) {
	v := reflect.ValueOf(info)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return
	}
	path := v.FieldByName("path")
	if !path.IsValid() || path.Kind() != reflect.String || path.Len() == 0 {
		return
	}
	// SameFile with itself reads the ID.
	_ = os.SameFile(info, info)
	// The path of a loaded fileStat is empty; clearing it, the ID read
	// stays.
	field := (*string)(unsafe.Pointer(v.FieldByName("path").UnsafeAddr()))
	*field = ""
}
