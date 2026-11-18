package platform

import (
	"errors"
	"io/fs"
)

// FileWatchTarget is a file or directory the core enrolls in a FileWatch.
// IDs are never reused within a FileWatch.
type FileWatchTarget struct {
	ID   uint64
	Path string
	// Directory says the target is a directory. A backend that watches a
	// file only through its directory, which the core enrolls before it
	// lists it, returns ErrUnsupported for a target that isn't one, and
	// the core enrolls no more files.
	Directory bool
	// Info is what the core's scan saw at Path. Backends open the target
	// and check it is still the same file, so that a replacement racing
	// the enrollment is not watched in its place.
	Info fs.FileInfo
	// Subtree asks to watch everything beneath the directory with this one
	// target, and to name what changes deeper down by its slash-separated
	// path relative to it. A backend that can't returns ErrUnsupported,
	// and the core enrolls every directory instead.
	Subtree bool
}

// FileWatchKind says what changed. Backends report what the system tells
// them; the core rescans what a notice names, so a kind is only a hint.
type FileWatchKind uint8

const (
	// FileWatchDirty means something about the target or its child changed.
	FileWatchDirty FileWatchKind = iota
	FileWatchWrite
	FileWatchCreate
	FileWatchRemove
	FileWatchMoveFrom
	FileWatchMoveTo
	// FileWatchLost means the system stopped watching the target, for
	// example because it was deleted; the core enrolls it again if it
	// still exists.
	FileWatchLost
)

// FileWatchNotice reports a change to a target (Name empty) or to one of
// the immediate children of a directory target (Name its name; a path for
// a Subtree target). A notice with Err set ends the watch, except
// ErrFileWatchOverflow, after which the core rescans everything.
type FileWatchNotice struct {
	Target uint64
	Kind   FileWatchKind
	Name   string
	Err    error
}

// ErrFileWatchOverflow is the Err of a notice when the system dropped
// changes, for example because its queue overflowed.
var ErrFileWatchOverflow = errors.New("file watch queue overflow")

// FileWatch watches the targets the core adds. Its methods may be called
// from any goroutine, one at a time. Close stops the watch and waits until
// every handle is released and notify will not be called again; it must
// not be called from notify.
type FileWatch interface {
	Add(target FileWatchTarget) error
	Remove(id uint64)
	Close()
}
