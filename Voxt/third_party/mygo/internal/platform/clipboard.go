package platform

import "errors"

// ErrClipboardChanged prevents a read from mixing representations from
// different native clipboard owners while a toolkit pumps its event loop.
var ErrClipboardChanged = errors.New("mygo: clipboard contents changed while reading")

// ErrClipboardPersistence means the desktop has no storage for clipboard
// data after the owner exits (for example, X11 without a clipboard manager).
var ErrClipboardPersistence = errors.New("mygo: desktop cannot persist clipboard contents")
