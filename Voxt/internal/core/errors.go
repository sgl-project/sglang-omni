package core

import "errors"

// Sentinel errors. Implementations wrap them (fmt.Errorf("history %s: %w",
// id, ErrNotFound)) so the UI can map them to messages without parsing
// text.
var (
	ErrNotFound    = errors.New("not found")
	ErrExists      = errors.New("already exists")
	ErrInvalid     = errors.New("invalid request")
	ErrBusy        = errors.New("busy")
	ErrCancelled   = errors.New("cancelled")
	ErrUnsupported = errors.New("not supported")
	// ErrHelper: the Swift helper is not running, or exited before it
	// answered (see internal/helper).
	ErrHelper = errors.New("helper unavailable")
)
