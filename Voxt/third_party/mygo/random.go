//go:build !js

package mygo

import "crypto/rand"

// randomText returns 26 random base32 characters (128 bits), which nobody
// can guess: the secret of a window's bridge, the session of a drag, the
// ID of a notification.
func randomText() string { return rand.Text() }
