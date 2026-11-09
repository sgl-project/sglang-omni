// Package macx is the little macOS glue VoxtNew keeps in Go (purego, no
// cgo), beside the helper's: bringing an app back to the front before a
// paste, when a click on one of VoxtNew's own windows activated
// VoxtNew (plan B of the M0 spike S3; the recording pill avoids it with
// mygo.WindowOptions.NonActivating). macOS 14 activates cooperatively:
// only the active app can hand activation over, so this runs in the app,
// not in the helper. It also reads the system's proxies for the model
// downloads. (The recording's cues are actnowkit's sounds package's,
// settings_sounds.go.) Off macOS it does nothing.
//
// Imports: nothing internal.
package macx

import "time"

// App is a running application.
type App struct {
	PID      int
	BundleID string
	Name     string
}

// Activate brings the app with pid to the front and waits, at most wait,
// until it is; it reports whether it got there and how long it took. When
// VoxtNew is the active app it yields activation to the target first.
// runOnMain runs AppKit calls on the main thread (mygo.RunOnMain); call
// Activate from a goroutine, never from the main thread.
func Activate(pid int, wait time.Duration, runOnMain func(func())) (bool, time.Duration) {
	return activate(pid, wait, runOnMain)
}

// Frontmost is the app in front (NSWorkspace); zero off macOS.
func Frontmost() App { return frontmost() }

// WindowNumber is the window server's number of the NSWindow at handle
// (mygo.Window.NativeHandle), as screencapture -l takes it; 0 off macOS.
// Main thread.
func WindowNumber(handle uintptr) int { return windowNumber(handle) }
