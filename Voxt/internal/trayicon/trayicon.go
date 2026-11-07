// Package trayicon holds the menu bar icon: Voxt's microphone
// (voxt.png, 512×512, unchanged from Voxt's voxt.imageset; NOTICE), as
// template PNGs of 36×36 pixels. MyGo's macOS tray sizes any icon to 18
// points tall, so on a Retina screen each pixel is one device pixel (Voxt
// shows the same image at 18×18 points); a 1× screen gets it scaled down
// by AppKit. The PNGs are made offline, with a Catmull-Rom filter:
//
//	go generate ./internal/trayicon
//
// Recording shows the same microphone with a solid dot at its top right,
// in a gap cut around it so that it reads apart from the capsule; macOS
// tints both like the rest of the menu bar.
package trayicon

//go:generate go run ./gen

import _ "embed"

// Size is the icons' width and height in pixels (18 pt at 2×).
const Size = 36

var (
	// Idle is the icon at rest.
	//go:embed menubar.png
	Idle []byte
	// Recording is the icon while a session listens or works.
	//go:embed menubar-recording.png
	Recording []byte
)

// Icon is the icon for the state: recording or not.
func Icon(recording bool) []byte {
	if recording {
		return Recording
	}
	return Idle
}
