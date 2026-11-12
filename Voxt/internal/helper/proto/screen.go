package proto

// ---- screen.window ----

// ScreenWindowParams asks for a picture of an app's front window (the
// window on top of that app's, on screen, at the normal level), for a
// language model that takes images. It needs Screen Recording: without
// it the request fails with permission_denied
// (PermissionScreenRecording).
type ScreenWindowParams struct {
	PID int `json:"pid"`
	// MaxSide bounds the picture's longer side in pixels
	// (ScreenMaxSide when 0, at most ScreenMaxSide).
	MaxSide int `json:"maxSide,omitempty"`
}

// ScreenWindowResult is the picture: JPEG is absent when the app has no
// window on screen.
type ScreenWindowResult struct {
	JPEG []byte `json:"jpeg,omitempty"` // base64 on the wire
}

// ScreenMaxSide is the longest side screen.window draws: enough to read a
// window's text, small enough for any model's image limit and the wire
// (MaxLine).
const ScreenMaxSide = 1600
