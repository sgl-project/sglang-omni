package main

// The hat studio (share_studio.go): a board the shape of the hat's box on
// the mascot, where the user draws a hat over Voxt's head, or starts from
// a hat in the wardrobe or a picture. Strokes are kept in the mascot's
// units, the ones its hats are drawn in, so a hat drawn here sits on the
// card's Voxt where it was drawn.

import "math"

// shareHatBox is the hat's box in the mascot's units (its microphone's
// viewBox is -100 -16 600 600): left, top, width and height. The board
// shows this box; a drawn hat is saved as a picture of it.
var shareHatBox = [4]float32{40, -210, 330, 270}

// shareBrushes are the brush sizes, in the mascot's units: thin, medium,
// thick.
var shareBrushes = [...]float32{8, 16, 28}

// shareWardrobeMax is how many drawn hats the wardrobe keeps: the oldest
// go when a new one would pass it (each is a small PNG).
const shareWardrobeMax = 60

// shareStroke is one stroke of the pen: its color, its width and its
// points, in the mascot's units.
type shareStroke struct {
	Color  string       `json:"color"`
	Width  float32      `json:"width"`
	Points [][2]float32 `json:"points"`
}

type shareStudioViewState struct {
	// Base is under the strokes: "outfit:<key>" (a built-in hat) or
	// "hat:<name>" (a drawn one); "" for none.
	Base    string        `json:"base,omitempty"`
	Strokes []shareStroke `json:"strokes,omitempty"`
	Color   string        `json:"color"`
	Brush   int           `json:"brush"`
	Erasing bool          `json:"erasing,omitempty"`
	CanUndo bool          `json:"can_undo,omitempty"`
	// Peek shows a small Voxt wearing the board's hat; Closet the
	// wardrobe in place of the tools.
	Peek   bool `json:"peek,omitempty"`
	Closet bool `json:"closet,omitempty"`
	// Wardrobe is what the closet offers: the built-in hats first
	// ("outfit:<key>"), then the drawn ones ("hat:<name>"), newest first.
	Wardrobe []string `json:"wardrobe,omitempty"`
}

// shareErase removes what of strokes lies within r of (x, y): a stroke
// the eraser crosses splits into the parts on either side of it.
func shareErase(strokes []shareStroke, x, y, r float32) []shareStroke {
	out := strokes[:0:0]
	for _, s := range strokes {
		reach := r + s.Width/2
		var run [][2]float32
		flush := func() {
			if len(run) > 0 {
				out = append(out, shareStroke{Color: s.Color, Width: s.Width, Points: run})
				run = nil
			}
		}
		for _, p := range s.Points {
			if math.Hypot(float64(p[0]-x), float64(p[1]-y)) <= float64(reach) {
				flush()
				continue
			}
			run = append(run, p)
		}
		flush()
	}
	return out
}

// shareStrokesEqual reports whether a and b are the same drawing (an
// eraser that touched nothing leaves no undo step).
func shareStrokesEqual(a, b []shareStroke) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Color != b[i].Color || a[i].Width != b[i].Width || len(a[i].Points) != len(b[i].Points) {
			return false
		}
		for j := range a[i].Points {
			if a[i].Points[j] != b[i].Points[j] {
				return false
			}
		}
	}
	return true
}
