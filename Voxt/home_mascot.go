package main

// Home's mascot: the brand microphone and the logo's two stars, by the
// page's title, in the day's outfit (mascotOutfitOn). A click plays the next of three
// short moves: a sway with a blink, a double blink, and the stars
// twinkling with a blink. When it puts on an outfit, it sways or blinks
// twice, at random.

import (
	"math"
	"math/rand/v2"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const (
	mascotSize   = 19 // the microphone's square, DIPs
	mascotMoves  = 3
	mascotLength = 650 * time.Millisecond
)

// mascotLocal is what the mascot keeps from frame to frame: how many
// times it was clicked, which picks the move and plays it; the outfit it
// last put on while Home shows, how many times it put one on, which plays
// dressMove.
type mascotLocal struct {
	pokes     uint64
	worn      string
	dresses   uint64
	dressMove int
}

// mascotDressMoves are the moves it may play as it puts on an outfit:
// the sway and the double blink.
var mascotDressMoves = [...]int{0, 1}

// mascotPose is the mascot at one moment of a move.
type mascotPose struct {
	tilt          float32 // degrees, around the microphone's foot
	blink         bool
	orange, peach float32 // the stars' scale
}

// mascotPoseAt is move's pose at progress p (0 to 1): at rest at both
// ends.
func mascotPoseAt(move int, p float32) mascotPose {
	pose := mascotPose{orange: 1, peach: 1}
	in := func(a, b float32) bool { return p >= a && p < b }
	bump := func(a, b float32) float32 {
		if p < a || p > b {
			return 0
		}
		return float32(math.Sin(math.Pi * float64((p-a)/(b-a))))
	}
	switch move {
	case 0: // a sway that dies down, blinking on the way
		pose.tilt = -7 * float32(math.Sin(3*math.Pi*float64(p))) * (1 - p)
		pose.blink = in(.25, .4)
	case 1: // a double blink
		pose.blink = in(.05, .2) || in(.35, .5)
	case 2: // the stars twinkle in turn
		pose.orange = 1 + .6*bump(0, .55)
		pose.peach = 1 + .7*bump(.25, .8)
		pose.blink = in(.4, .55)
	}
	return pose
}

// mascotPieces is what an outfit puts on the microphone.
func mascotPieces(outfit string) []icons.MascotPiece {
	switch outfit {
	case outfitChristmas:
		return icons.MascotChristmas
	case outfitNewYear:
		return icons.MascotNewYear
	case outfitSpring:
		return icons.MascotSpring
	}
	return nil
}

func homeMascot(c *ui.Context, outfit string) {
	t := c.Theme()
	k := tok(t)
	side := px(t, mascotSize)
	box := icons.MascotBox
	u := side / box[2] // one of the microphone's units, in DIPs
	// The stars stand right of the microphone's head, as in the logo.
	width := side + 60*u
	button := ui.ButtonBase(c.Key("home-mascot")).Size(width, side).Shrink(0).Label(i18n.T("Voxt mascot")).
		Margin(0, 0, 0, -px(t, 4)) // closer to the title than the header's gap
	local := ui.Local(button, "mascot", func() mascotLocal { return mascotLocal{} })
	if button.Clicked() {
		local.pokes++
	}
	if outfit != local.worn {
		local.worn = outfit
		if outfit != "" {
			local.dresses++
			local.dressMove = mascotDressMoves[rand.IntN(len(mascotDressMoves))]
		}
	}
	tl := button.Timeline("move", ui.TimelineOptions{Run: local.pokes, Paused: local.pokes == 0})
	p := tl.Tween(ui.Tween{From: 0, To: 1, Duration: mascotLength})
	pose := mascotPoseAt(int((local.pokes+mascotMoves-1)%mascotMoves), p)
	dress := button.Timeline("dress", ui.TimelineOptions{Run: local.dresses, Paused: local.dresses == 0, Delay: 300 * time.Millisecond})
	if dp := dress.Tween(ui.Tween{From: 0, To: 1, Duration: mascotLength}); local.dresses > 0 && dp < 1 {
		pose = mascotPoseAt(local.dressMove, dp)
	}
	microphone := icons.MascotMicrophone
	if pose.blink {
		microphone = icons.MascotMicrophoneBlink
	}
	at := func(x, y float32) (float32, float32) { return (x - box[0]) * u, (y - box[1]) * u }
	button.Children(func() {
		ui.Box(c).Absolute().Left(0).Top(0).Size(side, side).TransformOrigin(.5, .97).Rotate(pose.tilt).Children(func() {
			ui.Icon(c, microphone).Size(side, side).TextColor(k.Text2) // as the sidebar's icons
			for _, piece := range mascotPieces(outfit) {
				x, y := at(piece.Box[0], piece.Box[1])
				ui.Image(c, piece.SVG).Absolute().Left(x).Top(y).Size(piece.Box[2]*u, piece.Box[3]*u)
			}
		})
		x, y := at(380, -10)
		ui.Image(c, openingOrange).Absolute().Left(x).Top(y).Size(130*u, 130*u).Scale(pose.orange, pose.orange)
		x, y = at(470, 120)
		ui.Image(c, openingPeach).Absolute().Left(x).Top(y).Size(80*u, 80*u).Scale(pose.peach, pose.peach)
	})
}
