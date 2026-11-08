// Package game has what games need beyond what apps do, opt-in, out of
// MyGo's core ui: a game loop that updates a world in fixed steps with
// the player's input and draws it every frame (View), and a camera that
// looks at a 2D world (Camera), shown pixel-perfect at any size through
// a ui.Canvas (IntegerFit). Its package gamepad reads controllers.
//
//	type farm struct{ x, y float32 }
//
//	func (f *farm) Update(in *game.Input) {
//		if in.Down(ui.KeyD) {
//			f.x += 2
//		}
//	}
//
//	func (f *farm) Draw(p *ui.Painter, r ui.Rect, alpha float64) {
//		game.Camera{X: f.x, Y: f.y, Zoom: 3}.Draw(p, r, func() { drawWorld(p) })
//	}
//
//	ui.View(func(c *ui.Context) { game.View(c, f).Fill() })
package game

import (
	"math"
	"time"

	"github.com/egoist/mygo/ui"
)

// A Game is a game View runs: it updates the game's world in fixed
// steps, whatever the display's rate, and draws it every frame, as
// Ebitengine's Update and Draw.
type Game interface {
	// Update advances the game one step, of 1/Rate seconds unless
	// the game has a Rate method, with the input since the step before.
	Update(in *Input)
	// Draw draws the game into r. alpha, from 0 to 1, is how far the
	// frame's time is into the next step: a game may draw what moves
	// that far between its last two places, for motion smoother than its
	// steps.
	Draw(p *ui.Painter, r ui.Rect, alpha float64)
}

// Rate is how many steps a second a View runs a game, unless the
// game has a method Rate() int giving its own.
const Rate = 60

// maxCatchUp is the most time steps catch up on, after a pause such as a
// window out of sight: MonoGame's MaxElapsedTime.
const maxCatchUp = 250 * time.Millisecond

// Input is what a game's player did, as a step sees it: the keys and
// pointer buttons held, and those pressed and released since the step
// before, even if pressed and released between two steps. Keys are by
// position, as PhysicalKey: KeyW is the key in W's place whatever the
// layout, as games want for WASD.
type Input struct {
	down, pressed, released map[ui.Key]bool

	x, y                         float32
	inside                       bool
	buttons, bPressed, bReleased [3]bool
	wheelX, wheelY               float32
	w, h                         float32
	focused                      bool
}

func newInput() *Input {
	return &Input{down: map[ui.Key]bool{}, pressed: map[ui.Key]bool{}, released: map[ui.Key]bool{}}
}

// Down reports whether k is held.
func (in *Input) Down(k ui.Key) bool { return in.down[k] }

// Pressed reports whether k was pressed since the step before.
func (in *Input) Pressed(k ui.Key) bool { return in.pressed[k] }

// Released reports whether k was released since the step before.
func (in *Input) Released(k ui.Key) bool { return in.released[k] }

// Pointer returns where the pointer is in the view, from its top-left
// corner, and whether it is over the view.
func (in *Input) Pointer() (x, y float32, over bool) { return in.x, in.y, in.inside }

// PointerDown reports whether pointer button b (0 the primary, 1 the
// secondary, 2 the middle) is held.
func (in *Input) PointerDown(b int) bool { return b >= 0 && b < 3 && in.buttons[b] }

// PointerPressed reports whether button b was pressed since the step
// before.
func (in *Input) PointerPressed(b int) bool { return b >= 0 && b < 3 && in.bPressed[b] }

// PointerReleased reports whether button b was released since the step
// before.
func (in *Input) PointerReleased(b int) bool { return b >= 0 && b < 3 && in.bReleased[b] }

// Wheel returns how far the wheel or touchpad scrolled since the step
// before, in DIPs, down and right positive.
func (in *Input) Wheel() (dx, dy float32) { return in.wheelX, in.wheelY }

// Size returns the view's size.
func (in *Input) Size() (w, h float32) { return in.w, in.h }

// Focused reports whether the view has the keyboard, which it loses with
// its window or to another element: a game pauses without it, as the
// player is elsewhere.
func (in *Input) Focused() bool { return in.focused }

// step forgets what happened before this step.
func (in *Input) step() {
	clear(in.pressed)
	clear(in.released)
	in.bPressed, in.bReleased = [3]bool{}, [3]bool{}
	in.wheelX, in.wheelY = 0, 0
}

// releaseAll lets go of everything held, as when the window loses the
// keyboard, whose releases never come.
func (in *Input) releaseAll() {
	for k := range in.down {
		in.released[k] = true
	}
	clear(in.down)
	for b, held := range in.buttons {
		if held {
			in.bReleased[b] = true
		}
	}
	in.buttons = [3]bool{}
}

// loop is what a View keeps from frame to frame.
type loop struct {
	game      Game
	in        *Input
	last      time.Time
	lag       time.Duration
	wantFocus bool
}

func (l *loop) rate() int {
	if r, ok := l.game.(interface{ Rate() int }); ok && r.Rate() > 0 {
		return r.Rate()
	}
	return Rate
}

func (l *loop) input(ev ui.InputEvent) bool {
	in := l.in
	switch ev.Kind {
	case ui.InputKeyDown, ui.InputKeyUp:
		if ev.Mods&(ui.Super|ui.Ctrl|ui.Alt) != 0 && ev.Kind == ui.InputKeyDown {
			return false // the menus' shortcuts
		}
		k := ev.PhysicalKey
		if k == ui.KeyUnknown {
			k = ev.Key
		}
		if ev.Kind == ui.InputKeyDown {
			if !ev.Repeat && !in.down[k] {
				in.pressed[k] = true
			}
			in.down[k] = true
		} else if in.down[k] {
			in.released[k] = true
			delete(in.down, k)
		}
		return true
	case ui.InputPointerDown, ui.InputPointerUp:
		in.x, in.y = ev.X, ev.Y
		if ev.Button >= 0 && ev.Button < 3 {
			down := ev.Kind == ui.InputPointerDown
			if down && !in.buttons[ev.Button] {
				in.bPressed[ev.Button] = true
			} else if !down && in.buttons[ev.Button] {
				in.bReleased[ev.Button] = true
			}
			in.buttons[ev.Button] = down
		}
		if ev.Kind == ui.InputPointerDown {
			l.wantFocus = true
		}
		return true
	case ui.InputPointerMove:
		in.x, in.y = ev.X, ev.Y
		in.inside = ev.X >= 0 && ev.Y >= 0 && ev.X < in.w && ev.Y < in.h
		return true
	case ui.InputScroll:
		in.wheelX += ev.DX
		in.wheelY += ev.DY
		return true
	}
	return false
}

func (l *loop) draw(p *ui.Painter, r ui.Rect) {
	p.AnimationFrame()
	l.in.w, l.in.h = r.W, r.H
	step := time.Second / time.Duration(l.rate())
	now := frameTime(p)
	if !l.last.IsZero() {
		// Painting twice a frame, at its one time, runs no step twice.
		l.lag += min(max(now.Sub(l.last), 0), maxCatchUp)
	}
	l.last = now
	for l.lag >= step {
		l.game.Update(l.in)
		l.in.step()
		l.lag -= step
	}
	l.game.Draw(p, r, float64(l.lag)/float64(step))
}

type loopKey struct{}

// frameTime returns the time of the frame p paints; tests replace it.
var frameTime = func(p *ui.Painter) time.Time { return p.Now() }

// View creates an element that runs g: Update in fixed steps of the game's
// rate, with the keys, pointer and wheel the element takes as Input, and
// Draw every frame, as long as it shows. It takes the keyboard focus as it
// appears and when clicked, and lets go of what was held when the window
// loses the keyboard. Size it as any element, as with Fill:
//
//	game.View(c, g).Fill()
func View(c *ui.Context, g Game) ui.Element {
	e := ui.Box(c).Focusable().AutoFocus()
	l := *ui.Local(e, loopKey{}, func() *loop { return &loop{in: newInput()} })
	l.game = g
	if l.wantFocus {
		l.wantFocus = false
		e.Focus()
	}
	l.in.focused = e.Focused()
	if !l.in.focused {
		l.in.releaseAll()
	}
	return e.HandleInput(l.input).Draw(l.draw)
}

// IntegerFit returns where a canvas of w×h pixels shows in r as large as
// it can with a whole number of the screen's pixels for each of its own,
// at least one, centered (letterboxed): pixel-perfect scaling, for p's
// screen, as Godot's viewport stretch and Stardew Valley draw pixel art.
// k is that number. A game draws its world into a ui.Canvas of a logical
// size, as 480×270, and shows it there:
//
//	dst, _ := game.IntegerFit(p, 480, 270, r)
//	p.Image(world.Bitmap(), dst, ui.FillBox)
//
// The pointer at x, y of the view is at ((x+r.X-dst.X)*480/dst.W,
// (y+r.Y-dst.Y)*270/dst.H) of the canvas.
func IntegerFit(p *ui.Painter, w, h int, r ui.Rect) (dst ui.Rect, k int) {
	if w <= 0 || h <= 0 {
		return ui.Rect{}, 0
	}
	s := p.Scale()
	k = max(int(min(r.W*s/float32(w), r.H*s/float32(h))), 1)
	dw, dh := float32(w*k)/s, float32(h*k)/s
	// On whole pixels of the screen, as the snapped image is.
	x := float32(int((r.X+(r.W-dw)/2)*s+0.5)) / s
	y := float32(int((r.Y+(r.H-dh)/2)*s+0.5)) / s
	return ui.Rect{X: x, Y: y, W: dw, H: dh}, k
}

// A Camera looks at a 2D world: the world's point X, Y shows at the middle
// of the view, Zoom times as large (0 is 1), turned Rotation degrees, as
// raylib's Camera2D. World units are DIPs at a zoom of 1.
type Camera struct {
	X, Y     float32
	Zoom     float32
	Rotation float32
}

func (cam Camera) zoom() float32 {
	if cam.Zoom == 0 {
		return 1
	}
	return cam.Zoom
}

// Transform returns the transform from the world to the window, for the
// view r, with the origin (0, 0): Painter.Transform(cam.Transform(r), 0, 0,
// draw). Draw is that, snapped and clipped.
func (cam Camera) Transform(r ui.Rect) ui.Transform {
	z := cam.zoom()
	t := ui.Translation(-cam.X, -cam.Y)
	if cam.Rotation != 0 {
		t = t.Then(ui.Rotation(-cam.Rotation))
	}
	return t.Then(ui.Scaling(z, z)).Then(ui.Translation(r.X+r.W/2, r.Y+r.H/2))
}

// Draw draws what fn draws in world coordinates through the camera into r,
// clipped to it. Unturned, the world moves by whole pixels of the screen,
// so that the edges of tiles drawn at whole world units stay on them,
// while parts of bitmaps, which ui.Painter.ImageRegion places where they
// are under a transform, glide.
func (cam Camera) Draw(p *ui.Painter, r ui.Rect, fn func()) {
	t := cam.Transform(r)
	if cam.Rotation == 0 {
		z := cam.zoom()
		s := p.Scale()
		snap := func(v float32) float32 { return float32(math.Round(float64(v*s))) / s }
		t = ui.Scaling(z, z).Then(ui.Translation(snap(r.X+r.W/2-cam.X*z), snap(r.Y+r.H/2-cam.Y*z)))
	}
	p.Clip(r, 0, func() { p.Transform(t, 0, 0, fn) })
}

// ToWorld returns the world's point at x, y of the view r, from its
// top-left corner, as Input.Pointer gives it.
func (cam Camera) ToWorld(r ui.Rect, x, y float32) (wx, wy float32) {
	z := cam.zoom()
	dx, dy := (x-r.W/2)/z, (y-r.H/2)/z
	if cam.Rotation != 0 {
		a := float64(cam.Rotation) * math.Pi / 180
		sin, cos := float32(math.Sin(a)), float32(math.Cos(a))
		dx, dy = dx*cos-dy*sin, dx*sin+dy*cos
	}
	return cam.X + dx, cam.Y + dy
}

// Visible returns the part of the world the camera shows in the view r:
// the bounds of it when turned. A game draws only the tiles in it.
func (cam Camera) Visible(r ui.Rect) ui.Rect {
	xs, ys := [4]float32{}, [4]float32{}
	for i, c := range [4][2]float32{{0, 0}, {r.W, 0}, {0, r.H}, {r.W, r.H}} {
		xs[i], ys[i] = cam.ToWorld(r, c[0], c[1])
	}
	x0, x1 := min(xs[0], xs[1], xs[2], xs[3]), max(xs[0], xs[1], xs[2], xs[3])
	y0, y1 := min(ys[0], ys[1], ys[2], ys[3]), max(ys[0], ys[1], ys[2], ys[3])
	return ui.Rect{X: x0, Y: y0, W: x1 - x0, H: y1 - y0}
}
