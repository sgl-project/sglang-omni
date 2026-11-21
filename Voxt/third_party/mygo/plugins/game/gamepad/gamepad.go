// Package gamepad reads game controllers: their buttons, triggers and
// sticks in the standard layout of an Xbox controller, which the W3C's
// Gamepad API and SDL's game controller database name, whatever pad it is:
//
//	for _, pad := range gamepad.Pads() {
//		s := pad.State()
//		x, y := s.LeftStick()
//		if s.Pressed(gamepad.South) {
//			jump()
//		}
//	}
//
// A game reads the pads' states once a frame, or once a step of its world,
// and keeps the last to tell a button just pressed from one held
// (State.JustPressed).
//
// The pads come from GameController.framework on macOS (Xbox, PlayStation,
// Switch Pro and MFi controllers), XInput on Windows (Xbox controllers and
// those that act as one, as Steam Input makes of others), evdev on Linux
// (/dev/input, readable by the user at the seat), and the Gamepad API in
// browsers. On macOS, input reaches the app in front only, as games
// expect. Every function and method is safe to call from any goroutine.
package gamepad

import (
	"math"
	"sync"
	"time"
)

// Button is a button of the standard layout, named by place: South is A
// on an Xbox controller, Cross on a PlayStation one, B on a Nintendo one.
type Button uint8

// The buttons, in the order of the W3C's standard gamepad.
const (
	South Button = iota
	East
	West
	North
	LeftShoulder
	RightShoulder
	LeftTrigger // analog: Value is from 0 to 1
	RightTrigger
	Back // View, Select, Share, Minus
	Start
	LeftStickButton
	RightStickButton
	Up
	Down
	Left
	Right
	Guide // the Xbox, PlayStation or Home button, where the system leaves it
	ButtonCount
)

// Axis is an axis of the sticks, from -1 to 1: right and down are
// positive, as on the screen.
type Axis uint8

// The axes.
const (
	LeftX Axis = iota
	LeftY
	RightX
	RightY
	AxisCount
)

// State is what a pad's controls read at one time.
type State struct {
	// Buttons are from 0 to 1: 0 or 1 but for the triggers.
	Buttons [ButtonCount]float32
	// Axes are from -1 to 1, as the pad reports them, without a dead
	// zone.
	Axes [AxisCount]float32
}

// Deadzone is how far from the middle a stick must move to count, as a
// fraction of its reach: sticks rest a little off their middle.
const Deadzone = 0.15

// pressThreshold is how far an analog trigger must go to be pressed.
const pressThreshold = 0.25

// Value returns how far b is pressed, from 0 to 1.
func (s State) Value(b Button) float32 {
	if b >= ButtonCount {
		return 0
	}
	return s.Buttons[b]
}

// Pressed reports whether b is pressed: a trigger a quarter of the way.
func (s State) Pressed(b Button) bool { return s.Value(b) >= pressThreshold }

// JustPressed reports whether b is pressed in s and was not in last, the
// state read before.
func (s State) JustPressed(last State, b Button) bool { return s.Pressed(b) && !last.Pressed(b) }

// JustReleased reports whether b was pressed in last and is not in s.
func (s State) JustReleased(last State, b Button) bool { return !s.Pressed(b) && last.Pressed(b) }

// Axis returns the raw value of a, from -1 to 1.
func (s State) Axis(a Axis) float32 {
	if a >= AxisCount {
		return 0
	}
	return s.Axes[a]
}

// LeftStick returns where the left stick is, each coordinate from -1 to 1,
// right and down positive, with a radial dead zone: 0 near the middle,
// then from 0 at Deadzone to 1 at the rim, the same in every direction.
func (s State) LeftStick() (x, y float32) { return stick(s.Axes[LeftX], s.Axes[LeftY]) }

// RightStick returns where the right stick is, as LeftStick.
func (s State) RightStick() (x, y float32) { return stick(s.Axes[RightX], s.Axes[RightY]) }

func stick(x, y float32) (float32, float32) {
	m := float32(math.Hypot(float64(x), float64(y)))
	if m < Deadzone {
		return 0, 0
	}
	k := min((m-Deadzone)/(1-Deadzone), 1) / m
	return x * k, y * k
}

// A Pad is a game controller, from when it connects until it disconnects;
// connected again, it is a Pad anew.
type Pad struct {
	dev       device
	name      string
	connected bool
}

// Name returns the pad's name, as its maker gives it.
func (p *Pad) Name() string { return p.name }

// Connected reports whether the pad is still connected.
func (p *Pad) Connected() bool {
	mu.Lock()
	defer mu.Unlock()
	return p.connected
}

// State reads the pad's controls now; a pad no longer connected reads
// nothing pressed.
func (p *Pad) State() State {
	mu.Lock()
	defer mu.Unlock()
	if !p.connected {
		return State{}
	}
	var s State
	if !p.dev.read(&s) {
		p.disconnect()
		return State{}
	}
	return s
}

func (p *Pad) disconnect() {
	p.connected = false
	p.dev.close()
	for i, q := range pads {
		if q == p {
			pads = append(pads[:i], pads[i+1:]...)
			break
		}
	}
}

// A device is a pad as a platform knows it.
type device interface {
	// key identifies the device while it is connected.
	key() string
	name() string
	// read reads the device's controls, and reports false once it is
	// gone.
	read(s *State) bool
	close()
}

var (
	mu       sync.Mutex
	pads     []*Pad
	scanned  time.Time
	scanEach = time.Second // how often to look for pads anew
)

// Pads returns the pads connected, in the order they connected. It looks
// for pads connected since at most once a second.
func Pads() []*Pad {
	mu.Lock()
	defer mu.Unlock()
	if now := time.Now(); scanned.IsZero() || now.Sub(scanned) >= scanEach {
		scanned = now
		known := map[string]device{}
		for _, p := range pads {
			known[p.dev.key()] = p.dev
		}
		update(scanDevices(known))
	}
	return append([]*Pad(nil), pads...)
}

// scanDevices finds the devices connected, the known ones among them as
// they are; tests replace it.
var scanDevices = scan

// update takes the devices a scan found: the pads of devices gone
// disconnect, and new devices connect as pads.
func update(found []device) {
	present := map[string]bool{}
	for _, d := range found {
		present[d.key()] = true
	}
	for _, p := range append([]*Pad(nil), pads...) {
		if !present[p.dev.key()] {
			p.disconnect()
		}
	}
	have := map[string]bool{}
	for _, p := range pads {
		have[p.dev.key()] = true
	}
	for _, d := range found {
		if !have[d.key()] {
			have[d.key()] = true
			pads = append(pads, &Pad{dev: d, name: d.name(), connected: true})
		}
	}
}
