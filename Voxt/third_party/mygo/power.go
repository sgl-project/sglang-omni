package mygo

import (
	"sync"
	"time"

	"github.com/egoist/mygo/internal/platform"
)

// PowerModule reports power and session changes and keeps the computer
// awake. Use the Power singleton.
type PowerModule struct {
	mu       sync.Mutex
	watching bool

	onSuspend, onResume, onLockScreen, onUnlockScreen listeners[func()]
}

// Power reports power and session changes and keeps the computer awake.
var Power = &PowerModule{}

// OnSuspend is called when the system is about to sleep.
func (p *PowerModule) OnSuspend(fn func()) (off func()) { return p.add(&p.onSuspend, fn) }

// OnResume is called when the system woke up.
func (p *PowerModule) OnResume(fn func()) (off func()) { return p.add(&p.onResume, fn) }

// OnLockScreen is called when the screen locks (on Linux, when the screen
// saver starts, which usually locks it).
func (p *PowerModule) OnLockScreen(fn func()) (off func()) { return p.add(&p.onLockScreen, fn) }

// OnUnlockScreen is called when the screen unlocks.
func (p *PowerModule) OnUnlockScreen(fn func()) (off func()) { return p.add(&p.onUnlockScreen, fn) }

// add registers a listener and starts watching the system once.
func (p *PowerModule) add(l *listeners[func()], fn func()) func() {
	off := l.add(fn, false)
	p.mu.Lock()
	start := !p.watching
	p.watching = true
	p.mu.Unlock()
	if start {
		App.WhenReady(func() { backend().Power().Watch() })
	}
	return off
}

func (p *PowerModule) event(event string) {
	switch event {
	case "suspend":
		fire(&p.onSuspend)
	case "resume":
		fire(&p.onResume)
	case "lock-screen":
		fire(&p.onLockScreen)
	case "unlock-screen":
		fire(&p.onUnlockScreen)
	}
}

// KeepAwake keeps the computer from sleeping while idle, and with display
// the display from turning off, e.g. during a long task or while a video
// plays, until release is called:
//
//	release := mygo.Power.KeepAwake("Exporting the video", false)
//	defer release()
//
// The reason shows where the system lists what keeps it awake.
func (p *PowerModule) KeepAwake(reason string, display bool) (release func()) {
	needsApp("Power.KeepAwake")
	var r func()
	onMain(func() { r = backend().Power().KeepAwake(display, reason) })
	if r == nil {
		return func() {}
	}
	var once sync.Once
	return func() { once.Do(func() { onMain(r) }) }
}

// IsOnBattery reports whether the computer runs on battery power.
func (p *PowerModule) IsOnBattery() bool {
	return onMainValue(func() bool { return backend().Power().OnBattery() })
}

// IdleTime returns how long the user has not used the keyboard or mouse,
// or 0 when the system does not tell (some Linux desktops).
func (p *PowerModule) IdleTime() time.Duration {
	needsApp("Power.IdleTime")
	return onMainValue(func() time.Duration { return backend().Power().IdleTime() })
}

var _ platform.Power // the backends implement it
