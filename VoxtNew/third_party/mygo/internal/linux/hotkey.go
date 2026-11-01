//go:build linux && (amd64 || arm64)

package linux

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/egoist/mygo/internal/accelerator"
)

// On X11, global shortcuts grab their keys on the root window, and the key
// presses arrive through a GDK event filter. Elsewhere the desktop binds
// them through the XDG desktop portal (hotkey_portal.go).

var (
	x11 struct {
		loaded, ok bool

		stringToKeysym  func(name *byte) uint64
		keysymToKeycode func(dpy ptr, keysym uint64) uint8
		grabKey         func(dpy ptr, keycode int32, modifiers uint32, window uint64, ownerEvents bool, pointerMode, keyboardMode int32) int32
		ungrabKey       func(dpy ptr, keycode int32, modifiers uint32, window uint64) int32
		defaultRoot     func(dpy ptr) uint64
		flush           func(dpy ptr) int32

		displayType   func() uintptr
		xdisplay      func(display ptr) ptr
		errorTrapPush func(display ptr)
		errorTrapPop  func(display ptr) int32
		addFilter     func(window ptr, fn ptr, data ptr)
	}
	cbHotkeyFilter ptr
)

// X modifier masks.
const (
	shiftMask   = 1 << 0
	lockMask    = 1 << 1 // Caps Lock
	controlMask = 1 << 2
	mod1Mask    = 1 << 3 // Alt
	mod2Mask    = 1 << 4 // Num Lock
	mod4Mask    = 1 << 6 // Super
)

type grab struct {
	keycode uint8
	mods    uint32
}

// loadX11 binds the X11 functions once; it reports whether global
// shortcuts can work: GDK runs on an X server.
func loadX11() bool {
	if x11.loaded {
		return x11.ok
	}
	x11.loaded = true
	lib, err := open("libX11.so.6")
	if err != nil {
		return false
	}
	ok := bind(lib, &x11.stringToKeysym, "XStringToKeysym") &&
		bind(lib, &x11.keysymToKeycode, "XKeysymToKeycode") &&
		bind(lib, &x11.grabKey, "XGrabKey") &&
		bind(lib, &x11.ungrabKey, "XUngrabKey") &&
		bind(lib, &x11.defaultRoot, "XDefaultRootWindow") &&
		bind(lib, &x11.flush, "XFlush") &&
		bind(libGDK, &x11.displayType, "gdk_x11_display_get_type") &&
		bind(libGDK, &x11.xdisplay, "gdk_x11_display_get_xdisplay") &&
		bind(libGDK, &x11.errorTrapPush, "gdk_x11_display_error_trap_push") &&
		bind(libGDK, &x11.errorTrapPop, "gdk_x11_display_error_trap_pop") &&
		bind(libGDK, &x11.addFilter, "gdk_window_add_filter")
	if !ok {
		return false
	}
	x11.ok = gTypeCheckInstanceIsA(gdkDisplayGetDefault(), x11.displayType())
	if x11.ok {
		cbHotkeyFilter = purego.NewCallback(hotkeyFilter)
		x11.addFilter(0, cbHotkeyFilter, 0)
	}
	return x11.ok
}

// hotkeyFilter reports the key presses of grabbed shortcuts.
func hotkeyFilter(xevent, event, data ptr) int32 {
	const keyPress, filterContinue, filterRemove = 2, 0, 2
	ev := *(*unsafe.Pointer)(unsafe.Pointer(&xevent))
	if *(*int32)(ev) != keyPress {
		return filterContinue
	}
	// XKeyEvent: state and keycode follow the time.
	state := *(*uint32)(unsafe.Add(ev, 80)) & (shiftMask | controlMask | mod1Mask | mod4Mask)
	keycode := uint8(*(*uint32)(unsafe.Add(ev, 84)))
	for id, g := range theBackend.grabs {
		if g.keycode == keycode && g.mods == state {
			theBackend.h.HotkeyPressed(id)
			return filterRemove
		}
	}
	return filterContinue
}

func (b *Backend) RegisterHotkey(id int, acc string) error {
	a, err := accelerator.Parse(acc, "linux")
	if err != nil {
		return err
	}
	if portalShortcutsOnly || !loadX11() {
		return b.bindShortcut(id, a)
	}
	display := gdkDisplayGetDefault()
	dpy := x11.xdisplay(display)
	keycode := x11.keysymToKeycode(dpy, x11.stringToKeysym(cs(a.Keysym())))
	if keycode == 0 {
		return fmt.Errorf("mygo: the keyboard has no key for %q", acc)
	}
	var mods uint32
	for m, x := range map[accelerator.Modifiers]uint32{accelerator.Shift: shiftMask, accelerator.Ctrl: controlMask, accelerator.Alt: mod1Mask, accelerator.Super: mod4Mask} {
		if a.Has(m) {
			mods |= x
		}
	}
	g := grab{keycode, mods}
	root := x11.defaultRoot(dpy)
	x11.errorTrapPush(display)
	for _, lock := range []uint32{0, lockMask, mod2Mask, lockMask | mod2Mask} {
		x11.grabKey(dpy, int32(keycode), mods|lock, root, false, 1, 1) // GrabModeAsync
	}
	if x11.errorTrapPop(display) != 0 {
		b.ungrab(g)
		return errors.New("mygo: " + acc + " is already a global shortcut of another app")
	}
	if b.grabs == nil {
		b.grabs = map[int]grab{}
	}
	b.grabs[id] = g
	return nil
}

func (b *Backend) UnregisterHotkey(id int) {
	if g, ok := b.grabs[id]; ok {
		delete(b.grabs, id)
		b.ungrab(g)
	}
	b.unbindShortcut(id)
}

func (b *Backend) ungrab(g grab) {
	display := gdkDisplayGetDefault()
	dpy := x11.xdisplay(display)
	root := x11.defaultRoot(dpy)
	x11.errorTrapPush(display)
	for _, lock := range []uint32{0, lockMask, mod2Mask, lockMask | mod2Mask} {
		x11.ungrabKey(dpy, int32(g.keycode), g.mods|lock, root)
	}
	x11.flush(dpy)
	x11.errorTrapPop(display)
}
