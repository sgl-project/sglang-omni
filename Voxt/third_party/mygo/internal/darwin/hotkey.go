//go:build darwin

package darwin

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/accelerator"
)

// Global shortcuts use Carbon's RegisterEventHotKey, which needs no
// accessibility permission.

const (
	hotKeySignature         = 0x4D79476F // 'MyGo'
	kEventClassKeyboard     = 0x6B657962 // 'keyb'
	kEventHotKeyPressed     = 5
	kEventParamDirectObject = 0x2D2D2D2D // '----'
	typeEventHotKeyID       = 0x686B6964 // 'hkid'

	carbonCmd     = 1 << 8
	carbonShift   = 1 << 9
	carbonOption  = 1 << 11
	carbonControl = 1 << 12
)

// keyCodes maps accelerator keys to macOS virtual key codes (ANSI layout).
var keyCodes = map[string]uint32{
	"a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05, "z": 0x06, "x": 0x07,
	"c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C, "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10,
	"t": 0x11, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17, "=": 0x18,
	"9": 0x19, "7": 0x1A, "-": 0x1B, "8": 0x1C, "0": 0x1D, "]": 0x1E, "o": 0x1F, "u": 0x20,
	"[": 0x21, "i": 0x22, "p": 0x23, "l": 0x25, "j": 0x26, "'": 0x27, "k": 0x28, ";": 0x29,
	"\\": 0x2A, ",": 0x2B, "/": 0x2C, "n": 0x2D, "m": 0x2E, ".": 0x2F, "`": 0x32,
	"Enter": 0x24, "Tab": 0x30, "Space": 0x31, "Backspace": 0x33, "Escape": 0x35,
	"Delete": 0x75, "Home": 0x73, "End": 0x77, "PageUp": 0x74, "PageDown": 0x79, "Insert": 0x72,
	"Left": 0x7B, "Right": 0x7C, "Down": 0x7D, "Up": 0x7E,
	"F1": 0x7A, "F2": 0x78, "F3": 0x63, "F4": 0x76, "F5": 0x60, "F6": 0x61, "F7": 0x62, "F8": 0x64,
	"F9": 0x65, "F10": 0x6D, "F11": 0x67, "F12": 0x6F, "F13": 0x69, "F14": 0x6B, "F15": 0x71,
	"F16": 0x6A, "F17": 0x40, "F18": 0x4F, "F19": 0x50, "F20": 0x5A,
	"VolumeUp": 0x48, "VolumeDown": 0x49, "VolumeMute": 0x4A,
	"Num0": 0x52, "Num1": 0x53, "Num2": 0x54, "Num3": 0x55, "Num4": 0x56, "Num5": 0x57,
	"Num6": 0x58, "Num7": 0x59, "Num8": 0x5B, "Num9": 0x5C, "NumDec": 0x41, "NumMult": 0x43,
	"NumAdd": 0x45, "NumDiv": 0x4B, "NumSub": 0x4E,
}

var hotKeyHandler = purego.NewCallback(func(next, event, userData uintptr) uintptr {
	var hk eventHotKeyID
	if getEventParameter(event, kEventParamDirectObject, typeEventHotKeyID, 0, unsafe.Sizeof(hk), 0, unsafe.Pointer(&hk)) == 0 {
		theBackend.h.HotkeyPressed(int(hk.ID))
	}
	return 0
})

func (b *Backend) RegisterHotkey(hid int, acc string) error {
	a, err := accelerator.Parse(acc, "darwin")
	if err != nil {
		return err
	}
	key := a.Key
	mods := uint32(0)
	if key == "+" {
		key = "="
		mods |= carbonShift
	}
	code, ok := keyCodes[key]
	if !ok {
		return fmt.Errorf("mygo: key %q cannot be used in a global shortcut", a.Key)
	}
	if a.Has(accelerator.Super) {
		mods |= carbonCmd
	}
	if a.Has(accelerator.Ctrl) {
		mods |= carbonControl
	}
	if a.Has(accelerator.Alt) {
		mods |= carbonOption
	}
	if a.Has(accelerator.Shift) {
		mods |= carbonShift
	}
	if !b.hkInstalled {
		spec := eventTypeSpec{Class: kEventClassKeyboard, Kind: kEventHotKeyPressed}
		var ref uintptr
		if st := installEventHandler(getApplicationEventTarget(), hotKeyHandler, 1, &spec, 0, &ref); st != 0 {
			return fmt.Errorf("mygo: cannot install hot key handler (%d)", st)
		}
		b.hkInstalled = true
	}
	var ref uintptr
	st := registerEventHotKey(code, mods, eventHotKeyID{Signature: hotKeySignature, ID: uint32(hid)}, getApplicationEventTarget(), 0, &ref)
	if st != 0 {
		return fmt.Errorf("mygo: cannot register shortcut %s, it may be used by another application (%d)", acc, st)
	}
	b.hotkeys[hid] = ref
	return nil
}

func (b *Backend) UnregisterHotkey(hid int) {
	if ref, ok := b.hotkeys[hid]; ok {
		unregisterEventHotKey(ref)
		delete(b.hotkeys, hid)
	}
}
