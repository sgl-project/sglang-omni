package vt

import (
	"runtime"
	"unsafe"
)

// Key is a GhosttyKey: a key by its place on the keyboard, as the W3C's
// KeyboardEvent.code names them.
type Key int32

// Keys, in the order of GhosttyKey.
const (
	KeyUnidentified Key = iota
	KeyBackquote
	KeyBackslash
	KeyBracketLeft
	KeyBracketRight
	KeyComma
	KeyDigit0
	KeyDigit1
	KeyDigit2
	KeyDigit3
	KeyDigit4
	KeyDigit5
	KeyDigit6
	KeyDigit7
	KeyDigit8
	KeyDigit9
	KeyEqual
	KeyIntlBackslash
	KeyIntlRo
	KeyIntlYen
	KeyA
	KeyB
	KeyC
	KeyD
	KeyE
	KeyF
	KeyG
	KeyH
	KeyI
	KeyJ
	KeyK
	KeyL
	KeyM
	KeyN
	KeyO
	KeyP
	KeyQ
	KeyR
	KeyS
	KeyT
	KeyU
	KeyV
	KeyW
	KeyX
	KeyY
	KeyZ
	KeyMinus
	KeyPeriod
	KeyQuote
	KeySemicolon
	KeySlash
	KeyAltLeft
	KeyAltRight
	KeyBackspace
	KeyCapsLock
	KeyContextMenu
	KeyControlLeft
	KeyControlRight
	KeyEnter
	KeyMetaLeft
	KeyMetaRight
	KeyShiftLeft
	KeyShiftRight
	KeySpace
	KeyTab
	KeyConvert
	KeyKanaMode
	KeyNonConvert
	KeyDelete
	KeyEnd
	KeyHelp
	KeyHome
	KeyInsert
	KeyPageDown
	KeyPageUp
	KeyArrowDown
	KeyArrowLeft
	KeyArrowRight
	KeyArrowUp
)

// KeyEscape and the function keys follow the 41 keys of the numpad.
const (
	KeyEscape Key = KeyArrowUp + 1 + 41 + iota
	KeyF1
)

// Mods are modifier keys (GhosttyMods): Shift, Ctrl, Alt and Super, the
// same bits as MyGo's.
type Mods uint16

const (
	ModShift Mods = 1 << iota
	ModCtrl
	ModAlt
	ModSuper
)

// KeyAction is what happened to a key.
type KeyAction int32

const (
	KeyRelease KeyAction = iota
	KeyPress
	KeyRepeat
)

// KeyEvent is a key to encode for the program.
type KeyEvent struct {
	Action KeyAction
	Key    Key
	Mods   Mods
	// Consumed are the modifiers that made Text, as Shift an "A".
	Consumed Mods
	// Text is what the key types, without what Ctrl or Alt change; empty
	// for keys that type nothing.
	Text string
	// Unshifted is the character the key types without modifiers.
	Unshifted rune
	// Composing tells that an input method is composing.
	Composing bool
}

// KeyEncoder encodes keys as the terminal's modes ask: legacy sequences,
// xterm's modifyOtherKeys or the Kitty keyboard protocol.
type KeyEncoder struct {
	h, ev uintptr
	buf   [128]byte
}

// NewKeyEncoder makes a key encoder.
func NewKeyEncoder() (*KeyEncoder, error) {
	e := &KeyEncoder{}
	if err := result(call(fnKeyEncoderNew, 0, uintptr(unsafe.Pointer(&e.h)))); err != nil {
		return nil, err
	}
	if err := result(call(fnKeyEventNew, 0, uintptr(unsafe.Pointer(&e.ev)))); err != nil {
		e.Free()
		return nil, err
	}
	return e, nil
}

// Free frees the encoder.
func (e *KeyEncoder) Free() {
	if e.ev != 0 {
		call(fnKeyEventFree, e.ev)
	}
	if e.h != 0 {
		call(fnKeyEncoderFree, e.h)
	}
	*e = KeyEncoder{}
}

// Sync takes the terminal's modes, and whether Option is Alt on macOS.
func (e *KeyEncoder) Sync(t *Terminal, optionAsAlt bool) {
	call(fnKeyEncoderSetoptFromTerminal, e.h, t.h)
	v := int32(b2u(optionAsAlt))
	call(fnKeyEncoderSetopt, e.h, 6, uintptr(unsafe.Pointer(&v)))
}

// Encode returns the bytes the program reads for k, valid until the next
// call; nil for keys that send nothing.
func (e *KeyEncoder) Encode(k KeyEvent) []byte {
	call(fnKeyEventSetAction, e.ev, uintptr(k.Action))
	call(fnKeyEventSetKey, e.ev, uintptr(k.Key))
	call(fnKeyEventSetMods, e.ev, uintptr(k.Mods))
	call(fnKeyEventSetConsumedMods, e.ev, uintptr(k.Consumed))
	call(fnKeyEventSetComposing, e.ev, b2u(k.Composing))
	call(fnKeyEventSetUnshifted, e.ev, uintptr(k.Unshifted))
	// The event keeps the text's address until it is encoded.
	var pin runtime.Pinner
	defer pin.Unpin()
	text := []byte(k.Text)
	if len(text) > 0 {
		pin.Pin(&text[0])
		call(fnKeyEventSetUTF8, e.ev, uintptr(unsafe.Pointer(&text[0])), uintptr(len(text)))
	} else {
		call(fnKeyEventSetUTF8, e.ev, 0, 0)
	}
	var n uintptr
	r := call(fnKeyEncoderEncode, e.h, e.ev, uintptr(unsafe.Pointer(&e.buf[0])), uintptr(len(e.buf)), uintptr(unsafe.Pointer(&n)))
	if Result(int32(r)) == OutOfSpace && n > 0 {
		big := make([]byte, n)
		r = call(fnKeyEncoderEncode, e.h, e.ev, uintptr(unsafe.Pointer(&big[0])), n, uintptr(unsafe.Pointer(&n)))
		call(fnKeyEventSetUTF8, e.ev, 0, 0)
		if !ok(r) {
			return nil
		}
		return big[:n]
	}
	call(fnKeyEventSetUTF8, e.ev, 0, 0)
	if !ok(r) || n == 0 {
		return nil
	}
	return e.buf[:n]
}

// MouseAction is what the pointer did.
type MouseAction int32

const (
	MousePress MouseAction = iota
	MouseRelease
	MouseMotion
)

// Mouse buttons: 1 left, 2 right, 3 middle, 4 and 5 the wheel up and down,
// 6 and 7 left and right.
const (
	ButtonNone   = 0
	ButtonLeft   = 1
	ButtonRight  = 2
	ButtonMiddle = 3
	WheelUp      = 4
	WheelDown    = 5
	WheelLeft    = 6
	WheelRight   = 7
)

// MouseEncoder encodes pointer events in the format and for the events
// the program asked for.
type MouseEncoder struct {
	h, ev uintptr
	buf   [64]byte
}

// NewMouseEncoder makes a mouse encoder.
func NewMouseEncoder() (*MouseEncoder, error) {
	m := &MouseEncoder{}
	if err := result(call(fnMouseEncoderNew, 0, uintptr(unsafe.Pointer(&m.h)))); err != nil {
		return nil, err
	}
	if err := result(call(fnMouseEventNew, 0, uintptr(unsafe.Pointer(&m.ev)))); err != nil {
		m.Free()
		return nil, err
	}
	// Motion is reported once per cell, as xterm does.
	v := true
	call(fnMouseEncoderSetopt, m.h, 4, uintptr(unsafe.Pointer(&v)))
	return m, nil
}

// Free frees the encoder.
func (m *MouseEncoder) Free() {
	if m.ev != 0 {
		call(fnMouseEventFree, m.ev)
	}
	if m.h != 0 {
		call(fnMouseEncoderFree, m.h)
	}
	*m = MouseEncoder{}
}

// Sync takes the terminal's mouse modes, the geometry of the screen in
// pixels, and whether a button is held.
func (m *MouseEncoder) Sync(t *Terminal, width, height, cellWidth, cellHeight, padLeft, padTop int, pressed bool) {
	call(fnMouseEncoderSetoptFromTerminal, m.h, t.h)
	s := mouseEncoderSize{
		size: unsafe.Sizeof(mouseEncoderSize{}), screenWidth: uint32(max(width, 1)), screenHeight: uint32(max(height, 1)),
		cellWidth: uint32(max(cellWidth, 1)), cellHeight: uint32(max(cellHeight, 1)),
		paddingTop: uint32(max(padTop, 0)), paddingLeft: uint32(max(padLeft, 0)),
	}
	call(fnMouseEncoderSetopt, m.h, 2, uintptr(unsafe.Pointer(&s)))
	call(fnMouseEncoderSetopt, m.h, 3, uintptr(unsafe.Pointer(&pressed)))
}

// Encode returns the bytes the program reads for a pointer event at x, y
// pixels, valid until the next call; nil when it asked for no such event.
func (m *MouseEncoder) Encode(action MouseAction, button int, mods Mods, x, y float32) []byte {
	call(fnMouseEventSetAction, m.ev, uintptr(action))
	if button == ButtonNone {
		call(fnMouseEventClearButton, m.ev)
	} else {
		call(fnMouseEventSetButton, m.ev, uintptr(button))
	}
	call(fnMouseEventSetMods, m.ev, uintptr(mods))
	mouseEventSetPosition(m.ev, mousePosition{x, y})
	var n uintptr
	if !ok(call(fnMouseEncoderEncode, m.h, m.ev, uintptr(unsafe.Pointer(&m.buf[0])), uintptr(len(m.buf)), uintptr(unsafe.Pointer(&n)))) || n == 0 {
		return nil
	}
	return m.buf[:n]
}

// EncodeFocus returns the report of the terminal gaining or losing the
// focus (mode 1004).
func EncodeFocus(gained bool) []byte {
	var buf [8]byte
	var n uintptr
	if !ok(call(fnFocusEncode, b2u(!gained), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), uintptr(unsafe.Pointer(&n)))) {
		return nil
	}
	return buf[:n]
}

// EncodePaste returns text as the program reads a paste: bracketed (mode
// 2004), or with newlines as carriage returns, and without the control
// characters that could run commands.
func EncodePaste(text string, bracketed bool) []byte {
	data := []byte(text)
	if len(data) == 0 {
		return nil
	}
	out := make([]byte, len(data)+16)
	var n uintptr
	r := call(fnPasteEncode, uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), b2u(bracketed), uintptr(unsafe.Pointer(&out[0])), uintptr(len(out)), uintptr(unsafe.Pointer(&n)))
	if Result(int32(r)) == OutOfSpace && n > 0 {
		data = []byte(text) // encoding may have changed it in place
		out = make([]byte, n)
		r = call(fnPasteEncode, uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), b2u(bracketed), uintptr(unsafe.Pointer(&out[0])), n, uintptr(unsafe.Pointer(&n)))
	}
	if !ok(r) {
		return nil
	}
	return out[:n]
}

// PasteEvent tells the program what the clipboard offers a paste, by the
// MIME types, for it to read what it wants (mode 5522, the Kitty
// clipboard protocol); the program's reads come to Effects.ClipboardRead,
// granted. The event goes to the program through Effects.WritePTY. It
// reports whether the event was sent: it is not when the mode is not set
// or the terminal has no ClipboardRead, when the paste is the embedder's
// to encode (EncodePaste).
func (t *Terminal) PasteEvent(mimes []string) bool {
	if !t.Mode(ModePasteEvents) || t.effects.ClipboardRead == nil {
		return false
	}
	var keep [][]byte
	cs := make([]cString, 0, len(mimes))
	for _, m := range mimes {
		cs = append(cs, cStringOf(m, &keep))
	}
	p := paste{size: unsafe.Sizeof(paste{}), source: 0 /* the clipboard */, mimesLen: uintptr(len(cs))}
	if len(cs) > 0 {
		p.mimes = uintptr(unsafe.Pointer(&cs[0]))
		// The library asks a reader for data only when it pastes text
		// itself, which it does not for an event; one is still required.
		p.reader = mimeReader{read: cbNoRead}
	}
	var written bool
	r := call(fnTerminalPaste, t.h, uintptr(unsafe.Pointer(&p)), uintptr(unsafe.Pointer(&written)))
	runtime.KeepAlive(keep)
	runtime.KeepAlive(cs)
	return ok(r) && written
}

// PasteIsSafe reports whether pasting text cannot run a command by itself.
func PasteIsSafe(text string) bool {
	if text == "" {
		return true
	}
	b := []byte(text)
	return cbool(call(fnPasteIsSafe, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b))))
}
