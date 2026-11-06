package vt

import (
	"bytes"
	"errors"
	"runtime"
	"strings"
	"sync"
	"unsafe"
)

// Terminal is a GhosttyTerminal: the screen, its scrollback, the cursor
// and the modes, which the output of programs changes.
type Terminal struct {
	h  uintptr
	id uintptr
	// Effects are called during Write, on its goroutine; they must not
	// call Write.
	effects Effects
	// silent is set while a clipboard read is refused: the refusal the
	// library writes for it, a Kitty protocol EPERM, is dropped, as the
	// read of a terminal that is not the one to answer.
	silent bool
	// deferred is the clipboard read the embedder answers later: the
	// library's refusal, which follows the read's effect, is held back
	// and captured in it.
	deferred *DeferredClipboardRead
	// reply gathers what the library writes while it answers a clipboard
	// read, which WritePTY then gets at once.
	reply *[]byte
}

// ClipboardRead is a program's request to read the clipboard.
type ClipboardRead struct {
	// Mimes are the types asked for, in order of preference; List asks
	// for the types available.
	Mimes []string
	List  bool
	// Primary is the selection clipboard of X11.
	Primary bool
	// Granted tells that the program holds the terminal's permission, as
	// when it reads what a paste offered it.
	Granted bool
	// CanRemember tells that the program asked with a password (the one
	// a paste event gave it, with its name): a terminal that pasted
	// knows it — Granted says so — and one that does not never can, and
	// leaves the read to the one it is for.
	CanRemember bool
}

// ClipboardReply answers a ClipboardRead: Contents by MIME type, of the
// types asked for, and Available when the list was asked for. Without
// contents the read is refused. A deferred read carries Deferred instead.
type ClipboardReply struct {
	Contents  map[string][]byte
	Available []string
	// Deferred is the read to answer later, with ClipboardDeferred.
	Deferred *DeferredClipboardRead
}

// Effects are what a terminal asks of its embedder while it processes
// output, all optional. They run during Terminal.Write, which they must
// not call.
type Effects struct {
	// WritePTY answers the program, as to a query of the cursor's
	// position. p is only valid during the call.
	WritePTY func(p []byte)
	Bell     func()
	// TitleChanged and PwdChanged tell that Title or Pwd changed.
	TitleChanged func()
	PwdChanged   func()
	// RenderHold tells that the program started or ended a synchronized
	// update (mode 2026), during which the screen should not show.
	RenderHold func(held bool)
	// Clipboard writes text to the clipboard (OSC 52), and reports
	// whether it did. primary is the selection clipboard of X11.
	Clipboard func(text string, primary bool) bool
	// ClipboardRead answers a program's read of the clipboard (OSC 52
	// with "?", the Kitty clipboard protocol's OSC 5522): the contents of
	// the MIME types asked for, and the types available when the program
	// asks for the list. ClipboardAnswered writes the reply, encoded, to
	// WritePTY in one call as ClipboardRead returns; ClipboardRefused
	// refuses it (an empty OSC 52 reply, EPERM for OSC 5522), and the
	// EPERM is dropped, so that a terminal that is not the one to answer
	// stays silent; ClipboardDeferred holds the refusal back and answers
	// later through Terminal.AnswerClipboard, for a terminal that must
	// ask the user first.
	ClipboardRead func(req ClipboardRead) (reply ClipboardReply, answer ClipboardAnswer)
	// Size reports the size of the screen in cells and of a cell in
	// pixels, for size reports.
	Size func() (cols, rows, cellWidth, cellHeight int)
	// Dark reports whether the colors are dark, for color scheme reports.
	Dark func() bool
	// Notify shows a desktop notification (OSC 9, OSC 777).
	Notify func(title, body string)
	// Progress reports the progress of the program (OSC 9;4): state is
	// 0 to remove it, 1 to set it to progress (0 to 100), 2 for an error,
	// 3 for indeterminate progress and 4 paused.
	Progress func(state, progress int)
}

// terminals finds the terminal of a callback by its userdata.
var terminals struct {
	sync.Mutex
	next uintptr
	m    map[uintptr]*Terminal
}

func terminalOf(id uintptr) *Terminal {
	terminals.Lock()
	defer terminals.Unlock()
	return terminals.m[id]
}

// Terminal options and data (GhosttyTerminalOption, GhosttyTerminalData).
const (
	optUserdata            = 0
	optWritePTY            = 1
	optBell                = 2
	optTitleChanged        = 5
	optSize                = 6
	optColorScheme         = 7
	optDeviceAttributes    = 8
	optColorForeground     = 11
	optColorBackground     = 12
	optColorCursor         = 13
	optColorPalette        = 14
	optKittyImageLimit     = 15
	optDefaultCursorStyle  = 22
	optDefaultCursorBlink  = 23
	optPwdChanged          = 25
	optClipboardWrite      = 26
	optClipboardRead       = 38
	optScrollbackMaxBytes  = 27
	optDesktopNotification = 29
	optProgressReport      = 30
	optModeDefault         = 33
	optMode                = 34
	optTerminfoName        = 37
	optResizePullBack      = 40
	optRenderHold          = 41

	dataCols           = 1
	dataRows           = 2
	dataCursorX        = 3
	dataCursorY        = 4
	dataActiveScreen   = 6
	dataScrollbar      = 9
	dataMouseTracking  = 11
	dataTitle          = 12
	dataPwd            = 13
	dataViewportActive = 32
	dataMode           = 37
	dataMouseShape     = 41
)

// NewTerminal makes a terminal of cols columns and rows rows.
func NewTerminal(cols, rows int, effects Effects) (*Terminal, error) {
	if !Loaded() {
		return nil, errors.New("libghostty-vt is not loaded")
	}
	t := &Terminal{effects: effects}
	if err := result(call(fnTerminalNew, 0, uintptr(unsafe.Pointer(&t.h)), uintptr(clamp16(cols)), uintptr(clamp16(rows)))); err != nil {
		return nil, err
	}
	terminals.Lock()
	if terminals.m == nil {
		terminals.m = map[uintptr]*Terminal{}
	}
	terminals.next++
	t.id = terminals.next
	terminals.m[t.id] = t
	terminals.Unlock()
	t.setValue(optUserdata, t.id)
	t.setValue(optWritePTY, cbWritePTY)
	t.setValue(optBell, cbBell)
	t.setValue(optTitleChanged, cbTitleChanged)
	t.setValue(optPwdChanged, cbPwdChanged)
	t.setValue(optRenderHold, cbRenderHold)
	t.setValue(optClipboardWrite, cbClipboardWrite)
	t.setValue(optClipboardRead, cbClipboardRead)
	t.setValue(optSize, cbSize)
	t.setValue(optColorScheme, cbColorScheme)
	t.setValue(optDeviceAttributes, cbDeviceAttributes)
	t.setValue(optDesktopNotification, cbDesktopNotification)
	t.setValue(optProgressReport, cbProgressReport)
	// Images of the Kitty graphics protocol are not drawn: refuse them,
	// so that programs fall back to something else.
	var limit uint64
	t.set(optKittyImageLimit, unsafe.Pointer(&limit))
	return t, nil
}

// set sets an option to a pointer to its value. The pointer is converted
// in the call, so that the value stays put while the library reads it
// (see call).
func (t *Terminal) set(opt int, value unsafe.Pointer) error {
	return result(call(fnTerminalSet, t.h, uintptr(opt), uintptr(value)))
}

// setValue sets an option that is a value itself: a callback, or the
// userdata.
func (t *Terminal) setValue(opt int, value uintptr) error {
	return result(call(fnTerminalSet, t.h, uintptr(opt), value))
}

func (t *Terminal) get(data int, out unsafe.Pointer) error {
	return result(call(fnTerminalGet, t.h, uintptr(data), uintptr(out)))
}

// Free frees the terminal.
func (t *Terminal) Free() {
	if t.h == 0 {
		return
	}
	call(fnTerminalFree, t.h)
	t.h = 0
	terminals.Lock()
	delete(terminals.m, t.id)
	terminals.Unlock()
}

// Write processes output of the program.
func (t *Terminal) Write(p []byte) {
	if len(p) > 0 {
		// A read that was deferred but whose refusal never came, as of a
		// library of another version, is stale.
		t.deferred = nil
		call(fnTerminalVTWrite, t.h, uintptr(unsafe.Pointer(&p[0])), uintptr(len(p)))
		t.deferred = nil
		// A read that writes no refusal, as OSC 52's, leaves silent set;
		// it is for that read alone.
		t.silent = false
	}
}

// Resize sets the size of the screen in cells, and of a cell in pixels.
func (t *Terminal) Resize(cols, rows, cellWidth, cellHeight int) error {
	return result(call(fnTerminalResize, t.h, uintptr(clamp16(cols)), uintptr(clamp16(rows)), uintptr(max(cellWidth, 1)), uintptr(max(cellHeight, 1))))
}

// Size returns the size of the screen in cells.
func (t *Terminal) Size() (cols, rows int) {
	var c, r uint16
	t.get(dataCols, unsafe.Pointer(&c))
	t.get(dataRows, unsafe.Pointer(&r))
	return int(c), int(r)
}

// Cursor returns the cursor's column and row in the active screen, the
// bottom of the scrollback.
func (t *Terminal) Cursor() (x, y int) {
	var cx, cy uint16
	t.get(dataCursorX, unsafe.Pointer(&cx))
	t.get(dataCursorY, unsafe.Pointer(&cy))
	return int(cx), int(cy)
}

// Reset resets the terminal fully (RIS).
func (t *Terminal) Reset() { call(fnTerminalReset, t.h) }

// SetColors sets the default colors, which programs may change: the
// foreground, the background, the cursor's (nil for the text's) and the
// palette of 256 colors (nil for Ghostty's).
func (t *Terminal) SetColors(fg, bg RGB, cursor *RGB, palette *[256]RGB) {
	t.set(optColorForeground, unsafe.Pointer(&fg))
	t.set(optColorBackground, unsafe.Pointer(&bg))
	t.set(optColorCursor, unsafe.Pointer(cursor))
	t.set(optColorPalette, unsafe.Pointer(palette))
}

// SetScrollback limits the scrollback to about n bytes.
func (t *Terminal) SetScrollback(n int) {
	v := uintptr(max(n, 0))
	t.set(optScrollbackMaxBytes, unsafe.Pointer(&v))
}

// SetTerminfoName sets the name of the terminfo entry the terminal reports
// (XTGETTCAP TN).
func (t *Terminal) SetTerminfoName(name string) {
	b := []byte(name)
	s := cString{len: uintptr(len(b))}
	// The string holds the address of b, which the call does not see.
	var pin runtime.Pinner
	defer pin.Unpin()
	if len(b) > 0 {
		pin.Pin(&b[0])
		s.ptr = uintptr(unsafe.Pointer(&b[0]))
	}
	t.set(optTerminfoName, unsafe.Pointer(&s))
}

// CursorStyle is the shape of the cursor.
type CursorStyle int32

const (
	CursorBar CursorStyle = iota
	CursorBlock
	CursorUnderline
	CursorBlockHollow
)

// SetDefaultCursor sets the cursor that programs reset to (DECSCUSR 0).
func (t *Terminal) SetDefaultCursor(s CursorStyle, blink bool) {
	t.set(optDefaultCursorStyle, unsafe.Pointer(&s))
	t.set(optDefaultCursorBlink, unsafe.Pointer(&blink))
}

func (t *Terminal) str(data int) string {
	var s cString
	if t.get(data, unsafe.Pointer(&s)) != nil {
		return ""
	}
	return goString(s.ptr, s.len)
}

// Title returns the title programs set (OSC 0, OSC 2).
func (t *Terminal) Title() string { return t.str(dataTitle) }

// Pwd returns the working directory the shell reported (OSC 7), as it
// reported it: a file URL, or a path.
func (t *Terminal) Pwd() string { return t.str(dataPwd) }

// Mode is a GhosttyMode: a DEC private mode, or an ANSI one with bit 15.
type Mode uint16

// Modes.
const (
	ModeCursorKeys     Mode = 1
	ModeCursorVisible  Mode = 25
	ModeFocusEvent     Mode = 1004
	ModeAltScroll      Mode = 1007
	ModeBracketedPaste Mode = 2004
	// ModePasteEvents has a paste tell the program what the clipboard
	// holds, which it reads itself (the Kitty clipboard protocol).
	ModePasteEvents Mode = 5522
	ModeSyncOutput  Mode = 2026
	ModeGraphemes   Mode = 2027
)

// Mode reports whether a mode is set.
func (t *Terminal) Mode(m Mode) bool {
	c := modeConfig{mode: uint16(m)}
	return t.get(dataMode, unsafe.Pointer(&c)) == nil && c.value
}

// SetMode sets a mode.
func (t *Terminal) SetMode(m Mode, v bool) {
	c := modeConfig{mode: uint16(m), value: v}
	t.set(optMode, unsafe.Pointer(&c))
}

// SetModeDefault sets a mode, and the value a reset restores.
func (t *Terminal) SetModeDefault(m Mode, v bool) {
	c := modeConfig{mode: uint16(m), value: v}
	t.set(optModeDefault, unsafe.Pointer(&c))
}

// MouseTracking reports whether a program asked for mouse events.
func (t *Terminal) MouseTracking() bool {
	var v bool
	return t.get(dataMouseTracking, unsafe.Pointer(&v)) == nil && v
}

// MouseShape returns the pointer shape a program asked for (OSC 22), a
// GhosttyMouseShape.
func (t *Terminal) MouseShape() int {
	var v int32
	t.get(dataMouseShape, unsafe.Pointer(&v))
	return int(v)
}

// AltScreen reports whether the alternate screen shows, as in full-screen
// programs.
func (t *Terminal) AltScreen() bool {
	var v int32
	return t.get(dataActiveScreen, unsafe.Pointer(&v)) == nil && v == 1
}

// Scrollbar returns where the viewport is in the scrollback.
func (t *Terminal) Scrollbar() Scrollbar {
	var s Scrollbar
	t.get(dataScrollbar, unsafe.Pointer(&s))
	return s
}

// AtBottom reports whether the viewport follows the active screen, rather
// than showing the scrollback.
func (t *Terminal) AtBottom() bool {
	var v bool
	return t.get(dataViewportActive, unsafe.Pointer(&v)) != nil || v
}

// ScrollToTop, ScrollToBottom, ScrollBy and ScrollToRow move the viewport:
// to the top of the scrollback, to the active screen, by rows (negative
// up), or to show row first.
func (t *Terminal) ScrollToTop()    { terminalScrollViewport(t.h, scrollViewport{tag: 0}) }
func (t *Terminal) ScrollToBottom() { terminalScrollViewport(t.h, scrollViewport{tag: 1}) }
func (t *Terminal) ScrollBy(rows int) {
	terminalScrollViewport(t.h, scrollViewport{tag: 2, value: [2]uint64{uint64(int64(rows))}})
}
func (t *Terminal) ScrollToRow(row int) {
	terminalScrollViewport(t.h, scrollViewport{tag: 3, value: [2]uint64{uint64(max(row, 0))}})
}

// CellAt returns the cell at column x of row y of the viewport.
func (t *Terminal) CellAt(x, y int) (GridRef, bool) {
	ref := GridRef{size: unsafe.Sizeof(GridRef{})}
	p := point{tag: 1, x: uint16(max(0, min(x, 0xFFFF))), y: uint32(max(y, 0))}
	return ref, terminalGridRef(t.h, p, &ref) == 0
}

// SetResizePullBack sets whether growing the screen pulls rows back from
// the scrollback, which a pseudo-terminal keeping its own screen, as
// Windows' ConPTY, cannot follow.
func (t *Terminal) SetResizePullBack(v bool) {
	t.set(optResizePullBack, unsafe.Pointer(&v))
}

// RowText returns the text of row y of the viewport, and the column of
// each of its runes.
func (t *Terminal) RowText(y int) (text []rune, cols []int) {
	c, _ := t.Size()
	var buf [16]uint32
	for x := range c {
		ref, found := t.CellAt(x, y)
		if !found {
			break
		}
		var raw uint64
		if ok(call(fnGridRefCell, uintptr(unsafe.Pointer(&ref)), uintptr(unsafe.Pointer(&raw)))) {
			if w := Wide(cell.wide.of(raw)); w == SpacerTail || w == SpacerHead {
				continue
			}
		}
		var n uintptr
		if !ok(call(fnGridRefGraphemes, uintptr(unsafe.Pointer(&ref)), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), uintptr(unsafe.Pointer(&n)))) || n == 0 {
			text, cols = append(text, ' '), append(cols, x)
			continue
		}
		for _, r := range buf[:n] {
			text, cols = append(text, rune(r)), append(cols, x)
		}
	}
	return text, cols
}

// Hyperlink returns the URI of the hyperlink (OSC 8) of a cell, or "".
func (t *Terminal) Hyperlink(ref GridRef) string {
	var buf [512]byte
	var n uintptr
	r := call(fnGridRefHyperlinkURI, uintptr(unsafe.Pointer(&ref)), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), uintptr(unsafe.Pointer(&n)))
	if Result(int32(r)) == OutOfSpace && n > 0 {
		big := make([]byte, n)
		r = call(fnGridRefHyperlinkURI, uintptr(unsafe.Pointer(&ref)), uintptr(unsafe.Pointer(&big[0])), n, uintptr(unsafe.Pointer(&n)))
		if !ok(r) {
			return ""
		}
		return string(big[:n])
	}
	if !ok(r) {
		return ""
	}
	return string(buf[:n])
}

// SetSelection selects s, or nothing when s is nil.
func (t *Terminal) SetSelection(s *Selection) {
	t.set(21, unsafe.Pointer(s))
}

// SelectAll returns a selection of everything, scrollback included.
func (t *Terminal) SelectAll() (Selection, bool) {
	s := Selection{size: unsafe.Sizeof(Selection{})}
	return s, ok(call(fnSelectAll, t.h, uintptr(unsafe.Pointer(&s))))
}

// SelectionText returns the text of the selection, with soft-wrapped lines
// joined and trailing spaces trimmed, as Ghostty copies it.
func (t *Terminal) SelectionText() (string, bool) {
	var p, n uintptr
	opts := selectionFormatOptions{size: unsafe.Sizeof(selectionFormatOptions{}), unwrap: true, trim: true}
	if selectionFormatAlloc(t.h, 0, opts, &p, &n) != 0 {
		return "", false
	}
	s := goString(p, n)
	if p != 0 {
		call(fnFree, 0, p, n)
	}
	return s, true
}

// Text returns the text of the screen and its scrollback, with
// soft-wrapped lines joined and trailing spaces trimmed.
func (t *Terminal) Text() string {
	opts := formatterOptions{size: unsafe.Sizeof(formatterOptions{}), unwrap: true, trim: true}
	opts.extra.size = unsafe.Sizeof(opts.extra)
	opts.extra.screen.size = unsafe.Sizeof(opts.extra.screen)
	var f uintptr
	if formatterTerminalNew(0, &f, t.h, opts) != 0 {
		return ""
	}
	defer call(fnFormatterFree, f)
	var p, n uintptr
	if !ok(call(fnFormatterFormatAlloc, f, 0, uintptr(unsafe.Pointer(&p)), uintptr(unsafe.Pointer(&n)))) {
		return ""
	}
	s := goString(p, n)
	if p != 0 {
		call(fnFree, 0, p, n)
	}
	return s
}

// formatVT is GHOSTTY_FORMATTER_FORMAT_VT: text with escape sequences.
const formatVT = 1

// VT returns the terminal's state as escape sequences, which restore it in
// a terminal of the same size: the scrollback and the screen with their
// styles and hyperlinks, then the modes, the scrolling region, the tab
// stops, the working directory, the keyboard's flags and the cursor. The
// palette is left out, as the terminal it restores in has colors of its
// own.
func (t *Terminal) VT() []byte {
	opts := formatterOptions{size: unsafe.Sizeof(formatterOptions{}), emit: formatVT}
	x := &opts.extra
	x.size = unsafe.Sizeof(opts.extra)
	x.modes, x.scrollingRegion, x.tabstops, x.pwd, x.keyboard = true, true, true, true, true
	x.screen.size = unsafe.Sizeof(opts.extra.screen)
	x.screen.cursor, x.screen.style, x.screen.hyperlink, x.screen.protection, x.screen.kittyKeyboard, x.screen.charsets = true, true, true, true, true, true
	var f uintptr
	if formatterTerminalNew(0, &f, t.h, opts) != 0 {
		return nil
	}
	defer call(fnFormatterFree, f)
	var p, n uintptr
	if !ok(call(fnFormatterFormatAlloc, f, 0, uintptr(unsafe.Pointer(&p)), uintptr(unsafe.Pointer(&n)))) {
		return nil
	}
	b := []byte(goString(p, n))
	if p != 0 {
		call(fnFree, 0, p, n)
	}
	return b
}

// Compress compresses some of the scrollback, as when the terminal is
// idle, and reports whether more is left to compress.
func (t *Terminal) Compress() bool {
	var r int32
	if !ok(call(fnTerminalCompress, t.h, 0, uintptr(unsafe.Pointer(&r)))) {
		return false
	}
	return r == 1
}

// Activity returns a token that changes when the scrollback may compress.
func (t *Terminal) Activity() uint64 {
	var v uint64
	call(fnTerminalCompressionActivity, t.h, uintptr(unsafe.Pointer(&v)))
	return v
}

func clamp16(v int) int { return max(1, min(v, 0xFFFF)) }

// The callbacks of every terminal, made once: purego's are never freed.
var (
	cbWritePTY, cbBell, cbTitleChanged, cbPwdChanged, cbRenderHold, cbClipboardWrite, cbClipboardRead, cbSize,
	cbColorScheme, cbDeviceAttributes, cbDesktopNotification, cbProgressReport, cbNoRead uintptr
)

func makeCallbacks() {
	// A GhosttyMimeReader for PasteEvent, which the library never calls
	// (it reads nothing for an event); it gives no data if it did.
	cbNoRead = newCallback(func(_, _, _, _, _ uintptr) uintptr { return 0 })
	cbWritePTY = newCallback(func(_, id, data, n uintptr) uintptr {
		if t := terminalOf(id); t != nil && t.effects.WritePTY != nil && n > 0 {
			p := unsafe.Slice((*byte)(mem(data)), n)
			if d := t.deferred; d != nil {
				// A deferred read: what the library writes now is its
				// refusal, which the read's answer replaces.
				t.deferred = nil
				if d.setRefusal(p) {
					return 0
				}
			}
			if t.silent && bytes.HasPrefix(p, []byte("\x1b]5522;type=read:status=EPERM")) {
				// The refusal of a read that goes unanswered: the library
				// writes it as the read's callback returns without a reply.
				t.silent = false
				return 0
			}
			if t.reply != nil {
				*t.reply = append(*t.reply, p...)
				return 0
			}
			t.effects.WritePTY(p)
		}
		return 0
	})
	cbClipboardRead = newCallback(func(_, id, req uintptr) uintptr {
		r := (*clipboardRead)(mem(req))
		t := terminalOf(id)
		if t == nil || t.effects.ClipboardRead == nil {
			return 0 // refused by the library
		}
		var in ClipboardRead
		in.List, in.Primary, in.Granted, in.CanRemember = r.list, r.location != 0, r.granted, r.canRemember
		for _, m := range unsafe.Slice((*cString)(mem(r.mimes)), r.mimesLen) {
			in.Mimes = append(in.Mimes, goString(m.ptr, m.len))
		}
		out, answer := t.effects.ClipboardRead(in)
		switch answer {
		case ClipboardAnswered:
			reply := clipboardReadReply{size: unsafe.Sizeof(clipboardReadReply{}), result: 1} // denied
			var contents []clipboardContent
			var available []cString
			var keep [][]byte
			for _, m := range in.Mimes {
				if data, ok := out.Contents[m]; ok && len(data) > 0 {
					contents = append(contents, clipboardContent{mime: cStringOf(m, &keep), data: cBytes(data)})
				}
			}
			for _, m := range out.Available {
				available = append(available, cStringOf(m, &keep))
			}
			if len(contents) > 0 || in.List && len(available) > 0 {
				reply.result = 0
				if len(contents) > 0 {
					reply.contents, reply.contentsLen = uintptr(unsafe.Pointer(&contents[0])), uintptr(len(contents))
				}
				if len(available) > 0 {
					reply.available, reply.availableLen = uintptr(unsafe.Pointer(&available[0])), uintptr(len(available))
				}
			}
			var written []byte
			t.reply = &written
			call(r.reply, req, uintptr(unsafe.Pointer(&reply)))
			t.reply = nil
			if len(written) > 0 && t.effects.WritePTY != nil {
				t.effects.WritePTY(written)
			}
			runtime.KeepAlive(keep)
			runtime.KeepAlive(out)
			runtime.KeepAlive(contents)
			runtime.KeepAlive(available)
		case ClipboardDeferred:
			d := out.Deferred
			if d != nil {
				d.mimes, d.list, d.primary = in.Mimes, in.List, in.Primary
				t.deferred = d
				return 0
			}
			fallthrough
		default: // ClipboardRefused: the library writes the refusal itself.
			t.silent = true
		}
		return 0
	})
	cbBell = newCallback(func(_, id uintptr) uintptr {
		if t := terminalOf(id); t != nil && t.effects.Bell != nil {
			t.effects.Bell()
		}
		return 0
	})
	cbTitleChanged = newCallback(func(_, id uintptr) uintptr {
		if t := terminalOf(id); t != nil && t.effects.TitleChanged != nil {
			t.effects.TitleChanged()
		}
		return 0
	})
	cbPwdChanged = newCallback(func(_, id uintptr) uintptr {
		if t := terminalOf(id); t != nil && t.effects.PwdChanged != nil {
			t.effects.PwdChanged()
		}
		return 0
	})
	cbRenderHold = newCallback(func(_, id, held uintptr) uintptr {
		if t := terminalOf(id); t != nil && t.effects.RenderHold != nil {
			t.effects.RenderHold(cbool(held))
		}
		return 0
	})
	cbClipboardWrite = newCallback(func(_, id, req uintptr) uintptr {
		w := (*clipboardWrite)(mem(req))
		reply := clipboardWriteReply{size: unsafe.Sizeof(clipboardWriteReply{}), result: 1} // denied
		t := terminalOf(id)
		if t != nil && t.effects.Clipboard != nil {
			// No contents clears the clipboard; otherwise the first plain
			// text, else the first text of another kind, is written.
			text, found := "", w.contentsLen == 0
			for _, c := range unsafe.Slice((*clipboardContent)(mem(w.contents)), w.contentsLen) {
				mime := goString(c.mime.ptr, c.mime.len)
				if strings.HasPrefix(mime, "text/plain") || strings.HasPrefix(mime, "text/") && !found {
					text, found = goString(c.data.ptr, c.data.len), true
				}
			}
			switch {
			case !found:
				reply.result = 2 // unsupported
			case t.effects.Clipboard(text, w.location != 0):
				reply.result = 0
			}
		}
		call(w.reply, req, uintptr(unsafe.Pointer(&reply)))
		return 0
	})
	cbSize = newCallback(func(_, id, out uintptr) uintptr {
		t := terminalOf(id)
		if t == nil || t.effects.Size == nil {
			return 0
		}
		cols, rows, cw, ch := t.effects.Size()
		*(*sizeReport)(mem(out)) = sizeReport{rows: uint16(clamp16(rows)), columns: uint16(clamp16(cols)), cellWidth: uint32(max(cw, 1)), cellHeight: uint32(max(ch, 1))}
		return 1
	})
	cbColorScheme = newCallback(func(_, id, out uintptr) uintptr {
		t := terminalOf(id)
		if t == nil || t.effects.Dark == nil {
			return 0
		}
		v := int32(0)
		if t.effects.Dark() {
			v = 1
		}
		*(*int32)(mem(out)) = v
		return 1
	})
	cbDeviceAttributes = newCallback(func(_, id, out uintptr) uintptr {
		// As Ghostty answers: a VT220 with ANSI colors (DA1), of type 1
		// and version 10 (DA2).
		da := (*deviceAttributes)(mem(out))
		*da = deviceAttributes{conformance: 62, numFeatures: 1, deviceType: 1, firmware: 10}
		da.features[0] = 22
		return 1
	})
	cbDesktopNotification = newCallback(func(_, id, req uintptr) uintptr {
		if t := terminalOf(id); t != nil && t.effects.Notify != nil {
			n := (*desktopNotification)(mem(req))
			t.effects.Notify(goString(n.title.ptr, n.title.len), goString(n.body.ptr, n.body.len))
		}
		return 0
	})
	cbProgressReport = newCallback(func(_, id, req uintptr) uintptr {
		if t := terminalOf(id); t != nil && t.effects.Progress != nil {
			p := (*progressReport)(mem(req))
			t.effects.Progress(int(p.state), int(p.progress))
		}
		return 0
	})
}

// cStringOf points a GhosttyString at a copy of s, kept in keep for as
// long as the library may read it.
func cStringOf(s string, keep *[][]byte) cString {
	if s == "" {
		return cString{}
	}
	b := []byte(s)
	*keep = append(*keep, b)
	return cBytes(b)
}

// cBytes points a GhosttyString at b, which must stay alive and unchanged
// while the library reads it.
func cBytes(b []byte) cString {
	if len(b) == 0 {
		return cString{}
	}
	return cString{ptr: uintptr(unsafe.Pointer(&b[0])), len: uintptr(len(b))}
}
