package terminal

import (
	"math"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/plugins/terminal/internal/vt"
	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"
)

// View shows the terminal in a window of native UI, and takes its keyboard
// input once it has the focus, which a click gives it. Size it like any
// element, as with Fill or Grow; its size sets the terminal's, in cells of
// the font. A terminal shows in one view at a time.
//
// A key goes, in order, to a shortcut of the window or an element around
// the view (and on macOS to the menus' key equivalents), to
// Options.OnKey, to the terminal's keybinds (Options.Keybinds, over
// DefaultKeybinds: Command+C and Command+V copy and paste on macOS, as do
// Control+Shift+C and Control+Shift+V elsewhere), and to the program.
// Other Command keys on macOS never reach the program, and go to the
// app's shortcuts. The Edit menu's roles copy, paste and select all too.
// The wheel scrolls the scrollback unless the program takes the mouse,
// and zooms with Control held (Command on macOS; Options.OnZoom). Hold
// Shift to select while a program takes the mouse, and Command (Control
// elsewhere) to open a link on click: a hyperlink (OSC 8), a URL printed as
// text, or a file path with Options.OpenLink.
func View(c *ui.Context, t *Terminal) ui.Element {
	v := t.viewOf(c)
	// A terminal's grid is left to right in a right-to-left interface too.
	e := ui.Box(c).Direction(ui.LeftToRight).Focusable().FocusRing(false).Clip().Label("Terminal")
	v.build(c, e)
	cursor := ui.CursorText
	if v.hoverLink {
		cursor = ui.CursorPointer
	}
	e.Cursor(cursor)
	e.HandleInput(v.input)
	e.TextCaret(v.caret())
	e.Draw(v.paint)
	e.ContextMenu(v.menu)
	return e
}

// viewOf returns the terminal's view state, made on the first frame
// showing it.
func (t *Terminal) viewOf(c *ui.Context) *view {
	if t.v == nil {
		t.v = &view{t: t, blinkStart: time.Now()}
		t.v.shaped.init(4096)
	}
	v := t.v
	if v.services != c.Services() {
		v.services = c.Services()
		draw := v.services.Invalidate
		t.draw.Store(&draw)
	}
	return v
}

// view is what a frame of the terminal needs from frame to frame. Main
// thread only.
type view struct {
	t        *Terminal
	services ui.Services
	// lang is the window's language, for the menu's items.
	lang string

	theme   *Theme
	applied *Theme // the theme the emulator has

	// The grid, in device pixels: the cells, the baseline in a cell, and
	// the origin of the grid relative to the element; scale is device
	// pixels per DIP.
	font                   fontKey
	fonts                  [4]ui.Font
	scale                  float32
	cellW, cellH, baseline int
	ox, oy                 int
	cols, rows             int

	colors    vt.Colors
	cursor    vt.Cursor
	scrollbar vt.Scrollbar
	lines     []rowCache
	shaped    lru

	keys    *vt.KeyEncoder
	mouse   *vt.MouseEncoder
	gesture *vt.Gesture

	// pending is a key that types text, until the text comes; preedit
	// is an input method's composition and its caret.
	pending      *pendingKey
	preedit      string
	preeditCaret int

	focused    bool
	blinkStart time.Time
	caretCell  [2]int // the cursor's cell, for input methods
	// selecting tells that the primary button selects; reporting that a
	// press went to the program, which gets the release too.
	selecting, reporting bool
	pointer              [2]float32
	// hoverLink: Command is held over a link, which a click opens;
	// linkPress: the primary button pressed one, and the press is the
	// link's until the release.
	hoverLink, linkPress bool
	// opening is the link a click asked to open, opened once t.mu is
	// released; openingPath tells it is a file path, which the system
	// does not open.
	opening     string
	openingPath bool
	// copying is the selection a release ended with Options.CopyOnSelect,
	// and rightClick what a secondary click does, done once t.mu is
	// released, as opening is.
	copying    string
	rightClick RightClickAction
	scrolled   float32 // scrolling not yet a whole row
	zoomed     float32 // wheel with the zoom modifier not yet a notch
	ticking    bool    // an autoscroll tick is due

	// reads are the program's reads of the clipboard waiting to be
	// answered; clipPrompt is the one the terminal's own prompt asks
	// about while clipOpen, one at a time.
	reads      []*clipboardRequest
	clipPrompt *clipboardRequest
	clipOpen   bool
}

type pendingKey struct {
	key       vt.Key
	unshifted rune
	mods      ui.Modifiers
	repeat    bool
	// taken: Options.OnKey took the key, whose text and release are
	// dropped.
	taken bool
}

// build updates the view as a frame builds.
func (v *view) build(c *ui.Context, e ui.Element) {
	t := v.t
	v.lang = c.Locale()
	dark := c.Theme().Dark
	focused := e.Focused()
	t.mu.Lock()
	theme := t.opts.Theme
	if dark && t.opts.DarkTheme != nil {
		theme = t.opts.DarkTheme
	} else if theme == nil {
		theme = defaultTheme(dark)
	}
	v.theme = theme
	if t.term == nil {
		v.release()
		v.reads, v.clipPrompt, v.clipOpen = nil, nil, false
	} else {
		if v.applied != theme {
			theme.apply(t.term)
			v.applied = theme
		}
		t.curTheme = theme
		if focused != v.focused && t.term.Mode(vt.ModeFocusEvent) {
			t.in.push(vt.EncodeFocus(focused))
		}
		// The hand over a link follows Command as it is pressed and let
		// go of, with the pointer still.
		link, _ := v.linkUnder(c.Modifiers(), v.pointer[0], v.pointer[1])
		v.hoverLink = link != ""
		// Input methods compose where the cursor is now, which the last
		// frame may not show yet.
		v.caretCell = [2]int{v.cursor.X, v.cursor.Y}
		if t.term.AtBottom() {
			x, y := t.term.Cursor()
			v.caretCell = [2]int{x, y}
		}
	}
	copied := t.clipOut
	t.clipOut = nil
	reads := t.clipReads
	t.clipReads = nil
	term := t.term
	t.mu.Unlock()
	for _, s := range copied {
		c.WriteClipboard(s)
	}
	if term != nil {
		v.reads = append(v.reads, reads...)
		v.answerReads(c, term)
	}
	if focused != v.focused {
		v.focused = focused
		v.blinkStart = c.Now()
		if !focused {
			// Input methods drop their composition with the focus.
			v.preedit, v.pending = "", nil
		}
	}
	// The cursor blinks, every 600 ms, while the terminal has the focus.
	if v.focused && v.cursor.Blinking && v.cursor.Visible {
		since := c.Now().Sub(v.blinkStart)
		c.After(blinkPeriod - since%blinkPeriod)
	}
	if v.ticking {
		c.After(40 * time.Millisecond)
	}
}

const blinkPeriod = 600 * time.Millisecond

// answerReads answers the program's reads of the clipboard as the frame
// builds, in order, since an OSC 52 reply doesn't tell which read it
// answers: reads allowed are served from the clipboard, which only the
// main thread reads; a read the app is asked about, through
// Options.OnClipboardRequest or the terminal's own prompt, holds back
// those after it.
func (v *view) answerReads(c *ui.Context, term *vt.Terminal) {
	for len(v.reads) > 0 && v.answerRead(c, term, v.reads[0]) {
		if v.clipPrompt == v.reads[0] {
			v.clipPrompt, v.clipOpen = nil, false
		}
		v.reads = slices.Delete(v.reads, 0, 1)
	}
}

// answerRead answers r if it can this frame, and reports whether it did.
func (v *view) answerRead(c *ui.Context, term *vt.Terminal, r *clipboardRequest) bool {
	t := v.t
	if a := v.readAnswer(r); a != 0 {
		v.finishRead(term, r, a == 1)
		return true
	}
	if !r.ask {
		v.finishRead(term, r, true)
		return true
	}
	if on := t.opts.OnClipboardRequest; on != nil {
		if !r.asked {
			r.text, r.asked = v.services.ReadClipboard(), true
			on(ClipboardRequest{
				Mimes:   slices.Clone(r.mimes),
				List:    r.list,
				Primary: r.primary,
				Text:    r.text,
				req:     r,
			})
			if a := v.readAnswer(r); a != 0 {
				v.finishRead(term, r, a == 1)
				return true
			}
		}
		return false
	}
	// The terminal's own prompt, modal across the frames until it is
	// answered.
	if v.clipPrompt != r {
		r.text = v.services.ReadClipboard()
		v.clipPrompt, v.clipOpen = r, true
	}
	tx := l10n.In(v.lang)
	switch ui.AlertDialog(c, &v.clipOpen, tx.ClipboardRead(), v.readMessage(r), tx.ClipboardReadDeny(), tx.ClipboardReadAllow()) {
	case -1:
		return false
	case 0, ui.AlertDismissed:
		v.finishRead(term, r, false)
	default:
		v.finishRead(term, r, true)
	}
	return true
}

// readAnswer returns 1 when the app allowed the read, 2 when it denied
// it, and 0 while the read waits; Allow and Deny may run on any
// goroutine, so t.mu guards it.
func (v *view) readAnswer(r *clipboardRequest) int8 {
	v.t.mu.Lock()
	defer v.t.mu.Unlock()
	return r.answer
}

// finishRead answers a read with what the clipboard holds of what the
// program asked for, read here on the main thread, or with its refusal.
func (v *view) finishRead(term *vt.Terminal, r *clipboardRequest, allow bool) {
	if term == nil {
		return
	}
	if !allow {
		v.answer(term, r, false, nil, nil)
		return
	}
	formats := []transfer.Format{transfer.Text}
	for _, mime := range r.mimes {
		if f := transfer.Format(mime); !slices.Contains(formats, f) {
			formats = append(formats, f)
		}
	}
	d := v.services.ReadClipboardData(formats...)
	contents := make(map[string][]byte, len(r.mimes)+1)
	for _, mime := range r.mimes {
		if b := readFormat(d, mime); len(b) > 0 {
			contents[mime] = b
		}
	}
	// The text too, whatever was asked for: an OSC 52 read asks for
	// text/plain alone, and the binding serves the protocol it came by.
	if b := readFormat(d, string(transfer.Text)); len(b) > 0 {
		if _, ok := contents["text/plain"]; !ok {
			contents["text/plain"] = b
		}
	}
	var available []string
	for _, f := range v.services.ClipboardFormats() {
		if m := offerMIME(f); m != "" && !slices.Contains(available, m) {
			available = append(available, m)
		}
	}
	v.answer(term, r, true, contents, available)
}

// answer writes the answer of a read under t.mu, as the reader goroutine
// writes what the program's output answers, so that the answer doesn't
// come between a paste's reply and the flag that says it is one
// (Terminal.replying). A terminal closed since says nothing.
func (v *view) answer(term *vt.Terminal, r *clipboardRequest, allow bool, contents map[string][]byte, available []string) {
	t := v.t
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.term == term && !t.closed {
		term.AnswerClipboard(r.deferred, allow, contents, available)
	}
}

// readMessage is what the terminal's prompt says a read asks for: the
// text it would read as one line, the types it asked for when the
// clipboard holds no text, or that the clipboard is empty.
func (v *view) readMessage(r *clipboardRequest) string {
	if r.text != "" {
		return previewText(r.text)
	}
	if len(r.mimes) > 0 {
		return strings.Join(r.mimes, " ")
	}
	return l10n.In(v.lang).ClipboardReadEmpty()
}

// previewText is text as one line of at most 200 runes.
func previewText(text string) string {
	one := strings.Map(func(r rune) rune {
		switch r {
		case '\n', '\r', '\t':
			return ' '
		}
		return r
	}, text)
	if r := []rune(one); len(r) > 200 {
		return string(r[:200]) + "…"
	}
	return one
}

// release frees what the view made of the library, once the terminal is
// closed.
func (v *view) release() {
	if v.keys != nil {
		v.keys.Free()
		v.keys = nil
	}
	if v.mouse != nil {
		v.mouse.Free()
		v.mouse = nil
	}
	if v.gesture != nil {
		v.gesture.Free(nil)
		v.gesture = nil
	}
}

// caret returns where input methods compose: the cursor's cell.
func (v *view) caret() ui.Rect {
	if v.scale == 0 {
		return ui.Rect{}
	}
	cw, ch := float32(v.cellW)/v.scale, float32(v.cellH)/v.scale
	x := float32(v.ox)/v.scale + float32(v.caretCell[0])*cw
	y := float32(v.oy)/v.scale + float32(v.caretCell[1])*ch
	return ui.Rect{X: x, Y: y, W: cw, H: ch}
}

// menu is the terminal's context menu.
func (v *view) menu(m *ui.Menu) {
	t := v.t
	t.mu.Lock()
	text, selected := "", false
	if t.term != nil {
		text, selected = t.term.SelectionText()
	}
	t.mu.Unlock()
	tx := l10n.In(v.lang)
	if v.menuItem(m, tx.Copy(), ActionCopy).Disabled(!selected || text == "").Chosen() {
		v.services.WriteClipboard(text)
	}
	if v.menuItem(m, tx.Paste(), ActionPaste).Chosen() {
		v.paste()
	}
	if v.menuItem(m, tx.SelectAll(), ActionSelectAll).Chosen() {
		v.selectAll()
	}
	m.Separator()
	if m.Item(tx.ClearScrollback()).Chosen() {
		t.Feed([]byte("\x1b[3J"))
	}
}

// menuItem adds an item of the menu, showing the key bound to its action.
func (v *view) menuItem(m *ui.Menu, label string, a Action) *ui.MenuItem {
	it := m.Item(label)
	if b, ok := v.t.shortcut(a); ok {
		it.Shortcut(b.Mods, b.Key)
	}
	return it
}

// input handles the input of the view as it comes.
func (v *view) input(ev ui.InputEvent) bool {
	switch ev.Kind {
	case ui.InputKeyDown:
		return v.keyDown(ev)
	case ui.InputKeyUp:
		return v.keyUp(ev)
	case ui.InputText:
		v.typed(ev.Text)
		return true
	case ui.InputCompose:
		v.preedit, v.preeditCaret, v.pending = ev.Text, ev.Caret, nil
		return true
	case ui.InputCommand:
		switch ev.Text {
		case "copy":
			v.copy(false)
		case "paste":
			v.paste()
		case "selectAll":
			v.selectAll()
		default:
			return false
		}
		return true
	case ui.InputPointerDown, ui.InputPointerUp, ui.InputPointerMove:
		took := v.pointerEvent(ev)
		if link := v.opening; link != "" {
			v.opening = ""
			if open := v.t.opts.OpenLink; (open == nil || !open(link)) && !v.openingPath {
				v.services.OpenURL(link)
			}
		}
		if text := v.copying; text != "" {
			// Outside pointerEvent's lock, which the clipboard may hold
			// up while another program has it.
			v.copying = ""
			v.services.WriteClipboard(text)
		}
		if a := v.rightClick; a != RightClickContextMenu {
			// Outside pointerEvent's lock, which copying and pasting take.
			v.rightClick = RightClickContextMenu
			switch a {
			case RightClickCopy:
				v.copy(true)
			case RightClickCopyOrPaste:
				if !v.copy(true) {
					v.paste()
				}
			case RightClickPaste:
				v.clearSelection()
				v.paste()
			}
		}
		return took
	case ui.InputScroll:
		return v.scroll(ev)
	}
	return false
}

func (v *view) keyDown(ev ui.InputEvent) bool {
	t := v.t
	if on := t.opts.OnKey; on != nil && on(ev.Mods, ev.Key) {
		k, _ := vtKey(ev.Key)
		v.pending = &pendingKey{key: k, taken: true}
		return true
	}
	mac := runtime.GOOS == "darwin"
	mods, key := ev.Mods, ev.Key
	if b, ok := t.binding(mods, key); ok {
		if v.perform(b.Action, b.Performable) || !b.Performable {
			// Its text and release are the binding's too.
			k, _ := vtKey(key)
			v.pending = &pendingKey{key: k, taken: true}
			return true
		}
	}
	if mods&ui.Super != 0 {
		return false // the app's shortcuts
	}
	k, unshifted := vtKey(key)
	if k == vt.KeyUnidentified {
		return false
	}
	altMeta := mods&ui.Alt != 0 && (!mac || t.opts.OptionAsAlt)
	// AltGr types text on Windows, where it is Control and Alt.
	altGr := runtime.GOOS == "windows" && mods&(ui.Ctrl|ui.Alt) == ui.Ctrl|ui.Alt
	if typesText(key) && (mods&(ui.Ctrl|ui.Alt) == 0 || mods&ui.Alt != 0 && !altMeta && mods&ui.Ctrl == 0 || altGr) {
		// The text it types comes next.
		v.pending = &pendingKey{key: k, unshifted: unshifted, mods: mods, repeat: ev.Repeat}
		return true
	}
	v.pending = nil
	text := ""
	if altMeta && typesText(key) && mods&ui.Ctrl == 0 {
		// Alt as Meta: the character, which the encoder prefixes with Esc.
		text = string(unshifted)
		if mods&ui.Shift != 0 {
			text = shiftedText(key, unshifted)
		}
	}
	action := vt.KeyPress
	if ev.Repeat {
		action = vt.KeyRepeat
	}
	v.sendKey(vt.KeyEvent{Action: action, Key: k, Mods: vtMods(mods), Text: text, Unshifted: unshifted})
	return true
}

// perform does what a key bound to an action does, and reports whether
// it did something (Keybind.Performable). A performable copy lets go of
// the selection, so that the key's next press reaches the program.
func (v *view) perform(a Action, performable bool) bool {
	switch a {
	case ActionIgnore:
		return true
	case ActionCopy:
		return v.copy(performable)
	case ActionPaste:
		return v.paste()
	case ActionSelectAll:
		v.selectAll()
		return true
	case ActionClearScreen:
		return v.clearScreen()
	case ActionIncreaseFontSize:
		v.zoom(Zoom{Steps: 1})
		return true
	case ActionDecreaseFontSize:
		v.zoom(Zoom{Steps: -1})
		return true
	case ActionResetFontSize:
		v.zoom(Zoom{Reset: true})
		return true
	case ActionScrollPageUp, ActionScrollPageDown, ActionScrollToTop, ActionScrollToBottom:
		return v.scrollBack(a)
	}
	return false
}

// zoom zooms, as OnZoom takes it or the terminal does.
func (v *view) zoom(z Zoom) {
	if on := v.t.opts.OnZoom; on != nil && on(z) {
		return
	}
	v.t.zoom(z)
}

// scrollBack scrolls the scrollback as an action does, but for
// full-screen programs, which have none, and reports whether it did.
func (v *view) scrollBack(a Action) bool {
	t := v.t
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.term == nil || t.term.AltScreen() {
		return false
	}
	_, rows := t.term.Size()
	page := max(rows-1, 1)
	switch a {
	case ActionScrollPageUp:
		t.term.ScrollBy(-page)
	case ActionScrollPageDown:
		t.term.ScrollBy(page)
	case ActionScrollToTop:
		t.term.ScrollToTop()
	case ActionScrollToBottom:
		t.term.ScrollToBottom()
	}
	v.services.Invalidate()
	return true
}

// clearScreen clears the scrollback and the screen above the cursor's
// line, which moves to the top, as Kitty's clear_terminal to_cursor: the
// prompt the shell shows stays. A full-screen program's screen, which this
// would break, is left alone, as in Ghostty; it reports whether it
// cleared.
func (v *view) clearScreen() bool {
	t := v.t
	t.mu.Lock()
	if t.term == nil || t.term.AltScreen() {
		t.mu.Unlock()
		return false
	}
	t.term.SetSelection(nil)
	t.term.ScrollToBottom()
	x, y := t.term.Cursor()
	t.mu.Unlock()
	seq := "\x1b[3J"
	if y > 0 {
		// Delete the rows above the cursor's from the top, then put the
		// cursor back on its line, now the first.
		seq += "\x1b[H\x1b[" + strconv.Itoa(y) + "M\x1b[1;" + strconv.Itoa(x+1) + "H"
	}
	t.Feed([]byte(seq))
	return true
}

func (v *view) keyUp(ev ui.InputEvent) bool {
	k, unshifted := vtKey(ev.Key)
	if p := v.pending; p != nil && p.taken && p.key == k {
		// The release of a key OnKey or a keybind took, Command held or
		// not.
		v.pending = nil
		return true
	}
	if k == vt.KeyUnidentified || ev.Mods&ui.Super != 0 {
		return false
	}
	if p := v.pending; p != nil && p.key == k {
		v.pending = nil
		// No text came, as for Control, Alt and a letter on Windows,
		// where they may type one: the key alone.
		action := vt.KeyPress
		if p.repeat {
			action = vt.KeyRepeat
		}
		text := ""
		if p.mods&ui.Ctrl == 0 {
			text = string(p.unshifted) // what the key types on a US keyboard
			if p.mods&ui.Shift != 0 {
				text = shiftedText(ev.Key, p.unshifted)
			}
		}
		v.sendKey(vt.KeyEvent{Action: action, Key: p.key, Mods: vtMods(p.mods), Text: text, Unshifted: p.unshifted})
	}
	return v.encodeKey(vt.KeyEvent{Action: vt.KeyRelease, Key: k, Mods: vtMods(ev.Mods), Unshifted: unshifted}, false)
}

// typed sends text typed or committed by an input method.
func (v *view) typed(text string) {
	v.preedit = ""
	p := v.pending
	v.pending = nil
	if p != nil && p.taken {
		return // the text of a key OnKey took
	}
	if p == nil {
		// Text without a key: of an input method, or every key typing
		// text on Linux.
		r, size := utf8.DecodeRuneInString(text)
		if size != len(text) || r == utf8.RuneError {
			v.sendText(text)
			return
		}
		k, unshifted, shift := keyOfRune(r)
		mods := ui.Modifiers(0)
		if shift {
			mods = ui.Shift
		}
		p = &pendingKey{key: k, unshifted: unshifted, mods: mods}
	}
	if utf8.RuneCountInString(text) > 1 {
		v.sendText(text)
		return
	}
	var consumed vt.Mods
	r, _ := utf8.DecodeRuneInString(text)
	if r != p.unshifted {
		// Shift or Option made the character.
		consumed = vtMods(p.mods & (ui.Shift | ui.Alt))
	}
	action := vt.KeyPress
	if p.repeat {
		action = vt.KeyRepeat
	}
	v.sendKey(vt.KeyEvent{Action: action, Key: p.key, Mods: vtMods(p.mods), Consumed: consumed, Text: text, Unshifted: p.unshifted})
}

// sendKey sends a key, scrolling to the bottom and clearing the selection
// as Ghostty does when it sends something.
func (v *view) sendKey(k vt.KeyEvent) {
	if v.encodeKey(k, true) {
		v.blinkStart = time.Now()
	}
}

func (v *view) encodeKey(k vt.KeyEvent, press bool) bool {
	t := v.t
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.term == nil {
		return false
	}
	if v.keys == nil {
		var err error
		if v.keys, err = vt.NewKeyEncoder(); err != nil {
			return false
		}
	}
	v.keys.Sync(t.term, t.opts.OptionAsAlt)
	b := v.keys.Encode(k)
	if len(b) == 0 {
		return false
	}
	t.in.push(append([]byte(nil), b...))
	if press {
		t.term.ScrollToBottom()
		t.term.SetSelection(nil)
	}
	return true
}

// sendText sends text that is no key, as of an input method.
func (v *view) sendText(text string) {
	t := v.t
	t.mu.Lock()
	if t.term != nil {
		t.term.ScrollToBottom()
		t.term.SetSelection(nil)
	}
	t.mu.Unlock()
	t.Send([]byte(text))
	v.blinkStart = time.Now()
}

// copy copies the selection, letting go of it with clear, and reports
// whether there was one.
func (v *view) copy(clear bool) bool {
	t := v.t
	t.mu.Lock()
	text, ok := "", false
	if t.term != nil {
		if text, ok = t.term.SelectionText(); ok && text != "" && clear {
			t.term.SetSelection(nil)
		}
	}
	t.mu.Unlock()
	if !ok || text == "" {
		return false
	}
	v.services.WriteClipboard(text)
	if clear {
		v.services.Invalidate()
	}
	return true
}

// paste pastes the clipboard and reports whether it held anything. It
// reads, on the main thread, as the paste starts, every format for
// Options.OnPaste, those the program can read for one that reads the
// clipboard itself (mode 5522), and else only the text, so that pasting
// into a shell converts no image for nothing. The program's read comes
// later, from the terminal's goroutine, where the clipboard cannot be
// read: it reads what was read here.
func (v *view) paste() bool {
	t := v.t
	held := v.services.ClipboardFormats()
	if len(held) == 0 {
		return false
	}
	var d transfer.Data
	switch {
	case t.opts.OnPaste != nil:
		d = v.services.ReadClipboardData()
		if t.opts.OnPaste(d) {
			return true
		}
	case t.readsClipboard():
		var formats []transfer.Format
		for _, f := range held {
			if offerMIME(f) != "" {
				formats = append(formats, f)
			}
		}
		d = v.services.ReadClipboardData(formats...)
	default:
		d = v.services.ReadClipboardData(transfer.Text)
	}
	t.pasteData(d, held)
	return true
}

// clearSelection lets go of the selection.
func (v *view) clearSelection() {
	t := v.t
	t.mu.Lock()
	if t.term != nil {
		t.term.SetSelection(nil)
	}
	t.mu.Unlock()
	v.services.Invalidate()
}

func (v *view) selectAll() {
	t := v.t
	t.mu.Lock()
	text := ""
	if t.term != nil {
		if s, ok := t.term.SelectAll(); ok {
			t.term.SetSelection(&s)
			if t.opts.CopyOnSelect == CopyOnSelectClipboard {
				text, _ = t.term.SelectionText()
			}
		}
	}
	t.mu.Unlock()
	if text != "" {
		// Outside the lock, which the clipboard may hold up while
		// another program has it.
		v.services.WriteClipboard(text)
	}
	v.services.Invalidate()
}

// cellAt returns the cell under a point of the element, in DIPs, within
// the grid.
func (v *view) cellAt(x, y float32) (col, row int) {
	if v.cellW == 0 {
		return 0, 0
	}
	col = int(math.Floor(float64((x*v.scale - float32(v.ox)) / float32(v.cellW))))
	row = int(math.Floor(float64((y*v.scale - float32(v.oy)) / float32(v.cellH))))
	return max(0, min(col, v.cols-1)), max(0, min(row, v.rows-1))
}

// pointerEvent reports the pointer to the program that takes the mouse, or
// selects.
func (v *view) pointerEvent(ev ui.InputEvent) bool {
	t := v.t
	v.pointer = [2]float32{ev.X, ev.Y}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.term == nil || v.cellW == 0 {
		return false
	}
	if v.linkPress {
		// The press opened a link: its moves and its release are no one
		// else's.
		v.linkPress = ev.Kind != ui.InputPointerUp
		return true
	}
	// Command over a link: a hand, and a click opens it, before a
	// program that takes the mouse sees it.
	link, path := v.linkUnder(ev.Mods, ev.X, ev.Y)
	if hover := link != ""; hover != v.hoverLink {
		v.hoverLink = hover
		v.services.Invalidate()
	}
	if link != "" && ev.Kind == ui.InputPointerDown && ev.Button == 0 {
		v.opening, v.openingPath, v.linkPress = link, path, true
		return true
	}
	tracking := t.term.MouseTracking() && ev.Mods&ui.Shift == 0
	if tracking && !v.selecting || v.reporting {
		return v.report(ev)
	}
	if v.gesture == nil {
		g, err := vt.NewGesture(uint64(500*time.Millisecond), 5*float64(v.scale))
		if err != nil {
			return false
		}
		v.gesture = g
	}
	col, row := v.cellAt(ev.X, ev.Y)
	px, py := float64(ev.X*v.scale), float64(ev.Y*v.scale)
	switch ev.Kind {
	case ui.InputPointerDown:
		if ev.Button == 1 && t.opts.RightClick != RightClickContextMenu {
			// input does it once the lock is released.
			v.rightClick = t.opts.RightClick
			return true
		}
		if ev.Button != 0 {
			return false // the context menu
		}
		ref, ok := t.term.CellAt(col, row)
		if !ok {
			return false
		}
		v.selecting = true
		if s, ok := v.gesture.Press(t.term, ref, px, py, uint64(time.Now().UnixNano())); ok {
			t.term.SetSelection(&s)
		} else {
			t.term.SetSelection(nil)
		}
		return true
	case ui.InputPointerMove:
		if !v.selecting {
			return false
		}
		ref, ok := t.term.CellAt(col, row)
		if !ok {
			return true
		}
		if s, ok := v.gesture.Drag(t.term, ref, px, py, v.geometry(), ev.Mods&ui.Alt != 0); ok {
			t.term.SetSelection(&s)
		}
		v.ticking = v.gesture.Autoscroll(t.term) != 0
		return true
	case ui.InputPointerUp:
		if !v.selecting || ev.Button != 0 {
			return false
		}
		v.selecting, v.ticking = false, false
		ref, ok := t.term.CellAt(col, row)
		if ok {
			v.gesture.Release(t.term, &ref)
		} else {
			v.gesture.Release(t.term, nil)
		}
		if t.opts.CopyOnSelect == CopyOnSelectClipboard {
			if text, ok := t.term.SelectionText(); ok {
				v.copying = text
			}
		}
		return true
	}
	return false
}

// linkUnder is the link a click with mods at (x, y) opens, and whether it
// is a file path: none without Command, or while the primary button
// selects or the program has a press; t.mu is held.
func (v *view) linkUnder(mods ui.Modifiers, x, y float32) (link string, path bool) {
	if mods&ui.Cmd == 0 || v.selecting || v.reporting || v.cellW == 0 {
		return "", false
	}
	return v.linkAt(v.cellAt(x, y))
}

// linkAt is the link at the cell: a hyperlink (OSC 8), a URL printed as
// text, or with Options.OpenLink a file path, or ""; t.mu is held.
func (v *view) linkAt(col, row int) (link string, path bool) {
	t := v.t
	if ref, ok := t.term.CellAt(col, row); ok {
		if url := t.term.Hyperlink(ref); url != "" {
			return url, false
		}
	}
	text, cols := t.term.RowText(row)
	if url := urlAt(text, cols, col); url != "" {
		return url, false
	}
	if t.opts.OpenLink != nil {
		if p := pathAt(text, cols, col); p != "" {
			return p, true
		}
	}
	return "", false
}

// geometry is the grid in pixels, for selection gestures.
func (v *view) geometry() vt.Geometry {
	return vt.Geometry{Columns: v.cols, CellWidth: v.cellW, PadLeft: v.ox, Height: v.oy + v.rows*v.cellH}
}

// autoscroll scrolls a selection dragged past the top or the bottom; t.mu
// is held.
func (v *view) autoscroll() {
	t := v.t
	dir := v.gesture.Autoscroll(t.term)
	if dir == 0 || !v.selecting {
		v.ticking = false
		return
	}
	t.term.ScrollBy(dir)
	col, row := v.cellAt(v.pointer[0], v.pointer[1])
	if s, ok := v.gesture.Tick(t.term, col, row, float64(v.pointer[0]*v.scale), float64(v.pointer[1]*v.scale), v.geometry(), false); ok {
		t.term.SetSelection(&s)
	}
}

// report reports the pointer to the program; t.mu is held.
func (v *view) report(ev ui.InputEvent) bool {
	t := v.t
	if v.mouse == nil {
		m, err := vt.NewMouseEncoder()
		if err != nil {
			return false
		}
		v.mouse = m
	}
	button := []int{vt.ButtonLeft, vt.ButtonRight, vt.ButtonMiddle}
	var action vt.MouseAction
	b := vt.ButtonNone
	switch ev.Kind {
	case ui.InputPointerDown:
		action = vt.MousePress
		v.reporting = true
	case ui.InputPointerUp:
		action = vt.MouseRelease
		v.reporting = false
	default:
		action = vt.MouseMotion
	}
	if ev.Button >= 0 && ev.Button < len(button) {
		b = button[ev.Button]
	}
	pressed := v.reporting
	v.mouse.Sync(t.term, v.ox*2+v.cols*v.cellW, v.oy*2+v.rows*v.cellH, v.cellW, v.cellH, v.ox, v.oy, pressed)
	out := v.mouse.Encode(action, b, vtMods(ev.Mods), ev.X*v.scale, ev.Y*v.scale)
	if len(out) > 0 {
		t.in.push(append([]byte(nil), out...))
	}
	return ev.Kind != ui.InputPointerMove || len(out) > 0
}

// scroll scrolls the scrollback, or reports the wheel to the program.
func (v *view) scroll(ev ui.InputEvent) bool {
	t := v.t
	if v.cellH == 0 {
		return false
	}
	if ev.Mods&ui.Cmd != 0 {
		// Rolling up is a step up. DY is the scroll offset's change, so up
		// comes negative; a wheel's event is a notch, a trackpad's many
		// small moves, which add up to a step by 40.
		if !ev.Precise {
			switch {
			case ev.DY < 0:
				v.zoom(Zoom{Steps: 1})
			case ev.DY > 0:
				v.zoom(Zoom{Steps: -1})
			}
			return true
		}
		v.zoomed -= ev.DY
		if steps := int(v.zoomed / 40); steps != 0 {
			v.zoomed -= float32(steps) * 40
			v.zoom(Zoom{Steps: steps})
		}
		return true
	}
	ch := float32(v.cellH) / v.scale
	dy := ev.DY
	if !ev.Precise {
		dy = dy / 40 * 3 * ch // a notch scrolls three rows
	}
	v.scrolled += dy
	rows := int(v.scrolled / ch)
	v.scrolled -= float32(rows) * ch
	if rows == 0 {
		return true
	}
	// OnScroll runs unlocked: it may call the terminal's methods, as Feed.
	if on := t.opts.OnScroll; on != nil && v.wheelScrollsBack(ev.Mods) && on(rows) {
		return true
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.term == nil {
		return false
	}
	switch {
	case v.wheelTracked(ev.Mods):
		if v.mouse == nil {
			m, err := vt.NewMouseEncoder()
			if err != nil {
				return false
			}
			v.mouse = m
		}
		v.mouse.Sync(t.term, v.ox*2+v.cols*v.cellW, v.oy*2+v.rows*v.cellH, v.cellW, v.cellH, v.ox, v.oy, false)
		button := vt.WheelDown
		if rows < 0 {
			button = vt.WheelUp
		}
		for range abs(rows) {
			if out := v.mouse.Encode(vt.MousePress, button, vtMods(ev.Mods), ev.X*v.scale, ev.Y*v.scale); len(out) > 0 {
				t.in.push(append([]byte(nil), out...))
			}
		}
	case v.wheelArrows():
		// Full-screen programs scroll with the arrows (mode 1007).
		if v.keys == nil {
			var err error
			if v.keys, err = vt.NewKeyEncoder(); err != nil {
				return false
			}
		}
		v.keys.Sync(t.term, false)
		key := vt.KeyArrowDown
		if rows < 0 {
			key = vt.KeyArrowUp
		}
		for range abs(rows) {
			t.in.push(append([]byte(nil), v.keys.Encode(vt.KeyEvent{Action: vt.KeyPress, Key: key})...))
		}
	default:
		t.term.ScrollBy(rows)
	}
	return true
}

// wheelTracked reports whether the program takes the wheel as the mouse:
// Shift keeps it for the terminal. t.mu held.
func (v *view) wheelTracked(mods ui.Modifiers) bool {
	return v.t.term.MouseTracking() && mods&ui.Shift == 0
}

// wheelArrows reports whether a full-screen program takes the wheel as
// arrow keys (alternate scroll, mode 1007). t.mu held.
func (v *view) wheelArrows() bool {
	return v.t.term.AltScreen() && v.t.term.Mode(vt.ModeAltScroll)
}

// wheelScrollsBack reports whether the wheel scrolls the scrollback, no
// program taking it, as Options.OnScroll may then.
func (v *view) wheelScrollsBack(mods ui.Modifiers) bool {
	t := v.t
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.term != nil && !v.wheelTracked(mods) && !v.wheelArrows()
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// shiftedText returns what a key types with Shift on a US keyboard.
func shiftedText(k ui.Key, unshifted rune) string {
	for r, key := range shifted {
		if key == k {
			return string(r)
		}
	}
	if 'a' <= unshifted && unshifted <= 'z' {
		return string(unshifted - 'a' + 'A')
	}
	return string(unshifted)
}
