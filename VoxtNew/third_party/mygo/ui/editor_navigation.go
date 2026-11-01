package ui

import (
	"runtime"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/text"
)

func (ed *editor) selection() (int, int) {
	if ed.visualSelection {
		if len(ed.selected) == 0 {
			return ed.caret, ed.caret
		}
		return ed.selected[0].Start, ed.selected[len(ed.selected)-1].End
	}
	if ed.caret < ed.anchor {
		return ed.caret, ed.anchor
	}
	return ed.anchor, ed.caret
}

func (ed *editor) selectAll() { ed.editorSelection = editorSelection{caret: ed.buf.n} }

// wants reports whether the editor handles a key, rather than letting it
// reach shortcuts.
func (ed *editor) wants(k keyEvent) bool {
	m := k.mods &^ Shift
	if ed.readOnly {
		// Selecting and copying; the arrows without Shift scroll.
		switch k.key {
		case KeyA, KeyC:
			return k.mods == Cmd
		case KeyLeft, KeyRight, KeyUp, KeyDown, KeyHome, KeyEnd:
			return k.mods&Shift != 0
		}
		return false
	}
	switch k.key {
	case KeyBackspace:
		// Empty, a token field's input leaves it to take out a token.
		return !ed.leaveEmptyBackspace || ed.buf.n > 0 || m != 0
	case KeyLeft, KeyRight:
		// But Alt and an arrow, which go back and forward outside of macOS,
		// where they move by words.
		return m != Alt || runtime.GOOS == "darwin"
	case KeyHome, KeyEnd, KeyDelete:
		return true
	case KeyUp, KeyDown, KeyPageUp, KeyPageDown:
		return ed.multiline || m != 0
	case KeyEnter:
		return m == 0
	case KeyTab, KeyEscape:
		return false
	case KeyF1, KeyF2, KeyF3, KeyF4, KeyF5, KeyF6, KeyF7, KeyF8, KeyF9, KeyF10, KeyF11, KeyF12, KeyBack, KeyForward:
		// They type nothing: they go to shortcuts, as F3 finding the next
		// match.
		return false
	case KeyA, KeyC, KeyX, KeyV, KeyZ, KeyY:
		if m == Cmd {
			return true
		}
	}
	if m == Ctrl && runtime.GOOS == "darwin" {
		if _, ok := emacsKeys[k.key]; ok || k.key == KeyA || k.key == KeyE || k.key == KeyK {
			return true
		}
	}
	// Printable keys type text, which comes as TextInput.
	return m == 0 || m == Alt && runtime.GOOS == "darwin"
}

func (ed *editor) move(to int, extend bool) {
	to = max(0, min(to, ed.buf.n))
	ed.caret = to
	ed.caretAffinity = text.Downstream
	ed.visualSelection, ed.selected = false, nil
	if !extend {
		ed.anchor = to
		ed.anchorAffinity = text.Downstream
	}
	ed.coalesce = false
	if ed.area != nil {
		ed.area.reveal = true
	}
}

// lineEdges returns the edges of the visual line holding rune i.
func (ed *editor) lineEdges(i int) (int, int) {
	if ed.area != nil {
		return ed.area.lineEdges(ed, i)
	}
	if ed.layout == nil || len(ed.layout.Lines) == 0 {
		return 0, ed.buf.n
	}
	li := ed.layout.LineAt(ed.displayIndex(i))
	line := ed.layout.Lines[li]
	return ed.textIndex(line.Start), ed.textIndex(line.End)
}

// displayIndex maps a rune of the text to the layout, which holds the
// composition at the caret.
func (ed *editor) displayIndex(i int) int {
	if ed.compose != "" && i > ed.caret {
		return i + utf8.RuneCountInString(ed.compose)
	}
	return i
}

func (ed *editor) textIndex(i int) int {
	if ed.compose != "" {
		n := utf8.RuneCountInString(ed.compose)
		switch {
		case i > ed.caret+n:
			return i - n
		case i > ed.caret:
			return ed.caret
		}
	}
	return min(i, ed.buf.n)
}

// vertical moves the caret up or down lines, keeping its x.
func (ed *editor) vertical(lines int, extend bool) {
	if a := ed.area; a != nil {
		x, y, h := a.caretAt(ed, ed.caret, 0)
		if !ed.hasDesired {
			ed.desiredX, ed.hasDesired = x, true
		}
		target := y + float64(h)/2 + float64(lines)*float64(h)
		switch {
		case target < 0:
			ed.move(0, extend)
		case target > a.hs.top(len(ed.buf.paras)):
			ed.move(ed.buf.n, extend)
		default:
			ed.movePosition(a.positionAt(ed, ed.desiredX, target), extend)
		}
		return
	}
	if ed.layout == nil {
		return
	}
	x, y, h := ed.layout.CaretAt(text.CaretPosition{Index: ed.displayIndex(ed.caret), Affinity: ed.caretAffinity})
	if !ed.hasDesired {
		ed.desiredX, ed.hasDesired = x, true
	}
	target := y + h/2 + float32(lines)*h
	if target < 0 {
		ed.move(0, extend)
		return
	}
	if target > ed.layout.Height {
		ed.move(ed.buf.n, extend)
		return
	}
	p := ed.layout.PositionAt(ed.desiredX, target)
	p.Index = ed.textIndex(p.Index)
	ed.movePosition(p, extend)
}

// emacsKeys are the Control keys of macOS text fields that act as other
// keys, after Emacs.
var emacsKeys = map[Key]Key{KeyB: KeyLeft, KeyF: KeyRight, KeyP: KeyUp, KeyN: KeyDown, KeyH: KeyBackspace, KeyD: KeyDelete}

func (ed *editor) key(c *context, st *state, k editEvent) {
	shift := k.mods&Shift != 0
	m := k.mods &^ Shift
	mac := runtime.GOOS == "darwin"
	if mac && m == Ctrl {
		if to, ok := emacsKeys[k.key]; ok {
			k.key, m = to, 0
		} else if ed.emacsKey(k.key, shift) {
			return
		}
	}
	word := (!mac && m == Ctrl) || (mac && m == Alt)
	b := &ed.buf
	a, z := ed.selection()
	switch k.key {
	case KeyLeft, KeyRight:
		left := k.key == KeyLeft
		// Words and line ends go the way the arrow points, as AppKit's
		// moveWordLeft: and moveToLeftEndOfLine: do: back in the text of
		// a left-to-right paragraph, forward in a right-to-left one.
		forward := !left
		if (word || mac && m == Super) && ed.caretParagraphRTL() {
			forward = left
		}
		switch {
		case mac && m == Super:
			s, e := ed.lineEdges(ed.caret)
			if !forward {
				ed.move(s, shift)
			} else {
				ed.move(e, shift)
			}
		case word && !forward:
			ed.move(b.prevWord(ed.caret), shift)
		case word:
			ed.move(b.nextWord(ed.caret), shift)
		case a != z && !shift && left:
			ed.movePosition(ed.collapseVisual(-1), false)
		case a != z && !shift:
			ed.movePosition(ed.collapseVisual(1), false)
		case left:
			ed.movePosition(ed.visualNext(-1), shift)
		default:
			ed.movePosition(ed.visualNext(1), shift)
		}
		ed.hasDesired = false
		return
	case KeyUp, KeyDown:
		if mac && m == Super || !ed.multiline {
			if k.key == KeyUp {
				ed.move(0, shift)
			} else {
				ed.move(b.n, shift)
			}
			return
		}
		d := 1
		if k.key == KeyUp {
			d = -1
		}
		ed.vertical(d, shift)
		return
	case KeyPageUp, KeyPageDown:
		n := max(int(st.h/max(ed.lineHeight(), 1))-1, 1)
		if k.key == KeyPageUp {
			n = -n
		}
		ed.vertical(n, shift)
		return
	case KeyHome, KeyEnd:
		if m == Ctrl || !ed.multiline {
			if k.key == KeyHome {
				ed.move(0, shift)
			} else {
				ed.move(b.n, shift)
			}
			return
		}
		s, e := ed.lineEdges(ed.caret)
		if k.key == KeyHome {
			ed.move(s, shift)
		} else {
			ed.move(e, shift)
		}
		return
	case KeyBackspace:
		switch {
		case a != z:
			ed.insert("")
		case mac && m == Super:
			s, _ := ed.lineEdges(ed.caret)
			ed.deleteRange(s, ed.caret)
		case word:
			ed.deleteRange(b.prevWord(ed.caret), ed.caret)
		case ed.caret > 0:
			ed.record(true)
			ed.replace(ed.graphemes.prev(b, ed.caret), ed.caret, "")
		}
		ed.hasDesired = false
		return
	case KeyDelete:
		switch {
		case a != z:
			ed.insert("")
		case word:
			ed.deleteRange(ed.caret, b.nextWord(ed.caret))
		case ed.caret < b.n:
			ed.deleteRange(ed.caret, ed.graphemes.next(b, ed.caret))
		}
		return
	case KeyEnter:
		if ed.multiline {
			ed.insert("\n")
		} else {
			st.markSubmitted()
			st.submitMods = k.mods
			c.rt.consumed = true
		}
		return
	}
	if m != Cmd {
		return
	}
	switch k.key {
	case KeyA:
		ed.selectAll()
	case KeyC:
		ed.command(c, "copy")
	case KeyX:
		ed.command(c, "cut")
	case KeyV:
		ed.command(c, "paste")
	case KeyZ:
		if shift {
			ed.command(c, "redo")
		} else {
			ed.command(c, "undo")
		}
	case KeyY:
		ed.command(c, "redo")
	}
}

// emacsKey performs Control-A, -E and -K of macOS text fields, which move
// to the start and end of the paragraph and delete to its end, and reports
// whether key was one of them.
func (ed *editor) emacsKey(key Key, shift bool) bool {
	p := ed.buf.para(ed.caret)
	start, end := ed.buf.start(p), ed.buf.end(p)
	switch key {
	case KeyA:
		ed.move(start, shift)
	case KeyE:
		ed.move(end, shift)
	case KeyK:
		if end == ed.caret && end < ed.buf.n {
			end++ // at the end, join the next paragraph
		}
		ed.anchor = ed.caret
		ed.deleteRange(ed.caret, end)
	default:
		return false
	}
	ed.hasDesired = false
	return true
}

func (ed *editor) command(c *context, name string) {
	a, b := ed.selection()
	h := c.rt.host
	if ed.readOnly && name != "copy" && name != "selectAll" {
		return
	}
	switch name {
	case "copy":
		if a != b && !ed.password {
			h.writeClipboard(ed.selectedText())
		}
	case "cut":
		if a != b && !ed.password {
			h.writeClipboard(ed.selectedText())
			ed.insert("")
		}
	case "paste", "pasteAndMatchStyle":
		s := h.readClipboard()
		if ed.onPaste != nil {
			// The frame builds again, showing what fn changed in the
			// elements built before the input, as an attachment.
			c.rt.consumed = true
			if ed.onPaste(s) {
				// Taken: the text and its selection stay, as when a web
				// page prevents a paste's default.
				break
			}
		}
		if s != "" {
			ed.insert(s)
		}
	case "selectAll":
		ed.selectAll()
	case "delete":
		if a != b {
			ed.insert("")
		}
	case "undo":
		ed.takeBack(false)
	case "redo":
		ed.takeBack(true)
	}
}

// press puts the caret where the pointer went down, at (x, y) relative to
// the element; double and triple clicks select words and lines.
func (ed *editor) press(x, y float32, clicks, button int) {
	if button != 0 || !ed.laidOut() {
		return
	}
	ed.commitCompose()
	p := ed.hitPosition(x, y)
	i := p.Index
	ed.dragging = true
	ed.hasDesired = false
	ed.dragUnit = min(clicks, 3)
	if ed.dragUnit != 1 {
		ed.visualSelection, ed.selected = false, nil
	}
	switch ed.dragUnit {
	case 1:
		ed.movePosition(p, ed.pressMods&Shift != 0)
	case 2:
		s, e := ed.buf.wordAt(i)
		ed.anchor, ed.caret = s, e
	default:
		if ed.multiline {
			s, e := ed.lineEdges(i)
			ed.anchor, ed.caret = s, e
		} else {
			ed.selectAll()
		}
	}
	ed.dragStart = [2]int{ed.anchor, ed.caret}
	if ed.area != nil {
		ed.area.reveal = true
	}
}

func (ed *editor) release() { ed.dragging = false }

// laidOut reports whether a frame laid the editor's text out.
func (ed *editor) laidOut() bool { return ed.layout != nil || ed.area != nil && ed.area.version != 0 }

// firstLine returns the first line of the text, as last laid out, or nil.
func (ed *editor) firstLine() *text.Line {
	switch {
	case ed.area != nil && ed.area.version != 0:
		return ed.area.firstLine(&ed.buf)
	case ed.layout != nil && len(ed.layout.Lines) > 0:
		return &ed.layout.Lines[0]
	}
	return nil
}

func (ed *editor) hit(x, y float32) int {
	return ed.hitPosition(x, y).Index
}

// drag extends the selection to the pointer.
func (ed *editor) drag(x, y float32) {
	p := ed.hitPosition(x, y)
	i := p.Index
	switch ed.dragUnit {
	case 2:
		s, e := ed.buf.wordAt(i)
		if i < ed.dragStart[0] {
			ed.anchor, ed.caret = ed.dragStart[1], s
		} else {
			ed.anchor, ed.caret = ed.dragStart[0], max(e, ed.dragStart[1])
		}
	case 3:
	default:
		ed.movePosition(p, true)
	}
	if ed.area != nil {
		ed.area.reveal = true
	}
}

// textKey asks onTextKey whether it takes k as it comes, with the input
// queued before it applied, so that fn reads the text typed before k.
func (ed *editor) textKey(k keyEvent) bool {
	if w := ed.client; w != nil {
		w.prepare()
	}
	ed.keySeq++
	ed.keying = true
	defer func() { ed.keying = false }()
	return ed.onTextKey(TextKey{Key: k.key, Mods: k.mods, ed: ed, seq: ed.keySeq})
}

// typesText reports whether k may type text rather than act: a character
// key with no modifier but Shift and Alt, which type characters, or a key
// the platform does not name.
func typesText(k keyEvent) bool {
	if k.mods&^(Shift|Alt) != 0 {
		return false
	}
	switch k.key {
	case KeyUnknown, KeySpace, KeyMinus, KeyEqual, KeyComma, KeyPeriod, KeySlash, KeySemicolon, KeyQuote,
		KeyBracketLeft, KeyBracketRight, KeyBackslash, KeyBackquote:
		return true
	}
	return k.key >= Key0 && k.key <= Key9 || k.key >= KeyA && k.key <= KeyZ
}
