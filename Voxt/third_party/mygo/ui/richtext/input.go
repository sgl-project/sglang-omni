package richtext

import (
	"runtime"
	"unicode"

	"github.com/egoist/mygo/internal/rope"
	"github.com/egoist/mygo/ui"
)

const scrollbarWidth = 10

// input takes the editor's input as it comes: keys, text, edit commands,
// the pointer and the wheel.
func (v *view) input(ev ui.InputEvent) bool {
	v.sync()
	defer v.publish()
	switch ev.Kind {
	case ui.InputKeyDown:
		if v.key(ev.Key, ev.Mods) {
			v.blinkStart = ev.Time
			return true
		}
		return false
	case ui.InputText:
		if v.opts.ReadOnly || ev.Text == "" {
			return false
		}
		v.compose = ""
		v.insert(ev.Text)
		v.typed(ev.Text)
		v.blinkStart = ev.Time
		return true
	case ui.InputCommand:
		return v.editCommand(ev.Text)
	case ui.InputPointerDown:
		return v.press(ev)
	case ui.InputPointerMove:
		return v.move(ev)
	case ui.InputPointerUp:
		return v.release(ev)
	case ui.InputSelect:
		n := v.shown.Len()
		v.selAnchor, v.caret = max(0, min(ev.Start, n)), max(0, min(ev.End, n))
		v.affinity = ui.TextDownstream
		v.reveal = true
		return true
	case ui.InputScroll:
		before := v.scroll
		v.setScroll(max(0, min(v.scroll+float64(ev.DY), v.hs.top(len(v.paras))-float64(v.viewH))))
		// At an end, the wheel goes on to the containers around.
		return v.scroll != before
	}
	return false
}

// moveTo puts the caret at i, extending the selection or not.
func (v *view) moveTo(i int, affinity ui.TextAffinity, extend bool) {
	i = max(0, min(i, v.shown.Len()))
	v.caret, v.affinity = i, affinity
	if !extend {
		v.selAnchor = i
	}
	v.coalesce = false
	v.reveal = true
}

// runeAt returns rune i of the text shown.
func (v *view) runeAt(i int) rune {
	r, _ := rope.TextOf(v.shown.text).RuneAt(rope.ByteOfRune(v.shown.text, i))
	return r
}

func isWord(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' }

// nextWord returns the end of the word after i; prevWord the start of the
// word before it.
func (v *view) nextWord(i int) int {
	n, word := v.shown.Len(), false
	for ; i < n; i++ {
		if isWord(v.runeAt(i)) {
			word = true
		} else if word {
			break
		}
	}
	return i
}

func (v *view) prevWord(i int) int {
	word := false
	for ; i > 0; i-- {
		if isWord(v.runeAt(i - 1)) {
			word = true
		} else if word {
			break
		}
	}
	return i
}

// wordAt returns the word, or the run of other runes, around i, within its
// paragraph.
func (v *view) wordAt(i int) (int, int) {
	n := v.shown.Len()
	if n == 0 {
		return 0, 0
	}
	i = max(0, min(i, n-1))
	word := isWord(v.runeAt(i))
	a, z := i, i
	for a > 0 {
		r := v.runeAt(a - 1)
		if r == '\n' || isWord(r) != word {
			break
		}
		a--
	}
	for z < n {
		r := v.runeAt(z)
		if r == '\n' || isWord(r) != word {
			break
		}
		z++
	}
	return a, z
}

// grapheme returns the edge of the grapheme before i (dir < 0) or after
// it: a newline is one.
func (v *view) grapheme(i, dir int) int {
	n := v.shown.Len()
	p := v.shown.paraAt(i)
	ps := v.shown.paraStart(p)
	pe := ps + v.para(p).n
	if dir < 0 && i == ps {
		return max(i-1, 0)
	}
	if dir > 0 && i >= pe {
		return min(i+1, n)
	}
	pv := v.paraLayout(p)
	u := pv.units[min(v.local(p, i), len(pv.units)-1)]
	if dir < 0 {
		u = pv.layout.PreviousBoundary(u)
	} else {
		u = pv.layout.NextBoundary(u)
	}
	return v.global(p, pv.runeOfUnit(u))
}

// visual returns the caret one grapheme left (dir < 0) or right on screen,
// across bidi runs, and to the next paragraph at the edge of one.
func (v *view) visual(dir int) (int, ui.TextAffinity) {
	p := v.shown.paraAt(v.caret)
	pv := v.paraLayout(p)
	l := min(v.local(p, v.caret), len(pv.units)-1)
	pos := ui.TextCaretPosition{Index: pv.units[l], Affinity: v.affinity}
	next := pv.layout.MoveCaret(pos, dir)
	if next.Index == pos.Index {
		if dir < 0 {
			return max(v.shown.paraStart(p)-1, 0), ui.TextDownstream
		}
		return min(v.shown.paraStart(p)+v.para(p).n+1, v.shown.Len()), ui.TextDownstream
	}
	return v.global(p, pv.runeOfUnit(next.Index)), next.Affinity
}

// vertical moves the caret up or down lines, keeping its x.
func (v *view) vertical(lines int, extend bool) {
	x, y, h := v.caretAt(v.caret, 0, v.affinity)
	if !v.hasDesired {
		v.desiredX, v.hasDesired = x, true
	}
	target := y + float64(h)/2 + float64(lines)*float64(h)
	switch {
	case target < 0:
		v.moveTo(0, ui.TextDownstream, extend)
	case target > v.hs.top(len(v.paras)):
		v.moveTo(v.shown.Len(), ui.TextDownstream, extend)
	default:
		i, a := v.positionAt(v.desiredX, target)
		v.moveTo(i, a, extend)
	}
	v.hasDesired = true
}

// emacsKeys are the Control keys of macOS text views that act as other
// keys.
var emacsKeys = map[ui.Key]ui.Key{ui.KeyB: ui.KeyLeft, ui.KeyF: ui.KeyRight, ui.KeyP: ui.KeyUp, ui.KeyN: ui.KeyDown, ui.KeyH: ui.KeyBackspace, ui.KeyD: ui.KeyDelete}

// key handles a key pressed, and reports whether it took it.
func (v *view) key(key ui.Key, mods ui.Modifiers) bool {
	if cmd, ok := shortcut(key, mods); ok {
		return v.run(cmd)
	}
	mac := runtime.GOOS == "darwin"
	shift := mods&ui.Shift != 0
	m := mods &^ ui.Shift
	if mac && m == ui.Ctrl {
		if to, ok := emacsKeys[key]; ok {
			key, m = to, 0
		} else {
			p := v.shown.paraAt(v.caret)
			ps := v.shown.paraStart(p)
			pe := ps + v.para(p).n
			switch key {
			case ui.KeyA:
				v.moveTo(ps, ui.TextDownstream, shift)
				return true
			case ui.KeyE:
				v.moveTo(pe, ui.TextDownstream, shift)
				return true
			case ui.KeyK:
				if v.opts.ReadOnly {
					return true
				}
				if pe == v.caret && pe < v.shown.Len() {
					pe++
				}
				v.deleteRange(v.caret, pe, false)
				return true
			}
		}
	}
	word := (!mac && m == ui.Ctrl) || (mac && m == ui.Alt)
	readOnly := v.opts.ReadOnly
	if key == ui.KeyV && shift && (m == ui.Cmd || m == ui.Cmd|ui.Alt) {
		// Paste and Match Style: ⇧⌥⌘V, and ⇧⌘V as Notes has it.
		return v.editCommand("pasteAndMatchStyle")
	}
	a, z := v.selection()
	n := v.shown.Len()
	switch key {
	case ui.KeyLeft, ui.KeyRight:
		left := key == ui.KeyLeft
		switch {
		case m == ui.Alt && !mac:
			return false // back and forward
		case mac && m == ui.Super:
			s, e := v.lineEdges(v.caret)
			if left {
				v.moveTo(s, ui.TextDownstream, shift)
			} else {
				v.moveTo(e, ui.TextUpstream, shift)
			}
		case word && left:
			v.moveTo(v.prevWord(v.caret), ui.TextDownstream, shift)
		case word:
			v.moveTo(v.nextWord(v.caret), ui.TextDownstream, shift)
		case a != z && !shift && left:
			v.moveTo(a, ui.TextDownstream, false)
		case a != z && !shift:
			v.moveTo(z, ui.TextDownstream, false)
		case left:
			i, aff := v.visual(-1)
			v.moveTo(i, aff, shift)
		default:
			i, aff := v.visual(1)
			v.moveTo(i, aff, shift)
		}
		v.hasDesired = false
		return true
	case ui.KeyUp, ui.KeyDown:
		if mac && m == ui.Super {
			if key == ui.KeyUp {
				v.moveTo(0, ui.TextDownstream, shift)
			} else {
				v.moveTo(n, ui.TextDownstream, shift)
			}
			return true
		}
		if mac && m == ui.Alt || !mac && m == ui.Ctrl {
			// By paragraphs: up to the start of this one or the one
			// before; down to the end of this one or the next on macOS,
			// as NSTextView, to the start of the next elsewhere, as
			// WordPad and GTK.
			a, z := v.shown.ParagraphRange(v.caret)
			switch {
			case key == ui.KeyUp && v.caret == a && a > 0:
				a, _ = v.shown.ParagraphRange(a - 1)
				v.moveTo(a, ui.TextDownstream, shift)
			case key == ui.KeyUp:
				v.moveTo(a, ui.TextDownstream, shift)
			case mac && v.caret == z && z < n:
				_, z = v.shown.ParagraphRange(z + 1)
				v.moveTo(z, ui.TextUpstream, shift)
			case mac:
				v.moveTo(z, ui.TextUpstream, shift)
			default:
				v.moveTo(min(z+1, n), ui.TextDownstream, shift)
			}
			v.hasDesired = false
			return true
		}
		if m != 0 {
			return false
		}
		d := 1
		if key == ui.KeyUp {
			d = -1
		}
		v.vertical(d, shift)
		return true
	case ui.KeyPageUp, ui.KeyPageDown:
		lines := max(int(v.viewH/(v.look.size*1.3))-1, 1)
		if key == ui.KeyPageUp {
			lines = -lines
		}
		v.vertical(lines, shift)
		return true
	case ui.KeyHome, ui.KeyEnd:
		if m == ui.Ctrl {
			if key == ui.KeyHome {
				v.moveTo(0, ui.TextDownstream, shift)
			} else {
				v.moveTo(n, ui.TextDownstream, shift)
			}
			return true
		}
		s, e := v.lineEdges(v.caret)
		if key == ui.KeyHome {
			v.moveTo(s, ui.TextDownstream, shift)
		} else {
			v.moveTo(e, ui.TextUpstream, shift)
		}
		return true
	case ui.KeyA:
		if m == ui.Cmd {
			v.selAnchor, v.caret = 0, n
			return true
		}
	case ui.KeyC, ui.KeyX, ui.KeyV, ui.KeyZ, ui.KeyY:
		if m != ui.Cmd {
			break
		}
		switch key {
		case ui.KeyC:
			return v.editCommand("copy")
		case ui.KeyX:
			return v.editCommand("cut")
		case ui.KeyV:
			return v.editCommand("paste")
		case ui.KeyZ:
			if shift {
				return v.editCommand("redo")
			}
			return v.editCommand("undo")
		case ui.KeyY:
			return v.editCommand("redo")
		}
	}
	if readOnly {
		return false
	}
	p := v.shown.paraAt(v.caret)
	st := v.para(p).style
	ps := v.shown.paraStart(p)
	switch key {
	case ui.KeyBackspace:
		switch {
		case a != z:
			v.deleteRange(a, z, false)
		case m == 0 && v.caret == ps && special(st):
			// At the start of a list's item, a heading, a quotation or a
			// block of code, Backspace takes the style away before joining.
			v.command(func(d *Document) { d.FormatParagraphs(unstyle) })
		case mac && m == ui.Super:
			s, _ := v.lineEdges(v.caret)
			v.deleteRange(s, v.caret, false)
		case word:
			v.deleteRange(v.prevWord(v.caret), v.caret, false)
		case v.caret > 0:
			v.deleteRange(v.grapheme(v.caret, -1), v.caret, true)
		}
		return true
	case ui.KeyDelete:
		switch {
		case a != z:
			v.deleteRange(a, z, false)
		case word:
			v.deleteRange(v.caret, v.nextWord(v.caret), false)
		case v.caret < n:
			v.deleteRange(v.caret, v.grapheme(v.caret, 1), false)
		}
		return true
	case ui.KeyEnter:
		if m != 0 {
			return false
		}
		v.enter()
		return true
	case ui.KeyTab:
		switch {
		case m != 0:
			return false
		case st.List != NoList:
			if shift {
				return v.run("outdent")
			}
			return v.run("indent")
		case st.CodeBlock && !shift:
			v.insert("\t")
			return true
		}
		return false
	}
	return false
}

// shortcut returns the command a key with mods runs: ⌘B, ⌘I, ⌘U, ⇧⌘X, ⌘E,
// ⌘K, ⇧⌘7, ⇧⌘8, ⇧⌘L, ⌘], ⌘[, ⌘{, ⌘|, ⌘} and ⌥⌘0–6 on macOS, as Pages,
// Notes and Google Docs have them, with Control in place of Command
// elsewhere, where Ctrl+L, Ctrl+E, Ctrl+R align, as in Word.
func shortcut(key ui.Key, mods ui.Modifiers) (string, bool) {
	mac := runtime.GOOS == "darwin"
	switch mods {
	case ui.Cmd:
		switch key {
		case ui.KeyB:
			return "bold", true
		case ui.KeyI:
			return "italic", true
		case ui.KeyU:
			return "underline", true
		case ui.KeyK:
			return "link", true
		case ui.KeyBracketRight:
			return "indent", true
		case ui.KeyBracketLeft:
			return "outdent", true
		case ui.KeyE:
			if mac {
				return "code", true
			}
			return "alignCenter", true
		case ui.KeyL:
			if !mac {
				return "alignLeft", true
			}
		case ui.KeyR:
			if !mac {
				return "alignRight", true
			}
		}
	case ui.Cmd | ui.Shift:
		switch key {
		case ui.KeyX:
			return "strikethrough", true
		case ui.Key7:
			return "numbers", true
		case ui.Key8:
			return "bullets", true
		case ui.KeyL:
			return "tasks", true
		case ui.KeyBracketLeft:
			if mac {
				return "alignLeft", true
			}
		case ui.KeyBracketRight:
			if mac {
				return "alignRight", true
			}
		case ui.KeyBackslash:
			if mac {
				return "alignCenter", true
			}
		case ui.KeyE:
			if !mac {
				return "code", true
			}
		}
	case ui.Cmd | ui.Alt:
		if key >= ui.Key0 && key <= ui.Key6 {
			return "heading" + string(rune('0'+int(key-ui.Key0))), true
		}
	}
	return "", false
}

// editCommand performs an edit command of a menu or a shortcut.
func (v *view) editCommand(name string) bool {
	a, z := v.selection()
	switch name {
	case "copy":
		if a != z {
			v.copySelection()
		}
		return true
	case "cut":
		if a != z && !v.opts.ReadOnly {
			v.copySelection()
			v.deleteRange(a, z, false)
		}
		return true
	case "paste", "pasteAndMatchStyle":
		if !v.opts.ReadOnly {
			v.paste(name == "pasteAndMatchStyle")
		}
		return true
	case "selectAll":
		v.selAnchor, v.caret = 0, v.shown.Len()
		return true
	case "delete":
		if a != z && !v.opts.ReadOnly {
			v.deleteRange(a, z, false)
		}
		return true
	case "undo", "redo":
		if !v.opts.ReadOnly {
			v.commitCompose()
			v.takeBack(name == "redo")
		}
		return true
	}
	return v.run(name)
}

// press handles a press of the pointer.
func (v *view) press(ev ui.InputEvent) bool {
	if ev.Button != 0 {
		return false
	}
	v.pressMods = ev.Mods
	v.pressLink = ""
	if v.overScrollbar(ev.X) {
		v.thumb = true
		top, size := v.thumbBox()
		if ev.Y >= top && ev.Y < top+size {
			v.thumbGrab = ev.Y - top
		} else {
			v.thumbGrab = size / 2
			v.dragThumb(ev.Y)
		}
		return true
	}
	v.commitCompose()
	x, y := v.contentPoint(ev.X, ev.Y)
	if p, ok := v.taskBox(x, y); ok && ev.Clicks == 1 && !v.opts.ReadOnly {
		start := v.shown.paraStart(p)
		v.command(func(d *Document) { d.ToggleChecked(start) })
		return true
	}
	i, aff := v.positionAt(x, y)
	if ev.Clicks == 1 && ev.Mods&ui.Shift == 0 && i < v.shown.Len() {
		v.pressLink = v.shown.AttributesAt(i).Link
	}
	v.dragging = true
	v.hasDesired = false
	v.dragUnit = min(max(ev.Clicks, 1), 3)
	switch v.dragUnit {
	case 1:
		v.moveTo(i, aff, ev.Mods&ui.Shift != 0)
	case 2:
		v.selAnchor, v.caret = v.wordAt(i)
	default:
		a, z := v.shown.ParagraphRange(i)
		v.selAnchor, v.caret = a, min(z+1, v.shown.Len())
	}
	v.dragStart = [2]int{v.selAnchor, v.caret}
	v.coalesce = false
	return true
}

// move handles the pointer moving: a drag extends the selection.
func (v *view) move(ev ui.InputEvent) bool {
	if v.thumb {
		v.dragThumb(ev.Y)
		return true
	}
	x, y := v.contentPoint(ev.X, ev.Y)
	if !v.dragging {
		cursor := ui.CursorText
		if _, box := v.taskBox(x, y); box || v.overScrollbar(ev.X) {
			cursor = ui.CursorDefault
		} else if i, _ := v.positionAt(x, y); i < v.shown.Len() && v.shown.AttributesAt(i).Link != "" {
			cursor = ui.CursorPointer
		}
		if cursor != v.cursor {
			v.cursor = cursor
			return true
		}
		return false
	}
	// Near the edges, a drag scrolls.
	if ev.Y < v.padY {
		v.setScroll(max(0, v.scroll-float64(v.padY-ev.Y)))
	} else if ev.Y > v.padY+v.viewH {
		v.setScroll(v.scroll + float64(ev.Y-v.padY-v.viewH))
	}
	i, aff := v.positionAt(x, y)
	switch v.dragUnit {
	case 2:
		s, e := v.wordAt(i)
		if i < v.dragStart[0] {
			v.selAnchor, v.caret = v.dragStart[1], s
		} else {
			v.selAnchor, v.caret = v.dragStart[0], max(e, v.dragStart[1])
		}
	case 3:
		a, z := v.shown.ParagraphRange(i)
		if i < v.dragStart[0] {
			v.selAnchor, v.caret = v.dragStart[1], a
		} else {
			v.selAnchor, v.caret = v.dragStart[0], min(z+1, v.shown.Len())
		}
	default:
		v.caret, v.affinity = i, aff
	}
	if i != v.dragStart[0] || v.dragUnit != 1 {
		v.pressLink = ""
	}
	v.reveal = true
	return true
}

// release ends a drag, and opens the link a click began on.
func (v *view) release(ev ui.InputEvent) bool {
	if v.thumb {
		v.thumb = false
		return true
	}
	if !v.dragging {
		return false
	}
	v.dragging = false
	link := v.pressLink
	v.pressLink = ""
	if link != "" && v.caret == v.selAnchor {
		if v.opts.OnLink != nil {
			v.opts.OnLink(link)
		} else if ui.SafeToOpen(link) {
			v.services.OpenURL(link)
		}
	}
	return true
}

// taskBox reports whether (x, y), in the content, is on the check box of
// a task, and which paragraph's.
func (v *view) taskBox(x float32, y float64) (int, bool) {
	if len(v.paras) == 0 {
		return 0, false
	}
	p := v.hs.at(y)
	st := v.para(p).style
	if st.List != Tasks {
		return 0, false
	}
	left, _ := v.insets(st)
	if x < left-v.look.indent || x >= left-2 {
		return p, false
	}
	// Beside the first line, where the box is.
	_, top, h := v.caretAt(v.shown.paraStart(p), 0, ui.TextDownstream)
	return p, y >= top && y < top+float64(h)
}

// overScrollbar reports whether x of the element is on the scroll bar.
func (v *view) overScrollbar(x float32) bool {
	return v.hs.top(len(v.paras)) > float64(v.viewH) && x >= v.rect.W-scrollbarWidth
}

// thumbBox returns the top and the height of the scroll bar's thumb, in
// the element.
func (v *view) thumbBox() (top, size float32) {
	total := v.hs.top(len(v.paras))
	track := v.rect.H - 4
	size = max(24, track*float32(float64(v.viewH)/max(total, 1)))
	room := float64(track - size)
	at := 0.0
	if max := total - float64(v.viewH); max > 0 {
		at = v.scroll / max
	}
	return 2 + float32(room*at), size
}

// dragThumb scrolls as the thumb is dragged to y.
func (v *view) dragThumb(y float32) {
	_, size := v.thumbBox()
	track := v.rect.H - 4 - size
	total := v.hs.top(len(v.paras)) - float64(v.viewH)
	if track <= 0 || total <= 0 {
		return
	}
	f := float64((y - v.thumbGrab - 2) / track)
	v.setScroll(max(0, min(f, 1)) * total)
}

// menu builds the editor's context menu.
func (v *view) menu(m *ui.Menu) {
	m.EditItems()
}
