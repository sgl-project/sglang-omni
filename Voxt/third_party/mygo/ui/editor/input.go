package editor

import (
	"math"
	"slices"
	"time"

	"github.com/egoist/mygo/ui"
)

// scrollbarDrag is a press on a scroll bar.
type scrollbarDrag struct {
	active     bool
	horizontal bool
	grab       float32 // where in the thumb the press caught it
}

const scrollbarWidth = 10

// input takes the editor's input as it comes.
func (ed *editor) input(ev ui.InputEvent) bool {
	// The key after the first of a chord is the chord's, as VS Code's
	// keybindings resolve chords before an extension's type command.
	chord := ed.chord != nil && ev.Kind == ui.InputKeyDown
	if !chord && ed.offer(ev) {
		return true
	}
	return ed.retry(func() bool { return ed.dispatch(ev) })
}

// offer gives the app's OnInput an event before the editor, once, and
// reports whether it took it. It is not retried with the editor's own
// handling: what the app does with an event it leaves, as edit the
// buffer, would happen twice. State's methods the app calls catch up
// with the buffer themselves.
func (ed *editor) offer(ev ui.InputEvent) bool {
	return ed.o.onInput != nil && ed.o.onInput(ev)
}

// retry runs fn, which edits the text as the editor shows it, again on
// the text as it is when another goroutine edited the buffer meanwhile,
// rather than leaving the edit out (apply).
func (ed *editor) retry(fn func() bool) bool {
	for range 4 {
		ed.sync()
		ed.stale = false
		ok := fn()
		if !ed.stale {
			return ok
		}
	}
	return false
}

func (ed *editor) dispatch(ev ui.InputEvent) bool {
	ed.frameNow = ev.Time
	switch ev.Kind {
	case ui.InputKeyDown:
		return ed.key(ev)
	case ui.InputCommand:
		cmd := map[string]Command{"copy": Copy, "cut": Cut, "paste": Paste, "pasteAndMatchStyle": Paste, "selectAll": SelectAll,
			"undo": Undo, "redo": Redo, "delete": Backspace}[ev.Text]
		if cmd != NoCommand {
			ed.run(cmd)
			return true
		}
		return false
	case ui.InputText:
		if !ed.o.readOnly {
			ed.insert(ev.Text)
			return true
		}
	case ui.InputPointerDown:
		return ed.pointerDown(ev)
	case ui.InputPointerMove:
		return ed.pointerMove(ev)
	case ui.InputPointerUp:
		if ed.gesture.active || ed.scrollbar.active {
			ed.gesture.active = false
			ed.scrollbar.active = false
			ed.endGroup()
			return true
		}
		return false
	case ui.InputScroll:
		return ed.wheel(ev)
	}
	return false
}

// key runs the command bound to a key.
func (ed *editor) key(ev ui.InputEvent) bool {
	km := keymap{app: ed.o.bindings, def: defaultCodeBindings()}
	mods := ev.Mods
	chord := ed.chord
	ed.chord = nil
	b, prefix := km.lookup(chord, mods, ev.Key)
	if chord != nil {
		// The second key of a chord, bound or not, is the chord's.
		if b != nil && b.Command != NoCommand {
			ed.run(b.Command)
		}
		return true
	}
	if prefix && b == nil {
		ed.chord = &Binding{Mods: mods, Key: ev.Key}
		return true
	}
	if b == nil || b.Command == NoCommand {
		return false
	}
	if ed.o.readOnly && commandEdits(b.Command) {
		return false
	}
	did := ed.run(b.Command)
	switch b.Command {
	case Cancel, Tab, Backtab, GoToDefinition:
		// Leave Escape, Tab and F12 to the window when they did nothing.
		return did
	}
	return true
}

// commandEdits reports whether a command edits the text.
func commandEdits(c Command) bool {
	switch c {
	case Undo, Redo, Cut, Paste, Backspace, Delete, DeleteWordLeft, DeleteWordRight,
		DeleteSubwordLeft, DeleteSubwordRight, DeleteToLineStart, DeleteToLineEnd, CutToLineEnd, DeleteLine,
		Newline, NewlineAbove, NewlineBelow, MoveLinesUp, MoveLinesDown, DuplicateLinesUp,
		DuplicateLinesDown, DuplicateSelection, JoinLines, Transpose, Indent, Outdent, Tab,
		Backtab, ToggleComment, ToggleBlockComment, SortLines, SortLinesCaseInsensitive, ReverseLines,
		UniqueLines, Uppercase, Lowercase, TitleCase, SnakeCase, KebabCase, CamelCase,
		PascalCase, TrimTrailingWhitespace, RevertHunk:
		return true
	}
	return false
}

// toContent maps a point of the element to the content.
func (ed *editor) toContent(x, y float32) (float32, float64) {
	return x - ed.textX + ed.scrollX, float64(y) + ed.scrollY
}

// region is what part of the editor a point of it is over.
type region uint8

const (
	regionText region = iota
	regionGutter
	regionVScroll
	regionHScroll
)

func (ed *editor) regionAt(x, y float32) region {
	vertical, horizontal := ed.scrollbars()
	if vertical && x >= ed.r.W-scrollbarWidth {
		return regionVScroll
	}
	if horizontal && y >= ed.viewH-scrollbarWidth {
		return regionHScroll
	}
	if x < ed.gutterW {
		return regionGutter
	}
	return regionText
}

func (ed *editor) pointerDown(ev ui.InputEvent) bool {
	ed.pointer, ed.pointerIn = ui.Point{X: ev.X, Y: ev.Y}, true
	ed.region = ed.regionAt(ev.X, ev.Y)
	ed.hover.close()
	if ev.Button != 0 {
		// The secondary button keeps a selection under it for the menu.
		if ev.Button == 1 {
			cx, cy := ed.toContent(ev.X, ev.Y)
			off := ed.offsetAt(cx, cy)
			for _, s := range ed.sels {
				if off >= s.start() && off <= s.end() {
					return false
				}
			}
			ed.setSels([]sel{ed.newSel(off, off)}, revealNone)
		}
		return false
	}
	switch ed.region {
	case regionVScroll:
		return ed.scrollbarPress(ev, false)
	case regionHScroll:
		return ed.scrollbarPress(ev, true)
	case regionGutter:
		return ed.gutterPress(ev)
	}
	cx, cy := ed.toContent(ev.X, ev.Y)
	// A block's own input, as a diff's buttons.
	if b, r, ok := ed.blockAt(cy); ok && ed.o.decor != nil {
		if ed.o.decor.blockInput(ed, b, ev, r) {
			return true
		}
	}
	// The fold placeholder unfolds.
	if line, ok := ed.foldPlaceholderAt(cx, cy); ok {
		ed.toggleFoldLine(line)
		return true
	}
	off := ed.offsetAt(cx, cy)
	mods := ev.Mods
	ed.endGroup()
	ed.wholeWord = false
	ed.gesture = gesture{active: true, origin: ed.newSel(off, off)}
	switch {
	case mods&ui.Alt != 0 && mods&ui.Shift != 0 && ev.Clicks == 1:
		ed.gesture.mode = 3
		ed.gesture.colStart = ui.Point{X: cx, Y: float32(cy)}
		ed.columnSelect(cx, cy)
		return true
	case mods == ui.Cmd && ed.o.onDefinition != nil && ev.Clicks == 1:
		ed.gesture.active = false
		ed.setSels([]sel{ed.newSel(off, off)}, revealNone)
		ed.o.onDefinition(off)
		return true
	case mods&ui.Alt != 0:
		ed.gesture.add = true
		ed.gesture.others = slices.Clone(ed.sels)
		// Alt-clicking a caret takes it away.
		if ev.Clicks == 1 && len(ed.sels) > 1 {
			for i, s := range ed.sels {
				if s.empty() && s.head == off {
					ed.sels = slices.Delete(slices.Clone(ed.sels), i, i+1)
					ed.gesture.active = false
					return true
				}
			}
		}
	case mods&ui.Shift != 0 && ev.Clicks == 1:
		p := ed.primary()
		ed.gesture.origin = sel{anchor: p.anchor, head: p.anchor, goal: -1, id: p.id}
		s := p
		s.head, s.goal = off, -1
		ed.setSels([]sel{s}, revealNone)
		return true
	}
	switch {
	case ev.Clicks == 2:
		ed.gesture.mode = 1
		a, z := wordAt(ed.view.text, off, ed.wordChars())
		ed.gesture.origin.anchor, ed.gesture.origin.head = a, z
	case ev.Clicks >= 3:
		ed.gesture.mode = 2
		r := &ed.view.text
		line := r.LineOf(off)
		ed.gesture.origin.anchor = r.LineStart(line)
		ed.gesture.origin.head = min(r.Len(), r.LineStart(line+1))
		if line+1 >= r.LineCount() {
			ed.gesture.origin.head = r.Len()
		}
	}
	ed.extendGesture(off)
	return true
}

// extendGesture extends the selection being dragged to off.
func (ed *editor) extendGesture(off int) {
	g := &ed.gesture
	o := g.origin
	s := sel{goal: -1, id: o.id}
	r := &ed.view.text
	switch g.mode {
	case 0:
		s.anchor, s.head = o.anchor, off
	case 1:
		a, z := wordAt(*r, off, ed.wordChars())
		if off < o.start() {
			s.anchor, s.head = o.end(), a
		} else {
			s.anchor, s.head = o.start(), max(z, o.end())
		}
	case 2:
		line := r.LineOf(off)
		if off < o.start() {
			s.anchor, s.head = o.end(), r.LineStart(line)
		} else {
			end := min(r.Len(), r.LineStart(line+1))
			if line+1 >= r.LineCount() {
				end = r.Len()
			}
			s.anchor, s.head = o.start(), max(end, o.end())
		}
	}
	sels := []sel{s}
	if g.add {
		sels = append(slices.Clone(g.others), s)
	}
	ed.sels = mergeSels(sels)
	ed.blinkStart = ed.frameNow
}

// columnSelect selects the columns from where the press began to x, y on
// each line between, as Zed's columnar selection: lines too short for
// the columns get no selection, but for the line of the press.
func (ed *editor) columnSelect(x float32, y float64) {
	v := &ed.view
	g := &ed.gesture
	x0, y0 := g.colStart.X, float64(g.colStart.Y)
	l0, l1 := v.heights.lineAt(y0), v.heights.lineAt(y)
	step := 1
	if l1 < l0 {
		step = -1
	}
	var sels []sel
	for line := l0; ; line += step {
		if !v.heights.isHidden(line) {
			l := v.line(line)
			r := &l.rows[0]
			a := l.start + l.fromDisp(v.dispAt(l, r, x0))
			b := l.start + l.fromDisp(v.dispAt(l, r, x))
			// Lines ending before the columns start are left out.
			if v.xOf(l, r, len(l.disp)) >= min(x0, x)-v.charWidth/2 || line == l0 {
				ed.nextID++
				sels = append(sels, sel{anchor: a, head: b, goal: x, id: ed.nextID})
			}
		}
		if line == l1 {
			break
		}
	}
	if len(sels) > 0 {
		ed.sels = mergeSels(sels)
	}
}

func (ed *editor) pointerMove(ev ui.InputEvent) bool {
	ed.pointer, ed.pointerIn = ui.Point{X: ev.X, Y: ev.Y}, ev.X >= 0 && ev.Y >= 0 && ev.X < ed.r.W && ev.Y < ed.viewH
	if ed.scrollbar.active {
		ed.dragScrollbar(ev)
		return true
	}
	if ed.gesture.active && ev.Button == 0 {
		cx, cy := ed.toContent(ev.X, ev.Y)
		if ed.gesture.mode == 3 {
			ed.columnSelect(cx, cy)
			return true
		}
		ed.extendGesture(ed.offsetAt(cx, cy))
		return true
	}
	// Hovering: the region, which the cursor shows, a line's gutter, and
	// resting on text for the hover.
	changed := false
	if r := ed.regionAt(ev.X, ev.Y); r != ed.region {
		ed.region = r
		changed = true
	}
	line := -1
	if ed.regionAt(ev.X, ev.Y) == regionGutter || ed.o.decor != nil {
		_, cy := ed.toContent(ev.X, ev.Y)
		line = ed.view.heights.lineAt(cy)
	}
	if line != ed.hoverLine {
		ed.hoverLine = line
		changed = true
	}
	h := &ed.hover
	off := -1
	if ed.regionAt(ev.X, ev.Y) == regionText {
		cx, cy := ed.toContent(ev.X, ev.Y)
		off = ed.offsetAt(cx, cy)
		// Past the end of a line, or below the last, is over no text.
		if cr, ok := ed.caretRect(off); !ok || float64(cy) > float64(cr.Y+cr.H) ||
			off == ed.view.text.LineEnd(ed.view.text.LineOf(off)) && cx > cr.X+ed.view.charWidth/2 {
			off = -1
		}
	}
	if off < 0 {
		// Off the text, nothing rests.
		h.pending, h.restOff = false, -1
	}
	// Off the text hovered, the hover goes after a grace, so that the
	// pointer can go into it, as the build sees (hideAt).
	if h.shown {
		if off >= 0 && off >= h.start && off <= h.end {
			h.hideAt = time.Time{}
		} else if h.hideAt.IsZero() {
			h.hideAt = ed.frameNow.Add(hoverHideDelay)
			changed = true
		}
	}
	if off >= 0 && (ed.o.onHover != nil || len(ed.diags) > 0) && off != h.restOff {
		h.restOff, h.restAt, h.pending = off, ed.frameNow, true
		changed = true
	}
	return changed
}

func (ed *editor) wheel(ev ui.InputEvent) bool {
	ed.hover.close()
	moved := false
	if ev.DY != 0 && !ed.o.fitContent {
		before := ed.scrollY
		ed.setScrollShared(ed.scrollY + float64(ev.DY))
		moved = ed.scrollY != before
	}
	if ev.DX != 0 && !ed.view.wrap {
		before := ed.scrollX
		ed.scrollX = max(0, min(ed.scrollX+ev.DX, ed.maxScrollX()))
		if s := ed.o.sharedScroll; s != nil {
			s.setX(ed.scrollX)
		}
		moved = moved || ed.scrollX != before
	}
	if moved && ed.gesture.active {
		cx, cy := ed.toContent(ed.pointer.X, ed.pointer.Y)
		ed.extendGesture(ed.offsetAt(cx, cy))
	}
	return moved
}

// scrollbars reports which scroll bars the editor shows, and takes
// presses on: the vertical one while it scrolls, as it does past the end
// of a short text, and the horizontal one while a line runs under the
// vertical one or beyond.
func (ed *editor) scrollbars() (vertical, horizontal bool) {
	if ed.o.noScrollbar || ed.o.fitContent {
		return false, false
	}
	return ed.scrollMax() > 0, !ed.view.wrap && ed.view.maxLine > ed.r.W-ed.textX-scrollbarWidth
}

func (ed *editor) maxScrollX() float32 {
	return max(0, ed.view.maxLine+ed.view.charWidth*4-(ed.r.W-ed.textX))
}

// scrollbarPress grabs a thumb, or jumps to where the track was pressed.
func (ed *editor) scrollbarPress(ev ui.InputEvent, horizontal bool) bool {
	t0, t1 := ed.thumb(horizontal)
	pos := ev.Y
	if horizontal {
		pos = ev.X - ed.textX
	}
	ed.scrollbar = scrollbarDrag{active: true, horizontal: horizontal}
	if pos >= t0 && pos <= t1 {
		ed.scrollbar.grab = pos - t0
	} else {
		ed.scrollbar.grab = (t1 - t0) / 2
		ed.dragScrollbar(ev)
	}
	return true
}

func (ed *editor) dragScrollbar(ev ui.InputEvent) {
	t0, t1 := ed.thumb(ed.scrollbar.horizontal)
	size := t1 - t0
	if ed.scrollbar.horizontal {
		track := ed.r.W - ed.textX - scrollbarWidth
		f := (ev.X - ed.textX - ed.scrollbar.grab) / max(1, track-size)
		ed.scrollX = max(0, min(f*ed.maxScrollX(), ed.maxScrollX()))
		if s := ed.o.sharedScroll; s != nil {
			s.setX(ed.scrollX)
		}
		return
	}
	track := ed.viewH
	f := float64(ev.Y-ed.scrollbar.grab) / float64(max(1, track-size))
	ed.setScrollShared(f * ed.scrollMax())
}

// setScrollShared scrolls, through the owner when it shares the scroll
// among editors.
func (ed *editor) setScrollShared(y float64) {
	if s := ed.o.sharedScroll; s != nil {
		s.set(y)
		return
	}
	ed.setScroll(y)
}

// thumb returns where the scroll bar's thumb is along its track.
func (ed *editor) thumb(horizontal bool) (float32, float32) {
	if horizontal {
		track := ed.r.W - ed.textX - scrollbarWidth
		content := ed.view.maxLine + ed.view.charWidth*4
		size := max(24, track*track/max(content, track))
		f := float32(0)
		if m := ed.maxScrollX(); m > 0 {
			f = ed.scrollX / m
		}
		return f * (track - size), f*(track-size) + size
	}
	track := ed.viewH
	content := float32(ed.scrollMax()) + ed.viewH
	size := max(24, track*track/max(content, track))
	f := float32(0)
	if m := ed.scrollMax(); m > 0 {
		f = float32(ed.scrollY / m)
	}
	return f * (track - size), f*(track-size) + size
}

// gutterPress folds at a fold marker, selects lines at a number, or
// calls OnGutterClick.
func (ed *editor) gutterPress(ev ui.InputEvent) bool {
	_, cy := ed.toContent(ev.X, ev.Y)
	line := ed.view.heights.lineAt(cy)
	g := ed.view.heights.geometry(line)
	if cy < g.rows || cy >= g.below {
		if b, r, ok := ed.blockAt(cy); ok && ed.o.decor != nil {
			return ed.o.decor.blockInput(ed, b, ev, r)
		}
		return false
	}
	if ed.o.decor != nil {
		if ed.o.decor.blockInput(ed, nil, ev, ui.Rect{Y: float32(g.rows - ed.scrollY), H: ed.view.lineHeight, W: ed.gutterW}) {
			return true
		}
	}
	if ed.o.folding && ed.o.decor == nil && ev.X >= ed.foldX() {
		if ed.foldable(line) || ed.isFolded(line) {
			ed.toggleFoldLine(line)
			return true
		}
	}
	if ed.o.onGutterClick != nil {
		ed.o.onGutterClick(line)
		return true
	}
	// Select the line, and drag to select more.
	r := &ed.view.text
	off := r.LineStart(line)
	ed.gesture = gesture{active: true, mode: 2, origin: ed.newSel(off, off)}
	ed.gesture.origin.anchor = off
	ed.gesture.origin.head = min(r.Len(), r.LineStart(line+1))
	if line+1 >= r.LineCount() {
		ed.gesture.origin.head = r.Len()
	}
	if ev.Mods&ui.Shift != 0 {
		p := ed.primary()
		ed.gesture.origin.anchor, ed.gesture.origin.head = p.anchor, p.anchor
		ed.gesture.mode = 2
	}
	ed.extendGesture(off)
	return true
}

// foldX is where the fold markers are in the gutter.
func (ed *editor) foldX() float32 { return ed.gutterW - ed.view.lineHeight*0.8 - 4 }

// blockAt returns the block at height y of the content, and its box in
// the element.
func (ed *editor) blockAt(y float64) (*block, ui.Rect, bool) {
	h := &ed.view.heights
	line := h.lineAt(y)
	g := h.geometry(line)
	above, below := h.blocksOf(line)
	top := g.top
	for i := range above {
		if y >= top && y < top+float64(above[i].height) {
			return &above[i], ui.Rect{X: 0, Y: float32(top - ed.scrollY), W: ed.r.W, H: above[i].height}, true
		}
		top += float64(above[i].height)
	}
	top = g.below
	for i := range below {
		if y >= top && y < top+float64(below[i].height) {
			return &below[i], ui.Rect{X: 0, Y: float32(top - ed.scrollY), W: ed.r.W, H: below[i].height}, true
		}
		top += float64(below[i].height)
	}
	return nil, ui.Rect{}, false
}

// foldPlaceholderAt reports whether x, y is on the placeholder of a fold,
// after its line's text.
func (ed *editor) foldPlaceholderAt(x float32, y float64) (int, bool) {
	v := &ed.view
	line := v.heights.lineAt(y)
	if !ed.isFolded(line) {
		return 0, false
	}
	l := v.line(line)
	last := &l.rows[len(l.rows)-1]
	end := last.indent + last.width
	g := v.heights.geometry(line)
	rowTop := g.rows + float64(len(l.rows)-1)*float64(v.lineHeight)
	if y >= rowTop && x >= end+4 && x <= end+4+v.charWidth*3+8 {
		return line, true
	}
	return 0, false
}

// The editor as the system's input methods see it: the newest selection,
// and a composition typed at every caret.

func (ed *editor) TextForRange(r ui.TextInputRange) (string, ui.TextInputRange) {
	ed.sync()
	t := &ed.view.text
	a, z := t.FromUTF16(min(r.Start, r.End)), t.FromUTF16(max(r.Start, r.End))
	return t.Slice(a, z), ui.TextInputRange{Start: t.UTF16(a), End: t.UTF16(z)}
}

func (ed *editor) Selection() ui.TextInputSelection {
	ed.sync()
	t := &ed.view.text
	p := ed.primary()
	return ui.TextInputSelection{Range: ui.TextInputRange{Start: t.UTF16(p.start()), End: t.UTF16(p.end())}, Reversed: p.head < p.anchor}
}

func (ed *editor) MarkedRange() (ui.TextInputRange, bool) {
	if !ed.markedActive || len(ed.marked) == 0 {
		return ui.TextInputRange{}, false
	}
	t := &ed.view.text
	i := min(newest(ed.sels), len(ed.marked)-1)
	m := ed.marked[i]
	return ui.TextInputRange{Start: t.UTF16(m.Start), End: t.UTF16(m.End)}, true
}

// targets returns the ranges input replaces: those marked, else the
// selections, or the one range the input method names.
func (ed *editor) targets(r *ui.TextInputRange) []ui.TextInputRange {
	t := &ed.view.text
	if r != nil {
		a, z := t.FromUTF16(min(r.Start, r.End)), t.FromUTF16(max(r.Start, r.End))
		p := ed.primary()
		mr, marked := ed.MarkedRange()
		if !(a == p.start() && z == p.end()) && !(marked && *r == mr) {
			return []ui.TextInputRange{{Start: a, End: z}}
		}
	}
	if ed.markedActive && len(ed.marked) > 0 {
		return slices.Clone(ed.marked)
	}
	out := make([]ui.TextInputRange, len(ed.sels))
	for i, s := range ed.sels {
		out[i] = ui.TextInputRange{Start: s.start(), End: s.end()}
	}
	return out
}

func (ed *editor) ReplaceText(r *ui.TextInputRange, s string) {
	// Text typed at the selection is input; committing a composition, or
	// replacing other text, as the accents of press and hold replace the
	// letter before the caret, is the input method's.
	if !ed.markedActive && ed.atSelection(r) && ed.offer(ui.InputEvent{Kind: ui.InputText, Text: s}) {
		return
	}
	ed.retry(func() bool { ed.replaceText(r, s); return true })
}

// atSelection reports whether an input method's range is the newest
// selection: nil, or the same range.
func (ed *editor) atSelection(r *ui.TextInputRange) bool {
	if r == nil {
		return true
	}
	ed.sync()
	t := &ed.view.text
	p := ed.primary()
	return t.FromUTF16(min(r.Start, r.End)) == p.start() && t.FromUTF16(max(r.Start, r.End)) == p.end()
}

func (ed *editor) replaceText(r *ui.TextInputRange, s string) {
	if ed.o.readOnly {
		return
	}
	if r == nil && !ed.markedActive {
		ed.insert(s)
		return
	}
	ts := ed.targets(r)
	ed.markedActive, ed.marked = false, nil
	ed.replaceRanges(ts, s, editCompose, -1, -1)
	if b := ed.buf(); b != nil {
		b.endGroup()
	}
	ed.complete.close()
	ed.afterTyping(s)
}

func (ed *editor) SetMarkedText(r *ui.TextInputRange, s string, selected ui.TextInputRange) {
	ed.retry(func() bool { ed.setMarkedText(r, s, selected); return true })
}

func (ed *editor) setMarkedText(r *ui.TextInputRange, s string, selected ui.TextInputRange) {
	if ed.o.readOnly {
		return
	}
	ts := ed.targets(r)
	// Selected counts UTF-16 units of s.
	sa, sz := utf16ByteOffset(s, selected.Start), utf16ByteOffset(s, selected.End)
	ed.replaceRanges(ts, s, editCompose, sa, sz)
	ed.markedActive = s != ""
	ed.marked = ed.marked[:0]
	switch {
	case s == "":
	case len(ts) == 1 && len(ed.sels) > 1:
		// The range named alone holds the composition.
		p := ed.primary()
		ed.marked = append(ed.marked, ui.TextInputRange{Start: p.start() - sa, End: p.start() - sa + len(s)})
	default:
		for _, sel := range ed.sels {
			ed.marked = append(ed.marked, ui.TextInputRange{Start: sel.start() - sa, End: sel.start() - sa + len(s)})
		}
	}
}

func utf16ByteOffset(s string, u int) int {
	return ui.UTF16ByteOffset(s, max(0, u))
}

// replaceRanges replaces each range with s, selecting sa to sz of it, or
// putting the caret after it.
func (ed *editor) replaceRanges(ts []ui.TextInputRange, s string, kind editKind, sa, sz int) {
	edits := make([]Edit, len(ts))
	for i, t := range ts {
		edits[i] = Edit{Start: t.Start, End: t.End, Text: s}
	}
	single := len(ts) == 1 && len(ed.sels) > 1
	sels := slices.Clone(ed.sels)
	newest := ed.primary().id
	ok := ed.apply(edits, kind, func(move func(int, bias) int) []sel {
		if single {
			// One range the input method named: the newest caret goes in
			// it, where the input method says, the others move with the
			// text.
			for i := range sels {
				sl := &sels[i]
				if sl.id != newest {
					sl.anchor, sl.head = move(sl.anchor, biasRight), move(sl.head, biasRight)
					continue
				}
				start := move(ts[0].Start, biasLeft)
				sl.anchor, sl.head, sl.goal = start+len(s), start+len(s), -1
				if sa >= 0 {
					sl.anchor, sl.head = start+sa, start+sz
				}
			}
			return sels
		}
		var out []sel
		delta := 0
		for i, t := range ts {
			start := t.Start + delta
			delta += len(s) - (t.End - t.Start)
			id := ed.nextID
			if i < len(sels) {
				id = sels[i].id
			}
			sel := sel{anchor: start + len(s), head: start + len(s), goal: -1, id: id}
			if sa >= 0 {
				sel.anchor, sel.head = start+sa, start+sz
			}
			out = append(out, sel)
		}
		return out
	})
	if !ok && s == "" {
		return
	}
}

func (ed *editor) UnmarkText() {
	ed.markedActive, ed.marked = false, nil
	if b := ed.buf(); b != nil {
		b.endGroup()
	}
}

func (ed *editor) BoundsForRange(r ui.TextInputRange) (ui.Rect, ui.TextInputRange, bool) {
	ed.sync()
	t := &ed.view.text
	a := t.FromUTF16(min(r.Start, r.End))
	cr, ok := ed.caretRect(a)
	if !ok {
		return ui.Rect{}, r, false
	}
	x := ed.textX + cr.X - ed.scrollX
	y := float32(float64(cr.Y) - ed.scrollY)
	if y+cr.H < 0 || y > ed.viewH {
		return ui.Rect{}, r, false
	}
	z := t.FromUTF16(max(r.Start, r.End))
	w := float32(1)
	if z > a {
		if cz, ok := ed.caretRect(z); ok && cz.Y == cr.Y {
			w = max(1, cz.X-cr.X)
		}
	}
	return ui.Rect{X: x, Y: y, W: w, H: cr.H}, r, true
}

func (ed *editor) IndexForPoint(p ui.Point) (int, bool) {
	ed.sync()
	cx, cy := ed.toContent(p.X, p.Y)
	off := ed.offsetAt(cx, cy)
	return ed.view.text.UTF16(off), true
}

// autoscroll scrolls while a drag goes past the top or the bottom, and
// reports whether it did.
func (ed *editor) autoscroll() bool {
	if !ed.gesture.active {
		return false
	}
	y := ed.pointer.Y
	var dy float32
	switch {
	case y < 0:
		dy = y
	case y > ed.viewH:
		dy = y - ed.viewH
	default:
		return false
	}
	speed := float32(math.Min(float64(abs(dy)), 300)) / 4
	if dy < 0 {
		speed = -speed
	}
	ed.setScrollShared(ed.scrollY + float64(speed))
	cx, cy := ed.toContent(ed.pointer.X, ed.pointer.Y)
	if ed.gesture.mode == 3 {
		ed.columnSelect(cx, cy)
	} else {
		ed.extendGesture(ed.offsetAt(cx, cy))
	}
	return true
}

func abs(f float32) float32 {
	if f < 0 {
		return -f
	}
	return f
}
