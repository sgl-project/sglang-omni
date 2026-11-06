package ui

import (
	"runtime"
	"strings"
)

// textPoint identifies a rune of a paragraph by its stable element ID.
type textPoint struct {
	id uint64
	at int
}

// textSelection is one selection across the paragraphs of a container.
// Each paragraph keeps its own layout and editor; syncTextSelection projects
// the shared endpoints onto those editors for painting.
type textSelection struct {
	scope         uint64
	anchor, caret textPoint
	start         [2]textPoint
	unit          int
	dragging      bool
	desiredX      float32
	hasDesired    bool
}

// prepareSelectable waits until inline children have finished their text,
// so rebuilding a paragraph does not reset its selection to its own text.
func (rt *engine) prepareSelectable(e *node) {
	e.prepareSelectable()
	for ch := e.first; ch != nil; ch = ch.next {
		rt.prepareSelectable(ch)
	}
}

func (e *node) prepareSelectable() {
	if e.kind != kindText {
		return
	}
	if e.isInline() {
		e.st.textScope = 0
		return
	}
	scope := uint64(0)
	if e.flags&(flagClickable|flagFocusable|flagUnselectable) == 0 && e.inputFn == nil {
		for p := e.parent; p != nil; p = p.parent {
			if p.kind == kindBox && p.flags&flagSelectable != 0 {
				scope = p.id
				break
			}
			if p.flags&(flagClickable|flagFocusable|flagEditable|flagUnselectable) != 0 || p.inputFn != nil {
				break // a control owns its text
			}
		}
	}
	if old := e.st.textScope; old != 0 && old != scope {
		if e.c.rt.selection.scope == old && e.c.rt.textSelectionIncludes(e.id) {
			e.c.rt.selection = textSelection{}
		}
		if ed := e.st.editor; ed != nil {
			ed.anchor, ed.caret = 0, 0
		}
	}
	e.st.textScope = scope
	if scope != 0 {
		e.flags |= flagSelectable
	}
	if e.flags&flagSelectable == 0 {
		return
	}
	ed := e.st.editor
	if ed == nil {
		ed = newEditor(e.c.rt.now)
		ed.readOnly, ed.multiline = true, true
		e.st.editor = ed
	}
	if ed.source != e.text || len(ed.buf.paras) == 0 {
		// Text only added at the end, as a reply being written, keeps
		// the selection and its offsets.
		grew := len(ed.buf.paras) > 0 && strings.HasPrefix(e.text, ed.source)
		if !grew && scope != 0 && e.c.rt.selection.scope == scope && e.c.rt.textSelectionIncludes(e.id) {
			e.c.rt.selection = textSelection{}
		}
		caret, anchor := ed.caret, ed.anchor
		ed.source = e.text
		ed.setText(e.text)
		ed.caret, ed.anchor = 0, 0
		if grew {
			ed.caret, ed.anchor = caret, anchor
		}
	}
	if e.c.rt.focused == e.id || len(ed.queue) > 0 {
		ed.process(e.c, e)
	}
}

// collectSelectable follows build order, independently of painting order
// (absolute children paint last). Inline children are already in their
// paragraph's text, and must not be copied a second time.
func (rt *engine) collectSelectable(e *node, hidden bool) {
	hidden = hidden || e.flags&(flagInvisible|flagInert) != 0
	if !hidden && e.kind == kindText && !e.isInline() && e.st.flags&flagDisabled == 0 && e.flags&flagSelectable != 0 {
		rt.texts = append(rt.texts, e.st)
	}
	for ch := e.first; ch != nil; ch = ch.next {
		rt.collectSelectable(ch, hidden)
	}
}

func (rt *engine) textPointIndex(p textPoint) int {
	for i, s := range rt.texts {
		if s.id == p.id && s.textScope == rt.selection.scope {
			return i
		}
	}
	return -1
}

func (rt *engine) compareTextPoints(a, b textPoint) int {
	i, j := rt.textPointIndex(a), rt.textPointIndex(b)
	if i != j {
		return i - j
	}
	return a.at - b.at
}

func (rt *engine) textSelectionBounds() (a, b textPoint, selected bool) {
	a, b = rt.selection.anchor, rt.selection.caret
	if rt.selection.scope == 0 || rt.textPointIndex(a) < 0 || rt.textPointIndex(b) < 0 {
		return textPoint{}, textPoint{}, false
	}
	d := rt.compareTextPoints(a, b)
	if d > 0 {
		a, b = b, a
	}
	return a, b, d != 0
}

func (rt *engine) textSelectionIncludes(id uint64) bool {
	a, b, _ := rt.textSelectionBounds()
	if a.id == 0 {
		return false
	}
	i := rt.textPointIndex(textPoint{id: id})
	return i >= 0 && i >= rt.textPointIndex(a) && i <= rt.textPointIndex(b)
}

func (rt *engine) textSelectionVisible(s *state) bool {
	if !rt.windowFocus() || s.flags&flagDisabled != 0 {
		return false
	}
	if s.textScope == 0 {
		return rt.focused == s.id
	}
	f := rt.states[rt.focused]
	return s.textScope == rt.selection.scope && f != nil && f.textScope == s.textScope && f.flags&flagDisabled == 0
}

func (rt *engine) syncTextSelection() {
	for _, s := range rt.texts {
		if s.textScope != 0 {
			s.editor.anchor, s.editor.caret = 0, 0
		}
	}
	if rt.selection.scope == 0 {
		return
	}
	if rt.textPointIndex(rt.selection.anchor) < 0 || rt.textPointIndex(rt.selection.caret) < 0 {
		// An endpoint was removed, hidden, disabled, or moved to another
		// container. Never let a recycled state inherit the selection.
		rt.selection = textSelection{}
		return
	}
	if rt.selection.dragging && rt.pressed != nil {
		rt.moveTextSelection(rt.pointerX, rt.pointerY)
	}
	a, b, _ := rt.textSelectionBounds()
	first, last := rt.textPointIndex(a), rt.textPointIndex(b)
	for i := first; i <= last; i++ {
		s := rt.texts[i]
		if s.textScope != rt.selection.scope {
			continue
		}
		from, to := 0, s.editor.buf.n
		if i == first {
			from = min(a.at, to)
		}
		if i == last {
			to = min(b.at, to)
		}
		s.editor.anchor, s.editor.caret = from, to
	}
}

// textSelectionAt finds the nearest visible paragraph in scope. Gaps,
// padding, and a pointer dragged outside the container still have an
// endpoint; the editor's hit test clamps it to that paragraph's text.
func (rt *engine) textSelectionAt(scope uint64, x, y float32) textPoint {
	// Respect the topmost hit, including a link inside a rich paragraph.
	for _, id := range rt.hitChain(x, y) {
		if s := rt.states[id]; s != nil && s.textScope == scope && s.editor != nil && s.flags&flagDisabled == 0 {
			return textHit(s, x, y)
		}
	}
	var nearest *state
	distance := inf
	for _, s := range rt.texts {
		if s.textScope != scope || s.vw <= 0 || s.vh <= 0 || !s.editor.laidOut() {
			continue
		}
		for _, line := range s.editor.layout.Lines {
			r := intersect(transformRect(s.world, Rect{s.x + s.editor.originX + line.X, s.y + s.editor.originY + line.Y, line.Width, line.Height}), Rect{s.vx, s.vy, s.vw, s.vh})
			if r.W <= 0 || r.H <= 0 {
				continue
			}
			dx, dy := max(r.X-x, x-r.X-r.W, 0), max(r.Y-y, y-r.Y-r.H, 0)
			if d := dx*dx + dy*dy; nearest == nil || d < distance {
				nearest, distance = s, d
			}
		}
	}
	if nearest == nil {
		return textPoint{}
	}
	return textHit(nearest, x, y)
}

func textHit(s *state, x, y float32) textPoint {
	ed := s.editor
	lx, ly := s.local(x, y)
	at := ed.hit(lx, ly)
	if ly < ed.originY {
		at = 0
	} else if ly >= ed.originY+ed.layout.Height {
		at = ed.buf.n
	}
	return textPoint{s.id, at}
}

// scrollTextSelection follows a drag near a scroll container's edges. The
// captured paragraph identifies the scroll container even outside its clip;
// after layout, syncTextSelection hit-tests the newly visible text.
func (rt *engine) scrollTextSelection() {
	if !rt.selection.dragging || rt.pressed == nil {
		return
	}
	const edge = 32
	for s := rt.pressed; s != nil; s = rt.states[s.parent] {
		if s.flags&flagScrollY == 0 {
			continue
		}
		var dy float32
		_, y := s.local(rt.pointerX, rt.pointerY)
		switch top, bottom := y, s.h-y; {
		case top < edge:
			dy = -min(edge-top, edge) / 2
		case bottom < edge:
			dy = min(edge-bottom, edge) / 2
		}
		if dy != 0 && scrollBy(s, 0, dy) {
			rt.animating = true
		}
		return
	}
}

func textUnit(s *state, at, unit int) (int, int) {
	switch unit {
	case 2:
		return s.editor.buf.wordAt(at)
	case 3:
		return s.editor.lineEdges(at)
	}
	return at, at
}

func (rt *engine) pressTextSelection(s *state, x, y float32, mods Modifiers, clicks int) bool {
	if s.flags&flagDisabled != 0 {
		return false
	}
	scope := s.textScope
	if s.editor == nil && s.flags&flagSelectable != 0 {
		scope = s.id
	}
	if scope == 0 {
		return false
	}
	for _, id := range rt.hitChain(x, y) {
		if id == scope {
			break
		}
		if p := rt.states[id]; p != nil && p.flags&flagUnselectable != 0 {
			return false
		}
	}
	p := rt.textSelectionAt(scope, x, y)
	if p.id == 0 {
		return false
	}
	t := &rt.selection
	if t.scope != scope || mods&Shift == 0 || rt.textPointIndex(t.anchor) < 0 {
		*t = textSelection{scope: scope, anchor: p, caret: p}
	}
	t.caret, t.unit, t.dragging, t.hasDesired = p, min(clicks, 3), true, false
	if t.unit > 1 {
		a, b := textUnit(rt.states[p.id], p.at, t.unit)
		t.anchor, t.caret = textPoint{p.id, a}, textPoint{p.id, b}
	}
	t.start = [2]textPoint{t.anchor, t.caret}
	// A press in padding starts in the nearest paragraph, which keeps the
	// focus and pointer capture until the release.
	s.pressed = false
	s = rt.states[p.id]
	rt.focused, rt.focusVisible = s.id, false
	rt.pressed = s
	s.pressed, s.pressMods = true, mods
	s.editor.dragging = false
	return true
}

func (rt *engine) moveTextSelection(x, y float32) {
	t := &rt.selection
	p := rt.textSelectionAt(t.scope, x, y)
	if p.id == 0 {
		return
	}
	if t.unit > 1 {
		a, b := textUnit(rt.states[p.id], p.at, t.unit)
		if rt.compareTextPoints(p, t.start[0]) < 0 {
			t.anchor, t.caret = t.start[1], textPoint{p.id, a}
		} else {
			t.anchor, t.caret = t.start[0], textPoint{p.id, b}
			if rt.compareTextPoints(t.caret, t.start[1]) < 0 {
				t.caret = t.start[1]
			}
		}
	} else {
		t.caret = p
	}
}

func (rt *engine) textSelectionCommand(s *state, name string) bool {
	if s.textScope == 0 {
		return false
	}
	if s.flags&flagDisabled != 0 {
		return true
	}
	switch name {
	case "selectAll":
		var first, last *state
		for _, p := range rt.texts {
			if p.textScope == s.textScope {
				if first == nil {
					first = p
				}
				last = p
			}
		}
		if first != nil {
			rt.selection = textSelection{scope: s.textScope, anchor: textPoint{first.id, 0}, caret: textPoint{last.id, last.editor.buf.n}}
		}
	case "copy":
		a, b, selected := rt.textSelectionBounds()
		if !selected || rt.selection.scope != s.textScope {
			break
		}
		if sc := rt.states[s.textScope]; sc != nil && sc.copyData != nil {
			rt.host.writeClipboardData(sc.copyData(rt.selectedTexts(s.textScope)))
			break
		}
		var out strings.Builder
		first, last := rt.textPointIndex(a), rt.textPointIndex(b)
		wrote := false
		for i := first; i <= last; i++ {
			p := rt.texts[i]
			if p.textScope != s.textScope {
				continue
			}
			from, to := 0, p.editor.buf.n
			if i == first {
				from = min(a.at, to)
			}
			if i == last {
				to = min(b.at, to)
			}
			if wrote {
				out.WriteByte('\n')
			}
			out.WriteString(p.editor.buf.slice(from, to))
			wrote = true
		}
		rt.host.writeClipboard(out.String())
	}
	return true // other editing commands cannot change selectable text
}

// selectedTexts returns the parts of paragraphs the selection of
// container scope holds.
func (rt *engine) selectedTexts(scope uint64) []SelectedText {
	a, b, selected := rt.textSelectionBounds()
	if !selected || rt.selection.scope != scope {
		return nil
	}
	var parts []SelectedText
	first, last := rt.textPointIndex(a), rt.textPointIndex(b)
	for i := first; i <= last; i++ {
		p := rt.texts[i]
		if p.textScope != scope || p.editor == nil {
			continue
		}
		from, to := 0, p.editor.buf.n
		if i == first {
			from = min(a.at, to)
		}
		if i == last {
			to = min(b.at, to)
		}
		parts = append(parts, SelectedText{ID: p.id, Text: p.editor.source, Start: from, End: to})
	}
	return parts
}

func (rt *engine) neighboringText(p textPoint, direction int) *state {
	for i := rt.textPointIndex(p) + direction; i >= 0 && i < len(rt.texts); i += direction {
		if s := rt.texts[i]; s.textScope == rt.selection.scope {
			return s
		}
	}
	return nil
}

func (rt *engine) textSelectionKey(s *state, k keyEvent) bool {
	if s.textScope == 0 {
		return false
	}
	if s.flags&flagDisabled != 0 {
		return true
	}
	if k.mods == Cmd && (k.key == KeyA || k.key == KeyC) {
		name := "copy"
		if k.key == KeyA {
			name = "selectAll"
		}
		return rt.textSelectionCommand(s, name)
	}
	t := &rt.selection
	if t.scope != s.textScope || rt.textPointIndex(t.caret) < 0 {
		p := textPoint{s.id, s.editor.caret}
		*t = textSelection{scope: s.textScope, anchor: p, caret: p}
	}
	s = rt.states[t.caret.id]
	ed, at := s.editor, t.caret.at
	mods := k.mods &^ Shift
	if (runtime.GOOS == "darwin" && mods == Cmd && (k.key == KeyUp || k.key == KeyDown)) ||
		(mods == Ctrl && (k.key == KeyHome || k.key == KeyEnd)) {
		direction := 1
		if k.key == KeyUp || k.key == KeyHome {
			direction = -1
		}
		for next := rt.neighboringText(textPoint{s.id, 0}, direction); next != nil; next = rt.neighboringText(textPoint{s.id, 0}, direction) {
			s = next
		}
		at = s.editor.buf.n
		if direction < 0 {
			at = 0
		}
		t.caret, t.hasDesired = textPoint{s.id, at}, false
		return true
	}
	if k.key == KeyUp || k.key == KeyDown {
		x, y, h := ed.layout.Caret(at)
		if !t.hasDesired {
			t.desiredX, t.hasDesired = s.x+ed.originX+x, true
		}
		direction := 1
		if k.key == KeyUp {
			direction = -1
		}
		y += h/2 + float32(direction)*h
		if y < 0 || y >= ed.layout.Height {
			if next := rt.neighboringText(t.caret, direction); next != nil {
				s, ed = next, next.editor
				y = 0
				if direction < 0 {
					y = ed.layout.Height - 1
				}
			} else {
				at = ed.buf.n
				if direction < 0 {
					at = 0
				}
				t.caret = textPoint{s.id, at}
				return true
			}
		}
		t.caret = textPoint{s.id, ed.layout.IndexAt(t.desiredX-s.x-ed.originX, y)}
		return true
	}
	t.hasDesired = false
	if (k.key == KeyLeft && at == 0 || k.key == KeyRight && at == ed.buf.n) && !(runtime.GOOS == "darwin" && mods == Cmd) {
		direction := 1
		if k.key == KeyLeft {
			direction = -1
		}
		if next := rt.neighboringText(t.caret, direction); next != nil {
			at = 0
			if direction < 0 {
				at = next.editor.buf.n
			}
			t.caret = textPoint{next.id, at}
			return true
		}
	}
	ed.caret = at
	ed.key(&rt.c, s, editEvent{kind: editKey, mods: k.mods, key: k.key})
	t.caret = textPoint{s.id, ed.caret}
	return true
}

func (rt *engine) textSelectionMenu(s *state, x, y float32) bool {
	if s.textScope == 0 {
		return false
	}
	p := textHit(s, x, y)
	a, b, selected := rt.textSelectionBounds()
	if !rt.textSelectionVisible(s) || !selected || rt.compareTextPoints(p, a) < 0 || rt.compareTextPoints(p, b) > 0 {
		rt.selection = textSelection{scope: s.textScope, anchor: p, caret: p}
	}
	rt.focused, rt.focusVisible = s.id, false
	return true
}
