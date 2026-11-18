package ui

import (
	"strings"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/internal/text"
)

// widgetTextInput is the built-in string control's native-input client.
// Preedit is a virtual insertion: the bound value contains committed text,
// while the platform sees the document with marked text at its absolute
// UTF-16 position. Queries copy only their requested text.
type widgetTextInput struct {
	ed         *editor
	rt         *engine
	id         uint64
	processing bool
}

var _ TextInputClient = (*widgetTextInput)(nil)

func (w *widgetTextInput) state() *state {
	s := w.rt.states[w.id]
	if s != nil && s.editor == w.ed {
		return s
	}
	return nil
}

func (w *widgetTextInput) publish() {
	ed := w.ed
	if ed.document != nil {
		if ed.buf.root != ed.published.root {
			if !ed.bufferDirty {
				ed.nativeBufferBefore = ed.published
			}
			snapshot, ok := ed.document.compareRestore(ed.published, ed.buf.root)
			if !ok {
				ed.loadBuffer(snapshot)
			} else {
				ed.published = snapshot
				ed.bufferDirty = true
			}
		}
		w.rt.blinkStart = w.rt.now()
		return
	}
	if ed.value != nil && *ed.value != ed.buf.s {
		if !ed.nativeDirty {
			ed.nativeValue = *ed.value
		}
		ed.nativeDirty = true
	}
	w.rt.blinkStart = w.rt.now()
}

// A navigation key may precede another native callback without a frame.
// Drain the widget's queue before queries too, so every callback sees the
// latest selection. Reentrant native queries must not process it twice.
func (w *widgetTextInput) prepare() {
	if w.ed.document != nil {
		w.ed.syncBuffer()
	}
	if w.processing || len(w.ed.queue) == 0 {
		return
	}
	if s := w.state(); s != nil {
		w.processing = true
		defer func() { w.processing = false }()
		w.ed.process(&w.rt.c, &node{st: s})
		w.publish()
	}
}

func (w *widgetTextInput) virtualRune(index int) int {
	ed := w.ed
	start, units := ed.buf.utf16At(ed.caret), countUnits(ed.compose)
	if index <= start || ed.compose == "" {
		return ed.buf.runeAtUTF16(index)
	}
	if index <= start+units {
		return ed.caret + utf8.RuneCountInString(ed.compose[:UTF16ByteOffset(ed.compose, index-start)])
	}
	return ed.buf.runeAtUTF16(index-units) + utf8.RuneCountInString(ed.compose)
}

func (w *widgetTextInput) virtualUnits(index int) int {
	ed := w.ed
	n := utf8.RuneCountInString(ed.compose)
	if index <= ed.caret || n == 0 {
		return ed.buf.utf16At(index)
	}
	if index <= ed.caret+n {
		return ed.buf.utf16At(ed.caret) + countUnits(ed.compose[:runeOffset(ed.compose, index-ed.caret)])
	}
	return ed.buf.utf16At(index-n) + countUnits(ed.compose)
}

func (w *widgetTextInput) TextForRange(r TextInputRange) (string, TextInputRange) {
	w.prepare()
	ed := w.ed
	r = platform.NormalizeTextRange(r)
	if ed.password {
		a, z := UTF16ByteOffset(ed.compose, r.Start), UTF16ByteOffset(ed.compose, r.End)
		return strings.Clone(ed.compose[a:z]), TextInputRange{Start: countUnits(ed.compose[:a]), End: countUnits(ed.compose[:z])}
	}
	a, z := w.virtualRune(r.Start), w.virtualRune(r.End)
	actual := TextInputRange{Start: w.virtualUnits(a), End: w.virtualUnits(z)}
	n := utf8.RuneCountInString(ed.compose)
	if n == 0 {
		if a == 0 && z == ed.buf.n {
			return nativeText(ed.buf.string()), actual
		}
		return nativeText(strings.Clone(ed.buf.slice(a, z))), actual
	}
	var out strings.Builder
	if a < ed.caret {
		out.WriteString(ed.buf.slice(a, min(z, ed.caret)))
	}
	if a < ed.caret+n && z > ed.caret {
		out.WriteString(ed.compose[runeOffset(ed.compose, max(0, a-ed.caret)):runeOffset(ed.compose, min(n, z-ed.caret))])
	}
	if z > ed.caret+n {
		out.WriteString(ed.buf.slice(max(a-n, ed.caret), z-n))
	}
	return nativeText(out.String()), actual
}

func nativeText(s string) string {
	if !utf8.ValidString(s) {
		return string([]rune(s))
	}
	return s
}

func (w *widgetTextInput) Selection() TextInputSelection {
	w.prepare()
	ed := w.ed
	if ed.compose != "" {
		start := ed.buf.utf16At(ed.caret)
		if ed.password {
			start = 0
		}
		return TextInputSelection{Range: TextInputRange{Start: start + ed.composeSelected.Start, End: start + ed.composeSelected.End}}
	}
	if ed.password {
		return TextInputSelection{}
	}
	a, z := ed.selection()
	if ranges := ed.selectedRanges(); len(ranges) > 1 {
		a, z = ed.caret, ed.caret
		for _, r := range ranges {
			if ed.caret >= r.Start && ed.caret <= r.End {
				a, z = r.Start, r.End
				break
			}
		}
	}
	return TextInputSelection{Range: TextInputRange{Start: ed.buf.utf16At(a), End: ed.buf.utf16At(z)}, Reversed: ed.caret == a && a != z}
}

func (w *widgetTextInput) MarkedRange() (TextInputRange, bool) {
	w.prepare()
	if w.ed.compose == "" {
		return TextInputRange{}, false
	}
	start := w.ed.buf.utf16At(w.ed.caret)
	if w.ed.password {
		start = 0
	}
	return TextInputRange{Start: start, End: start + countUnits(w.ed.compose)}, true
}

func (w *widgetTextInput) ReplaceText(r *TextInputRange, s string) {
	w.prepare()
	ed := w.ed
	if ed.readOnly {
		return
	}
	if r == nil || ed.password {
		if ed.compose != "" {
			ed.compose = ""
			ed.replace(ed.caret, ed.caret, s)
		} else {
			ed.insert(s)
		}
	} else {
		wanted := platform.NormalizeTextRange(*r)
		a, z := w.virtualRune(wanted.Start), w.virtualRune(wanted.End)
		transaction := ed.compositionActive
		ed.commitCompose()
		if !transaction {
			ed.record(false)
		}
		ed.replace(a, z, s)
	}
	ed.compose, ed.compositionActive = "", false
	ed.composeCaret, ed.composeSelected = 0, TextInputRange{}
	ed.hasDesired = false
	if ed.area != nil {
		ed.area.reveal = true
	}
	w.publish()
}

func (w *widgetTextInput) SetMarkedText(r *TextInputRange, s string, selected TextInputRange) {
	w.prepare()
	ed := w.ed
	if ed.readOnly {
		return
	}
	if ed.compose == "" || r != nil {
		a, z := ed.selection()
		ranges := ed.selectedRanges()
		if r != nil && !ed.password {
			wanted := platform.NormalizeTextRange(*r)
			a, z = w.virtualRune(wanted.Start), w.virtualRune(wanted.End)
		}
		transaction := ed.compositionActive && ed.compose != ""
		ed.commitCompose()
		if !transaction {
			ed.record(false)
		}
		ed.compositionActive = true
		if r == nil && len(ranges) > 1 {
			for i := len(ranges) - 1; i >= 0; i-- {
				ed.replace(ranges[i].Start, ranges[i].End, "")
			}
		} else {
			ed.replace(a, z, "")
		}
	}
	selected = platform.NormalizeTextRange(selected)
	n := countUnits(s)
	selected.Start, selected.End = min(selected.Start, n), min(selected.End, n)
	selected.Start = countUnits(s[:UTF16ByteOffset(s, selected.Start)])
	selected.End = countUnits(s[:UTF16ByteOffset(s, selected.End)])
	ed.compose, ed.composeSelected = s, selected
	ed.composeCaret = utf8.RuneCountInString(s[:UTF16ByteOffset(s, selected.End)])
	if ed.area != nil {
		ed.area.reveal = true
	}
	w.publish()
}

func (w *widgetTextInput) UnmarkText() {
	w.prepare()
	ed := w.ed
	transaction := ed.compositionActive
	ed.commitCompose()
	// GTK/IMM32 can end preedit before committing their replacement. The
	// platform remembers the range; both operations share this undo step.
	ed.compositionActive = transaction
	w.publish()
}

func (w *widgetTextInput) layout() *text.Layout {
	ed := w.ed
	if ed.layout != nil && (ed.layoutVersion != ed.buf.version || ed.layoutCompose != ed.compose || ed.layoutPassword != ed.password || ed.layoutGeneration != textSystem().Generation()) {
		params := ed.layout.Params
		params.Text = ed.displayText()
		params.Spans = ""
		if !ed.password {
			params.Spans, _ = ed.rangeSpans(0, ed.buf.n)
		}
		ed.shapeInput(params)
	}
	return ed.layout
}

func (w *widgetTextInput) BoundsForRange(r TextInputRange) (Rect, TextInputRange, bool) {
	w.prepare()
	ed, st := w.ed, w.state()
	if st == nil {
		return Rect{}, TextInputRange{}, false
	}
	r = platform.NormalizeTextRange(r)
	i := w.virtualRune(r.Start)
	if ed.password {
		i = ed.displayIndex(ed.caret) + ed.composeCaret
	}
	var l *text.Layout
	var y float32
	local := i
	if a := ed.area; a != nil {
		base := i
		n := utf8.RuneCountInString(ed.compose)
		if i > ed.caret {
			base = max(ed.caret, i-n)
		}
		p := ed.buf.para(base)
		l = a.paraLayout(ed, p)
		local = i - ed.buf.start(p)
		if ed.compose != "" && p > ed.buf.para(ed.caret) {
			local -= n
		}
		y = float32(a.hs.top(p) - a.scroll)
	} else {
		l = w.layout()
	}
	if l == nil {
		return Rect{}, TextInputRange{}, false
	}
	local = max(0, min(local, len(l.Runes)))
	affinity := text.Downstream
	if i == ed.displayIndex(ed.caret)+ed.composeCaret {
		affinity = ed.caretAffinity
	}
	x, ly, h := l.CaretAt(text.CaretPosition{Index: local, Affinity: affinity})
	b := Rect{X: ed.originX + x - ed.scrollX, Y: ed.originY + y + ly, W: 1, H: h}
	actual := TextInputRange{Start: w.virtualUnits(i), End: w.virtualUnits(i)}
	if r.End > r.Start && local < len(l.Runes) {
		var boundaries text.Boundaries
		boundaries.Reset(l.Runes)
		end := min(boundaries.NextGrapheme(local), local+w.virtualRune(r.End)-i)
		if rects := l.SelectionVisual(local, end, false); len(rects) > 0 {
			q := rects[0]
			b = Rect{X: ed.originX + q.X - ed.scrollX, Y: ed.originY + y + q.Y, W: q.W, H: q.H}
			actual.End = w.virtualUnits(i + end - local)
		}
	}
	visible := b.Y+b.H > 0 && b.Y < st.h && b.X+b.W > 0 && b.X < st.w
	if ed.password {
		n := countUnits(ed.compose)
		actual = TextInputRange{Start: min(r.Start, n), End: min(r.End, n)}
	}
	return b, actual, visible
}

func (w *widgetTextInput) IndexForPoint(p Point) (int, bool) {
	w.prepare()
	if !w.ed.laidOut() || w.ed.password {
		return 0, false
	}
	var position text.CaretPosition
	if a := w.ed.area; a != nil {
		position = a.documentPositionAt(w.ed, p.X-w.ed.originX, float64(p.Y-w.ed.originY)+a.scroll)
	} else {
		l := w.layout()
		position = l.PositionAt(p.X-w.ed.originX+w.ed.scrollX, p.Y-w.ed.originY)
	}
	return w.virtualUnits(position.Index), true
}
