package richtext

import (
	"strings"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/rope"
	"github.com/egoist/mygo/ui"
)

// The editor is the client of the system's input methods: they read the
// text around the caret and compose there, in UTF-16 units of the text
// with the composition, which is a virtual insertion at the caret until it
// is committed, as one step of undo with its start.

var _ ui.TextInputClient = (*view)(nil)

// units returns the UTF-16 offset of rune i of the text shown.
func (v *view) units(i int) int {
	return rope.Prefix(v.shown.text, rope.ByteOfRune(v.shown.text, max(0, min(i, v.shown.Len())))).Units
}

// runeOfUnits returns the rune holding UTF-16 unit u, or after it.
func (v *view) runeOfUnits(u int) int {
	return rope.RuneOfUnit(v.shown.text, max(0, u))
}

func countUnits(s string) int {
	n := 0
	for _, r := range s {
		n += utf16.RuneLen(r)
	}
	return n
}

// virtualRune returns the rune of the text with the composition at UTF-16
// unit u of it.
func (v *view) virtualRune(u int) int {
	start, n := v.units(v.caret), countUnits(v.compose)
	if u <= start || v.compose == "" {
		return v.runeOfUnits(u)
	}
	if u <= start+n {
		return v.caret + utf8.RuneCountInString(v.compose[:ui.UTF16ByteOffset(v.compose, u-start)])
	}
	return v.runeOfUnits(u-n) + utf8.RuneCountInString(v.compose)
}

// virtualUnits returns the UTF-16 offset of rune i of the text with the
// composition.
func (v *view) virtualUnits(i int) int {
	n := utf8.RuneCountInString(v.compose)
	if i <= v.caret || n == 0 {
		return v.units(i)
	}
	if i <= v.caret+n {
		return v.units(v.caret) + countUnits(string([]rune(v.compose)[:i-v.caret]))
	}
	return v.units(i-n) + countUnits(v.compose)
}

func (v *view) TextForRange(r ui.TextInputRange) (string, ui.TextInputRange) {
	v.sync()
	if r.Start > r.End {
		r.Start, r.End = r.End, r.Start
	}
	a, z := v.virtualRune(r.Start), v.virtualRune(r.End)
	actual := ui.TextInputRange{Start: v.virtualUnits(a), End: v.virtualUnits(z)}
	n := utf8.RuneCountInString(v.compose)
	if n == 0 {
		return v.shown.Slice(a, z).String(), actual
	}
	var out strings.Builder
	if a < v.caret {
		out.WriteString(v.shown.Slice(a, min(z, v.caret)).String())
	}
	if a < v.caret+n && z > v.caret {
		cr := []rune(v.compose)
		out.WriteString(string(cr[max(0, a-v.caret):min(n, z-v.caret)]))
	}
	if z > v.caret+n {
		out.WriteString(v.shown.Slice(max(a-n, v.caret), z-n).String())
	}
	return out.String(), actual
}

func (v *view) Selection() ui.TextInputSelection {
	v.sync()
	if v.compose != "" {
		start := v.units(v.caret)
		return ui.TextInputSelection{Range: ui.TextInputRange{Start: start + v.composeSelected.Start, End: start + v.composeSelected.End}}
	}
	a, z := v.selection()
	return ui.TextInputSelection{Range: ui.TextInputRange{Start: v.units(a), End: v.units(z)}, Reversed: v.caret == a && a != z}
}

func (v *view) MarkedRange() (ui.TextInputRange, bool) {
	if v.compose == "" {
		return ui.TextInputRange{}, false
	}
	start := v.units(v.caret)
	return ui.TextInputRange{Start: start, End: start + countUnits(v.compose)}, true
}

func (v *view) ReplaceText(r *ui.TextInputRange, s string) {
	v.sync()
	if v.opts.ReadOnly {
		return
	}
	switch {
	case r == nil && v.compose != "":
		v.compose = ""
		v.joinStep()
		v.replace(v.caret, v.caret, s)
	case r == nil:
		v.insert(s)
		v.typed(s)
	default:
		a, z := v.virtualRune(r.Start), v.virtualRune(r.End)
		if a > z {
			a, z = z, a
		}
		transaction := v.composing
		v.commitCompose()
		if !transaction {
			v.record(false)
		} else {
			v.joinStep()
		}
		v.replace(a, z, s)
	}
	v.compose, v.composing = "", false
	v.composeCaret, v.composeSelected = 0, ui.TextInputRange{}
	v.reveal = true
	v.publish()
}

func (v *view) SetMarkedText(r *ui.TextInputRange, s string, selected ui.TextInputRange) {
	v.sync()
	if v.opts.ReadOnly {
		return
	}
	if v.compose == "" || r != nil {
		a, z := v.selection()
		if r != nil {
			a, z = v.virtualRune(r.Start), v.virtualRune(r.End)
			if a > z {
				a, z = z, a
			}
		}
		transaction := v.composing && v.compose != ""
		v.commitCompose()
		if !transaction {
			v.record(false)
		} else {
			v.joinStep()
		}
		v.composing = true
		if a != z {
			v.replace(a, z, "")
		}
	}
	if selected.Start > selected.End {
		selected.Start, selected.End = selected.End, selected.Start
	}
	n := countUnits(s)
	selected.Start, selected.End = min(selected.Start, n), min(selected.End, n)
	selected.Start = countUnits(s[:ui.UTF16ByteOffset(s, selected.Start)])
	selected.End = countUnits(s[:ui.UTF16ByteOffset(s, selected.End)])
	v.compose, v.composeSelected = s, selected
	v.composeCaret = utf8.RuneCountInString(s[:ui.UTF16ByteOffset(s, selected.End)])
	v.reveal = true
	v.publish()
}

func (v *view) UnmarkText() {
	transaction := v.composing
	v.commitCompose()
	// GTK and IMM32 can end the composition before committing their
	// replacement, which shares its step of undo.
	v.composing = transaction
	v.publish()
}

func (v *view) BoundsForRange(r ui.TextInputRange) (ui.Rect, ui.TextInputRange, bool) {
	v.sync()
	if len(v.paras) == 0 || v.width <= 0 {
		return ui.Rect{}, ui.TextInputRange{}, false
	}
	i := v.virtualRune(min(r.Start, r.End))
	// A rune of the text and how far past it in the composition.
	base, extra := i, 0
	if n := utf8.RuneCountInString(v.compose); i > v.caret && n > 0 {
		if i <= v.caret+n {
			base, extra = v.caret, i-v.caret
		} else {
			base = i - n
		}
	}
	x, y, h := v.caretAt(base, extra, v.affinity)
	b := ui.Rect{X: v.padX + x, Y: v.padY + float32(y-v.scroll), W: 1, H: h}
	actual := ui.TextInputRange{Start: v.virtualUnits(i), End: v.virtualUnits(i)}
	if r.End > r.Start {
		x2, _, _ := v.caretAt(base, extra+1, v.affinity)
		if x2 > x {
			b.W = x2 - x
			actual.End = v.virtualUnits(i + 1)
		}
	}
	visible := b.Y+b.H > 0 && b.Y < v.rect.H
	return b, actual, visible
}

func (v *view) IndexForPoint(p ui.Point) (int, bool) {
	v.sync()
	if v.width <= 0 {
		return 0, false
	}
	x, y := v.contentPoint(p.X, p.Y)
	para, l, _ := v.layoutPositionAt(x, y)
	n := utf8.RuneCountInString(v.compose)
	// The rune of the text with the composition: the layout of the
	// caret's paragraph holds it, those after it are past it.
	i := v.shown.Len() + n
	if para >= 0 {
		i = v.shown.paraStart(para) + l
		if cp := v.shown.paraAt(v.caret); n > 0 && para > cp {
			i += n
		}
	}
	return v.virtualUnits(i), true
}
