//go:build windows && (amd64 || arm64)

package windows

import (
	"math"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"unicode"
	"unicode/utf16"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

// The text of a text field is UI Automation's Text pattern: the field is
// an ITextProvider, and ranges of its text are ITextRangeProviders, COM
// objects of their own, which Narrator moves by characters, formats,
// words and lines, reads, and asks the formats of: the font, its size and
// weight, italics, lines, colors and headings of a rich text's runs
// (AccessNode.TextRuns). Ranges count runes of the node's Value, as the
// tree does, and read the node as it is when they are used.

var (
	procUiaGetReservedMixedAttributeValue = uiaCore.NewProc("UiaGetReservedMixedAttributeValue")
	procUiaGetReservedNotSupportedValue   = uiaCore.NewProc("UiaGetReservedNotSupportedValue")

	iidITextRangeProvider = guid("5347ad7b-c355-46f8-aff5-909033582f63")

	textRangeVtbl []uintptr
	eInvalidArg   = uintptr(0x80070057)
	// textRanges keeps the ranges COM holds alive.
	textRanges = map[*textRange]bool{}
)

// TextUnit and TextPatternRangeEndpoint.
const (
	unitCharacter = iota
	unitFormat
	unitWord
	unitLine
	unitParagraph
	unitPage
	unitDocument
)

// Text attributes of UI Automation.
const (
	attrBackgroundColor    = 40001
	attrFontName           = 40005
	attrFontSize           = 40006
	attrFontWeight         = 40007
	attrForegroundColor    = 40008
	attrIsItalic           = 40014
	attrIsReadOnly         = 40015
	attrStrikethroughStyle = 40026
	attrUnderlineStyle     = 40030
	attrStyleID            = 40034

	styleIDHeading1 = 70001
	styleIDNormal   = 70012
)

// textRange is an ITextRangeProvider: runes start to end of the text of
// element e.
type textRange struct {
	vtbl       *uintptr
	refs       int32
	e          *uiaElement
	start, end int
}

func newTextRange(e *uiaElement, start, end int) *textRange {
	r := &textRange{vtbl: &textRangeVtbl[0], refs: 1, e: e, start: start, end: end}
	r.clamp()
	textRanges[r] = true
	e.addRef()
	return r
}

func rangeOf(this uintptr) *textRange { return (*textRange)(native(this)) }

func (r *textRange) ptr() uintptr { return uintptr(unsafe.Pointer(r)) }

func (r *textRange) release() int32 {
	if r.refs--; r.refs == 0 {
		delete(textRanges, r)
		r.e.release()
	}
	return r.refs
}

// text returns the runes of the element's text.
func (r *textRange) text() []rune { return r.e.runes() }

// runes returns the runes of the element's value, decoded once for each
// value it has, as ranges ask for them call after call.
func (e *uiaElement) runes() []rune {
	if e.runesOf != e.n.Value || e.textRunes == nil {
		e.runesOf, e.textRunes = e.n.Value, []rune(e.n.Value)
	}
	return e.textRunes
}

// clamp keeps the range within the text, start before end.
func (r *textRange) clamp() {
	n := len(r.e.runes())
	r.start, r.end = max(0, min(r.start, n)), max(0, min(r.end, n))
	if r.end < r.start {
		r.end = r.start
	}
}

// outRange stores a range at a pointer, which its caller owns.
func outRange(p uintptr, r *textRange) uintptr {
	if r == nil {
		*(*uintptr)(native(p)) = 0
		return sOK
	}
	*(*uintptr)(native(p)) = r.ptr()
	return sOK
}

// rangeArray returns a SAFEARRAY of ranges, which own a reference each.
func rangeArray(ranges ...*textRange) uintptr {
	arr, _, _ := procSafeArrayCreateVector.Call(vtUnknown, 0, uintptr(len(ranges)))
	for i, r := range ranges {
		index := int32(i)
		procSafeArrayPutElement.Call(arr, uintptr(unsafe.Pointer(&index)), r.ptr())
		r.release()
	}
	return arr
}

// textProviderMethods are ITextProvider's.
func textProviderMethods() []uintptr {
	cb := syscall.NewCallback
	live := func(this uintptr) (*uiaElement, uintptr) {
		e := uiaOf(this)
		if e.dead {
			return nil, uiaElementNotAvailable
		}
		return e, sOK
	}
	return []uintptr{
		cb(func(this, p uintptr) uintptr { // GetSelection
			e, hr := live(this)
			if e == nil {
				return hr
			}
			*(*uintptr)(native(p)) = rangeArray(newTextRange(e, e.n.SelStart, e.n.SelEnd))
			return sOK
		}),
		cb(func(this, p uintptr) uintptr { // GetVisibleRanges
			e, hr := live(this)
			if e == nil {
				return hr
			}
			*(*uintptr)(native(p)) = rangeArray(newTextRange(e, 0, len(e.runes())))
			return sOK
		}),
		cb(func(this, child, p uintptr) uintptr { // RangeFromChild
			e, hr := live(this)
			if e == nil {
				return hr
			}
			return outRange(p, newTextRange(e, 0, 0))
		}),
		// RangeFromPoint(UiaPoint, out): the point, two doubles, comes by
		// reference on x64 and in floating point registers on ARM64,
		// where the out pointer is the next integer argument. Without the
		// geometry of characters, the caret's place answers.
		cb(func(this, a1, a2 uintptr) uintptr { // RangeFromPoint
			e, hr := live(this)
			p := a2
			if runtime.GOARCH == "arm64" {
				p = a1
			}
			if e == nil {
				*(*uintptr)(native(p)) = 0
				return hr
			}
			return outRange(p, newTextRange(e, e.n.SelEnd, e.n.SelEnd))
		}),
		cb(func(this, p uintptr) uintptr { // get_DocumentRange
			e, hr := live(this)
			if e == nil {
				*(*uintptr)(native(p)) = 0
				return hr
			}
			return outRange(p, newTextRange(e, 0, len(e.runes())))
		}),
		cb(func(this, p uintptr) uintptr { // get_SupportedTextSelection
			*(*int32)(native(p)) = 1 // SupportedTextSelection_Single
			return sOK
		}),
	}
}

// isBoundary reports whether rune i of text starts a unit: a word, a
// line or paragraph (after a newline), a run of a format (runs), or the
// document.
func isBoundary(text []rune, runs []platform.AccessTextRun, i, unit int) bool {
	n := len(text)
	if i <= 0 || i >= n {
		return true
	}
	switch unit {
	case unitCharacter:
		return true
	case unitFormat:
		k := sort.Search(len(runs), func(k int) bool { return runs[k].End >= i })
		return k < len(runs) && (runs[k].Start == i || runs[k].End == i)
	case unitWord:
		a, b := text[i-1], text[i]
		return !unicode.IsSpace(b) && (unicode.IsSpace(a) || wordClass(a) != wordClass(b)) || a == '\n'
	case unitLine, unitParagraph:
		return text[i-1] == '\n'
	}
	return false
}

// wordClass sorts runes into words, punctuation and spaces.
func wordClass(r rune) int {
	switch {
	case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_':
		return 1
	case unicode.IsSpace(r):
		return 0
	}
	return 2
}

// next returns the next start of a unit after i, or the end; prev the
// last one before i, or 0.
func next(text []rune, runs []platform.AccessTextRun, i, unit int) int {
	if unit == unitFormat {
		// The nearest edge of a run after i, found by halves.
		k := sort.Search(len(runs), func(k int) bool { return runs[k].End > i })
		if k < len(runs) {
			if runs[k].Start > i {
				return min(runs[k].Start, len(text))
			}
			return min(runs[k].End, len(text))
		}
		return len(text)
	}
	for i++; i < len(text) && !isBoundary(text, runs, i, unit); i++ {
	}
	return min(i, len(text))
}

func prev(text []rune, runs []platform.AccessTextRun, i, unit int) int {
	if unit == unitFormat {
		// The nearest edge of a run before i.
		k := sort.Search(len(runs), func(k int) bool { return runs[k].Start >= i }) - 1
		if k >= 0 {
			if runs[k].End < i {
				return runs[k].End
			}
			return runs[k].Start
		}
		return 0
	}
	for i--; i > 0 && !isBoundary(text, runs, i, unit); i-- {
	}
	return max(i, 0)
}

// expand makes the range the unit holding its start.
func (r *textRange) expand(unit int) {
	text, runs := r.text(), r.e.n.TextRuns
	n := len(text)
	if unit >= unitPage {
		r.start, r.end = 0, n
		return
	}
	s := r.start
	if s >= n && n > 0 {
		s = n - 1
	}
	if !isBoundary(text, runs, s, unit) {
		s = prev(text, runs, s, unit)
	}
	r.start, r.end = s, next(text, runs, s, unit)
}

// moveEndpoint moves the start (end false) or the end of the range by
// count units, the other with it where they cross, and returns how many
// it moved.
func (r *textRange) moveEndpoint(end bool, unit, count int) int {
	text, runs := r.text(), r.e.n.TextRuns
	at := r.start
	if end {
		at = r.end
	}
	moved := 0
	if unit >= unitPage {
		unit = unitDocument
	}
	for ; count > 0 && at < len(text); count-- {
		at = next(text, runs, at, unit)
		moved++
	}
	for ; count < 0 && at > 0; count++ {
		at = prev(text, runs, at, unit)
		moved--
	}
	if end {
		r.end = at
		r.start = min(r.start, at)
	} else {
		r.start = at
		r.end = max(r.end, at)
	}
	return moved
}

// formatOf returns the value of a text attribute over the range: the
// format of each piece, the same for all, or the mixed value; a caret
// has that of the rune after it.
func (r *textRange) formatOf(attr int) (variant, bool) {
	n := r.e.n
	text := r.text()
	value := func(run platform.AccessTextRun) (variant, bool) {
		switch attr {
		case attrFontName:
			family := run.Family
			if family == "" || family == "system-ui" {
				family = "Segoe UI"
			}
			return variant{VT: vtBSTR, Val: uint64(bstr(family))}, true
		case attrFontSize:
			size := float64(run.Size)
			if size <= 0 {
				size = 14
			}
			return variant{VT: vtR8, Val: math.Float64bits(size * 0.75)}, true
		case attrFontWeight:
			w := run.Weight
			if w <= 0 {
				w = 400
			}
			return variant{VT: vtI4, Val: uint64(w)}, true
		case attrIsItalic:
			return boolVariant(run.Italic), true
		case attrUnderlineStyle:
			if run.Underline {
				return variant{VT: vtI4, Val: 1}, true // TextDecorationLineStyle_Single
			}
			return variant{VT: vtI4}, true
		case attrStrikethroughStyle:
			if run.Strikethrough {
				return variant{VT: vtI4, Val: 1}, true
			}
			return variant{VT: vtI4}, true
		case attrForegroundColor, attrBackgroundColor:
			c := run.Color
			if attr == attrBackgroundColor {
				c = run.Background
			}
			if c[3] == 0 {
				return variant{}, false
			}
			return variant{VT: vtI4, Val: uint64(c[0]) | uint64(c[1])<<8 | uint64(c[2])<<16}, true // COLORREF
		case attrStyleID:
			if run.Heading > 0 {
				return variant{VT: vtI4, Val: uint64(styleIDHeading1 + min(run.Heading, 9) - 1)}, true
			}
			return variant{VT: vtI4, Val: styleIDNormal}, true
		case attrIsReadOnly:
			return boolVariant(n.States&platform.AccessReadOnly != 0), true
		}
		return variant{}, false
	}
	// The pieces of the range, of one format each.
	start, end := r.start, r.end
	if start == end {
		end = min(start+1, len(text))
		if start == end && start > 0 {
			start--
		}
	}
	var first variant
	firstOK, mixed := false, false
	for at := start; at < end || at == start; {
		run := runAt(n.TextRuns, at)
		v, ok := value(run)
		if at == start {
			first, firstOK = v, ok
		} else if ok != firstOK || !sameVariant(v, first) {
			if v.VT == vtBSTR {
				procSysFreeString.Call(uintptr(v.Val))
			}
			mixed = true
			break
		} else if v.VT == vtBSTR {
			procSysFreeString.Call(uintptr(v.Val))
		}
		if at >= end {
			break
		}
		at = next(text, n.TextRuns, at, unitFormat)
	}
	if mixed {
		if first.VT == vtBSTR {
			procSysFreeString.Call(uintptr(first.Val))
		}
		var unk uintptr
		procUiaGetReservedMixedAttributeValue.Call(uintptr(unsafe.Pointer(&unk)))
		return variant{VT: vtUnknown, Val: uint64(unk)}, true
	}
	return first, firstOK
}

// runAt returns the run of runs holding rune i, or one without formats.
func runAt(runs []platform.AccessTextRun, i int) platform.AccessTextRun {
	k := sort.Search(len(runs), func(k int) bool { return runs[k].End > i })
	if k < len(runs) && runs[k].Start <= i {
		return runs[k]
	}
	return platform.AccessTextRun{}
}

// notSupported returns the value of attributes the text has not.
func notSupported() variant {
	var unk uintptr
	procUiaGetReservedNotSupportedValue.Call(uintptr(unsafe.Pointer(&unk)))
	return variant{VT: vtUnknown, Val: uint64(unk)}
}

// sameVariant reports whether a value of an attribute is that of v,
// comparing strings by their text.
func sameVariant(a, b variant) bool {
	if a.VT != b.VT {
		return false
	}
	if a.VT == vtBSTR {
		return bstrString(uintptr(a.Val)) == bstrString(uintptr(b.Val))
	}
	if a.VT == vtBool {
		return (a.Val&0xFFFF != 0) == (b.Val&0xFFFF != 0)
	}
	if a.VT == vtI4 {
		return uint32(a.Val) == uint32(b.Val)
	}
	return a.Val == b.Val
}

// bstrString reads a BSTR.
func bstrString(b uintptr) string {
	if b == 0 {
		return ""
	}
	// The length in bytes comes before the string.
	n := *(*uint32)(unsafe.Add(native(b), -4)) / 2
	return string(utf16.Decode(unsafe.Slice((*uint16)(native(b)), n)))
}

func initTextRanges() {
	cb := syscall.NewCallback
	live := func(this uintptr) (*textRange, uintptr) {
		r := rangeOf(this)
		if r.e.dead {
			return nil, uiaElementNotAvailable
		}
		r.clamp()
		return r, sOK
	}
	other := func(p uintptr) *textRange {
		if p == 0 {
			return nil
		}
		r := rangeOf(p)
		if !textRanges[r] {
			return nil
		}
		return r
	}
	endpoint := func(r *textRange, end uintptr) int {
		if end != 0 {
			return r.end
		}
		return r.start
	}
	textRangeVtbl = []uintptr{
		cb(func(this, riid, p uintptr) uintptr { // QueryInterface
			iid := *(*GUID)(native(riid))
			if iid == iidIUnknown || iid == iidITextRangeProvider {
				r := rangeOf(this)
				r.refs++
				*(*uintptr)(native(p)) = this
				return sOK
			}
			*(*uintptr)(native(p)) = 0
			return eNoInterface
		}),
		cb(func(this uintptr) uintptr { r := rangeOf(this); r.refs++; return uintptr(r.refs) }),
		cb(func(this uintptr) uintptr { return uintptr(rangeOf(this).release()) }),
		cb(func(this, p uintptr) uintptr { // Clone
			r := rangeOf(this)
			return outRange(p, newTextRange(r.e, r.start, r.end))
		}),
		cb(func(this, q, p uintptr) uintptr { // Compare
			r, o := rangeOf(this), other(q)
			setBool(p, o != nil && o.e == r.e && o.start == r.start && o.end == r.end)
			return sOK
		}),
		cb(func(this, ep, q, qep, p uintptr) uintptr { // CompareEndpoints
			r, o := rangeOf(this), other(q)
			if o == nil {
				return eInvalidArg
			}
			*(*int32)(native(p)) = int32(endpoint(r, ep) - endpoint(o, qep))
			return sOK
		}),
		cb(func(this, unit uintptr) uintptr { // ExpandToEnclosingUnit
			r, hr := live(this)
			if r == nil {
				return hr
			}
			r.expand(int(unit))
			return sOK
		}),
		cb(func(this, attr, val, backward, p uintptr) uintptr { // FindAttribute
			r, hr := live(this)
			if r == nil {
				*(*uintptr)(native(p)) = 0
				return hr
			}
			want := *(*variant)(native(val))
			text := r.text()
			// The pieces of one format within the range, in order.
			var pieces [][2]int
			for at := r.start; at < r.end; {
				to := min(next(text, r.e.n.TextRuns, at, unitFormat), r.end)
				pieces = append(pieces, [2]int{at, to})
				at = to
			}
			if backward != 0 {
				for i, j := 0, len(pieces)-1; i < j; i, j = i+1, j-1 {
					pieces[i], pieces[j] = pieces[j], pieces[i]
				}
			}
			for _, piece := range pieces {
				probe := &textRange{e: r.e, start: piece[0], end: piece[1]}
				v, ok := probe.formatOf(int(attr))
				match := ok && sameVariant(v, want)
				if v.VT == vtBSTR {
					procSysFreeString.Call(uintptr(v.Val))
				}
				if match {
					return outRange(p, newTextRange(r.e, piece[0], piece[1]))
				}
			}
			return outRange(p, nil)
		}),
		cb(func(this, s, backward, ignoreCase, p uintptr) uintptr { // FindText
			r, hr := live(this)
			if r == nil {
				*(*uintptr)(native(p)) = 0
				return hr
			}
			text := r.text()
			hay, needle := string(text[r.start:r.end]), bstrString(s)
			if ignoreCase != 0 {
				hay, needle = strings.ToLower(hay), strings.ToLower(needle)
			}
			i := strings.Index(hay, needle)
			if backward != 0 {
				i = strings.LastIndex(hay, needle)
			}
			if needle == "" || i < 0 {
				return outRange(p, nil)
			}
			at := r.start + len([]rune(hay[:i]))
			return outRange(p, newTextRange(r.e, at, at+len([]rune(needle))))
		}),
		cb(func(this, attr, p uintptr) uintptr { // GetAttributeValue
			v := (*variant)(native(p))
			r, hr := live(this)
			if r == nil {
				*v = variant{}
				return hr
			}
			if value, ok := r.formatOf(int(attr)); ok {
				*v = value
			} else {
				*v = notSupported()
			}
			return sOK
		}),
		cb(func(this, p uintptr) uintptr { // GetBoundingRectangles
			r, hr := live(this)
			if r == nil {
				*(*uintptr)(native(p)) = 0
				return hr
			}
			// The field's box: the geometry of its characters is the
			// content's.
			var rects []float64
			if r.end > r.start {
				b := r.e.screenRect()
				rects = []float64{b.Left, b.Top, b.Width, b.Height}
			}
			arr, _, _ := procSafeArrayCreateVector.Call(vtR8, 0, uintptr(len(rects)))
			for i := range rects {
				index := int32(i)
				procSafeArrayPutElement.Call(arr, uintptr(unsafe.Pointer(&index)), uintptr(unsafe.Pointer(&rects[i])))
			}
			*(*uintptr)(native(p)) = arr
			return sOK
		}),
		cb(func(this, p uintptr) uintptr { // GetEnclosingElement
			return out(p, rangeOf(this).e, ifaceSimple)
		}),
		cb(func(this, maxLength, p uintptr) uintptr { // GetText
			r, hr := live(this)
			if r == nil {
				*(*uintptr)(native(p)) = 0
				return hr
			}
			u := utf16.Encode(r.text()[r.start:r.end])
			if m := int32(maxLength); m >= 0 && int(m) < len(u) {
				u = u[:m]
			}
			*(*uintptr)(native(p)) = bstr(string(utf16.Decode(u)))
			return sOK
		}),
		cb(func(this, unit, count, p uintptr) uintptr { // Move
			r, hr := live(this)
			if r == nil {
				return hr
			}
			n, u := int(int32(count)), int(unit)
			moved := 0
			switch {
			case n == 0:
			case r.start == r.end:
				// An insertion point moves as one.
				moved = r.moveEndpoint(false, u, n)
				r.end = r.start
			case u >= unitPage:
				// The one page is the whole text: there is none to move to.
				r.start, r.end = 0, len(r.text())
			default:
				// As the UIA documentation says: collapse to the start, back
				// to the start of its unit, move by units, then take the unit
				// there; a range doesn't move past the last unit.
				text, runs := r.text(), r.e.n.TextRuns
				at := r.start
				if at < len(text) && !isBoundary(text, runs, at, u) {
					at = prev(text, runs, at, u)
				}
				for ; moved < n; moved++ {
					nx := next(text, runs, at, u)
					if nx >= len(text) || nx == at {
						break
					}
					at = nx
				}
				for ; moved > n && at > 0; moved-- {
					at = prev(text, runs, at, u)
				}
				r.start, r.end = at, next(text, runs, at, u)
			}
			*(*int32)(native(p)) = int32(moved)
			return sOK
		}),
		cb(func(this, ep, unit, count, p uintptr) uintptr { // MoveEndpointByUnit
			r, hr := live(this)
			if r == nil {
				return hr
			}
			*(*int32)(native(p)) = int32(r.moveEndpoint(ep != 0, int(unit), int(int32(count))))
			return sOK
		}),
		cb(func(this, ep, q, qep uintptr) uintptr { // MoveEndpointByRange
			r, o := rangeOf(this), other(q)
			if o == nil {
				return eInvalidArg
			}
			at := endpoint(o, qep)
			if ep != 0 {
				r.end, r.start = at, min(r.start, at)
			} else {
				r.start, r.end = at, max(r.end, at)
			}
			return sOK
		}),
		cb(func(this uintptr) uintptr { // Select
			r, hr := live(this)
			if r == nil {
				return hr
			}
			return r.e.selectText(r.start, r.end)
		}),
		cb(func(this uintptr) uintptr { return uiaInvalidOperation }), // AddToSelection
		cb(func(this uintptr) uintptr { return uiaInvalidOperation }), // RemoveFromSelection
		cb(func(this, top uintptr) uintptr { // ScrollIntoView
			r, hr := live(this)
			if r == nil {
				return hr
			}
			return r.e.act(platform.AccessScrollIntoView, "")
		}),
		cb(func(this, p uintptr) uintptr { // GetChildren
			arr, _, _ := procSafeArrayCreateVector.Call(vtUnknown, 0, 0)
			*(*uintptr)(native(p)) = arr
			return sOK
		}),
	}
}

// selectText selects runes start to end of a text field, as assistive
// technology asks.
func (e *uiaElement) selectText(start, end int) uintptr {
	switch {
	case e.dead:
		return uiaElementNotAvailable
	case e.n.States&platform.AccessDisabled != 0:
		return uiaElementNotEnabled
	}
	e.tree.s.send(platform.SurfaceEvent{Kind: platform.AccessAction, ID: e.n.ID, Action: platform.AccessSelectText, From: start, To: end})
	return sOK
}
