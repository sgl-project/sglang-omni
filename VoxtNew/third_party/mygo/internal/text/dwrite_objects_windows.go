//go:build windows

package text

import (
	"runtime"
	"slices"
	"sync"
	"syscall"
	"unsafe"
)

// The inline objects of a text layout are IDWriteInlineObjects, COM
// objects of Go whose methods are created once (callbacks are never
// freed): each tells DirectWrite the size of its box, which stands on the
// baseline and draws nothing, where the run of an object's rune would be.

var iidIDWriteInlineObject = guid{0x8339fde3, 0x106f, 0x47ab, [8]byte{0x83, 0x73, 0x1c, 0x62, 0x95, 0xeb, 0x10, 0xb3}}

// dwObject is an IDWriteInlineObject of a box of w×h DIPs.
type dwObject struct {
	vtbl *[7]uintptr
	w, h float32
}

var (
	objectOnce sync.Once
	objectVtbl [7]uintptr
)

func inlineObjectVtbl() *[7]uintptr {
	objectOnce.Do(func() {
		objectVtbl = [7]uintptr{
			syscall.NewCallback(objectQueryInterface),
			syscall.NewCallback(rendererAddRef), // AddRef
			syscall.NewCallback(rendererAddRef), // Release
			syscall.NewCallback(objectDraw),
			syscall.NewCallback(objectGetMetrics),
			syscall.NewCallback(objectGetOverhangMetrics),
			syscall.NewCallback(objectGetBreakConditions),
		}
	})
	return &objectVtbl
}

func objectQueryInterface(this, iid, out uintptr) uintptr {
	switch *(*guid)(ptr(iid)) {
	case iidIUnknown, iidIDWriteInlineObject:
		*(*uintptr)(ptr(out)) = this
		return 0
	}
	*(*uintptr)(ptr(out)) = 0
	return errNoInterface
}

// objectDraw draws nothing: the object's box is the caller's.
func objectDraw(this, context, renderer, x, y, sideways, rtl, effect uintptr) uintptr { return 0 }

// objectGetMetrics writes DWRITE_INLINE_OBJECT_METRICS: width, height,
// baseline, supportsSideways.
func objectGetMetrics(this, out uintptr) uintptr {
	o := (*dwObject)(ptr(this))
	m := (*struct {
		width, height, baseline float32
		sideways                int32
	})(ptr(out))
	m.width, m.height, m.baseline, m.sideways = o.w, o.h, o.h, 0
	return 0
}

// objectGetOverhangMetrics writes DWRITE_OVERHANG_METRICS: none.
func objectGetOverhangMetrics(this, out uintptr) uintptr {
	*(*[4]float32)(ptr(out)) = [4]float32{}
	return 0
}

// objectGetBreakConditions lets lines break around the object as around
// what is next to it (DWRITE_BREAK_CONDITION_NEUTRAL).
func objectGetBreakConditions(this, before, after uintptr) uintptr {
	*(*int32)(ptr(before)) = 0
	*(*int32)(ptr(after)) = 0
	return 0
}

// layoutObjects keeps the objects of the layout being made alive and in
// place until the next.
var layoutObjects struct {
	objects []dwObject
	pinner  runtime.Pinner
}

// setObjects makes each rune of the spans with an ObjectWidth an inline
// object of the layout; units are the code units of the runes.
func setObjects(layout uintptr, spans []Span, units []uint32) {
	n := 0
	from := 0
	for _, sp := range spans {
		to := max(from, min(sp.End, len(units)-1))
		if sp.ObjectWidth > 0 {
			n += to - from
		}
		from = to
	}
	lo := &layoutObjects
	lo.pinner.Unpin()
	if n == 0 {
		lo.objects = nil
		return
	}
	lo.objects = make([]dwObject, n)
	lo.pinner.Pin(&lo.objects[0])
	vtbl := inlineObjectVtbl()
	k := 0
	from = 0
	for _, sp := range spans {
		to := max(from, min(sp.End, len(units)-1))
		if sp.ObjectWidth > 0 {
			for i := from; i < to; i++ {
				o := &lo.objects[k]
				*o = dwObject{vtbl: vtbl, w: sp.ObjectWidth, h: sp.ObjectHeight}
				k++
				r := uintptr(units[i]) | uintptr(units[i+1]-units[i])<<32
				call(layout, layoutSetInlineObject, uintptr(unsafe.Pointer(o)), r)
			}
		}
		from = to
	}
}

// objectRuns returns a run of a glyph for each inline object of the text,
// which DirectWrite draws through DrawInlineObject, whose position Go
// cannot read: hit testing gives where it starts. ends are the code units
// ending the lines; index the rune of each code unit.
func objectRuns(layout uintptr, lines []shapedLine, ends []uint32, spans []Span, units []uint32, index []int) {
	from := 0
	for _, sp := range spans {
		to := max(from, min(sp.End, len(units)-1))
		if sp.ObjectWidth > 0 {
			for i := from; i < to; i++ {
				pos := units[i]
				li := 0
				for li < len(ends)-1 && ends[li] <= pos {
					li++
				}
				var x, y float32
				var hit dwHitTestMetrics
				call(layout, layoutHitTestTextPosition, uintptr(pos), 0, uintptr(unsafe.Pointer(&x)), uintptr(unsafe.Pointer(&y)), uintptr(unsafe.Pointer(&hit)))
				g := Glyph{X: hit.left, Advance: sp.ObjectWidth, Cluster: index[pos], RTL: hit.bidiLevel&1 != 0}
				// Runs are in visual order, as DirectWrite draws them.
				runs := lines[li].runs
				at := 0
				for at < len(runs) && (len(runs[at].glyphs) == 0 || runs[at].glyphs[0].X < g.X) {
					at++
				}
				lines[li].runs = slices.Insert(runs, at, shapedRun{start: i, end: i + 1, glyphs: []Glyph{g}})
			}
		}
		from = to
	}
}
