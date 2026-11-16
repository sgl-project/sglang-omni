package vt

import "unsafe"

// Selection gesture events, options and data (GhosttySelectionGesture*).
const (
	gesturePress    = 0
	gestureRelease  = 1
	gestureDrag     = 2
	gestureAutoTick = 3

	gestureOptRef            = 0
	gestureOptPosition       = 1
	gestureOptRepeatDistance = 2
	gestureOptTime           = 3
	gestureOptRepeatInterval = 4
	gestureOptRectangle      = 7
	gestureOptGeometry       = 8
	gestureOptViewport       = 9

	gestureDataDragged    = 1
	gestureDataAutoscroll = 2
)

// Gesture turns presses and drags of the pointer into selections as
// Ghostty does: a click and a drag select cells, a double click words and
// a triple click lines, and dragging past the top or the bottom asks to
// scroll.
type Gesture struct {
	h                          uintptr
	press, release, drag, tick uintptr
}

// Geometry is the grid in pixels, for drags.
type Geometry struct {
	Columns, CellWidth, PadLeft, Height int
}

// NewGesture makes a gesture: repeated clicks count within interval
// nanoseconds of each other and distance pixels.
func NewGesture(interval uint64, distance float64) (*Gesture, error) {
	g := &Gesture{}
	if err := result(call(fnGestureNew, 0, uintptr(unsafe.Pointer(&g.h)))); err != nil {
		return nil, err
	}
	for _, e := range []struct {
		h   *uintptr
		typ int
	}{{&g.press, gesturePress}, {&g.release, gestureRelease}, {&g.drag, gestureDrag}, {&g.tick, gestureAutoTick}} {
		if err := result(call(fnGestureEventNew, 0, uintptr(unsafe.Pointer(e.h)), uintptr(e.typ))); err != nil {
			g.Free(nil)
			return nil, err
		}
	}
	call(fnGestureEventSet, g.press, gestureOptRepeatInterval, uintptr(unsafe.Pointer(&interval)))
	call(fnGestureEventSet, g.press, gestureOptRepeatDistance, uintptr(unsafe.Pointer(&distance)))
	return g, nil
}

// Free frees the gesture, which belongs to t (nil if it was never used).
func (g *Gesture) Free(t *Terminal) {
	for _, e := range []uintptr{g.press, g.release, g.drag, g.tick} {
		if e != 0 {
			call(fnGestureEventFree, e)
		}
	}
	if g.h != 0 {
		th := uintptr(0)
		if t != nil {
			th = t.h
		}
		call(fnGestureFree, g.h, th)
	}
	*g = Gesture{}
}

func (g *Gesture) apply(t *Terminal, ev uintptr) (Selection, bool) {
	s := Selection{size: unsafe.Sizeof(Selection{})}
	return s, ok(call(fnGestureEvent, g.h, t.h, ev, uintptr(unsafe.Pointer(&s))))
}

// Press presses the pointer on cell ref at x, y pixels, at time ns of a
// monotonic clock. A double or triple click selects at once.
func (g *Gesture) Press(t *Terminal, ref GridRef, x, y float64, ns uint64) (Selection, bool) {
	pos := surfacePosition{x, y}
	call(fnGestureEventSet, g.press, gestureOptRef, uintptr(unsafe.Pointer(&ref)))
	call(fnGestureEventSet, g.press, gestureOptPosition, uintptr(unsafe.Pointer(&pos)))
	call(fnGestureEventSet, g.press, gestureOptTime, uintptr(unsafe.Pointer(&ns)))
	return g.apply(t, g.press)
}

// Drag moves the pressed pointer to cell ref at x, y pixels, selecting a
// rectangle when rect is set.
func (g *Gesture) Drag(t *Terminal, ref GridRef, x, y float64, geo Geometry, rect bool) (Selection, bool) {
	pos := surfacePosition{x, y}
	gg := gestureGeometry{uint32(max(geo.Columns, 1)), uint32(max(geo.CellWidth, 1)), uint32(max(geo.PadLeft, 0)), uint32(max(geo.Height, 1))}
	call(fnGestureEventSet, g.drag, gestureOptRef, uintptr(unsafe.Pointer(&ref)))
	call(fnGestureEventSet, g.drag, gestureOptPosition, uintptr(unsafe.Pointer(&pos)))
	call(fnGestureEventSet, g.drag, gestureOptGeometry, uintptr(unsafe.Pointer(&gg)))
	call(fnGestureEventSet, g.drag, gestureOptRectangle, uintptr(unsafe.Pointer(&rect)))
	return g.apply(t, g.drag)
}

// Tick continues a drag past an edge after the viewport scrolled, with the
// pointer at x, y pixels over column col of row row of the viewport.
func (g *Gesture) Tick(t *Terminal, col, row int, x, y float64, geo Geometry, rect bool) (Selection, bool) {
	pos := surfacePosition{x, y}
	gg := gestureGeometry{uint32(max(geo.Columns, 1)), uint32(max(geo.CellWidth, 1)), uint32(max(geo.PadLeft, 0)), uint32(max(geo.Height, 1))}
	vp := struct {
		x uint16
		_ uint16
		y uint32
	}{x: uint16(max(col, 0)), y: uint32(max(row, 0))}
	call(fnGestureEventSet, g.tick, gestureOptPosition, uintptr(unsafe.Pointer(&pos)))
	call(fnGestureEventSet, g.tick, gestureOptGeometry, uintptr(unsafe.Pointer(&gg)))
	call(fnGestureEventSet, g.tick, gestureOptViewport, uintptr(unsafe.Pointer(&vp)))
	call(fnGestureEventSet, g.tick, gestureOptRectangle, uintptr(unsafe.Pointer(&rect)))
	return g.apply(t, g.tick)
}

// Release releases the pointer over cell ref, or outside the grid when ref
// is nil.
func (g *Gesture) Release(t *Terminal, ref *GridRef) {
	call(fnGestureEventSet, g.release, gestureOptRef, uintptr(unsafe.Pointer(ref)))
	g.apply(t, g.release)
}

// Dragged reports whether the pointer moved to another cell since it was
// pressed.
func (g *Gesture) Dragged(t *Terminal) bool {
	var v bool
	return ok(call(fnGestureGet, g.h, t.h, gestureDataDragged, uintptr(unsafe.Pointer(&v)))) && v
}

// Autoscroll reports whether a drag past an edge asks to scroll up (-1) or
// down (1).
func (g *Gesture) Autoscroll(t *Terminal) int {
	var v int32
	call(fnGestureGet, g.h, t.h, gestureDataAutoscroll, uintptr(unsafe.Pointer(&v)))
	switch v {
	case 1:
		return -1
	case 2:
		return 1
	}
	return 0
}

// Reset forgets the gesture's clicks.
func (g *Gesture) Reset(t *Terminal) { call(fnGestureReset, g.h, t.h) }
