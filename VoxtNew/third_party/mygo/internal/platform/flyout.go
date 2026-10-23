package platform

// Flyout makes a window a flyout (WindowOptions.Flyout): a borderless
// window owned by its parent and placed next to a rectangle of it, which
// may extend beyond the parent but stays in the work area of a display.
type Flyout struct {
	// Anchor is the rectangle the flyout is placed against, in DIPs
	// relative to the parent's content area.
	Anchor Rect
	// Tray anchors the flyout to a tray icon, where it shows without a
	// parent, above the windows of other apps: the icon's Bounds are the
	// anchor, on the screen. Where they are empty, it goes in the middle
	// of the primary display's work area.
	Tray Tray
	// Side is where the flyout goes from Anchor, and Align how it lines
	// up with it along that side.
	Side  Side
	Align Align
	// Gap is the distance between Anchor and the flyout, in DIPs.
	Gap int
	// Focusable makes the flyout take the keyboard as it shows, and close
	// (WindowHandler.ShouldClose) when the user presses outside it or
	// another window takes the keyboard. Others never take the keyboard
	// and stay until closed.
	Focusable bool
	// Popover shows the flyout in a popover of the system, which draws its
	// background and an arrow pointing at the middle of Anchor, and places
	// it (macOS); Align and Gap do not apply. Elsewhere it is ignored.
	Popover bool
}

// Side is the side of the anchor a flyout goes to.
type Side uint8

// Sides.
const (
	SideBottom Side = iota
	SideTop
	SideRight
	SideLeft
)

// Align lines a flyout up with its anchor along the side it goes to: its
// left or top edges (start), centers, or right or bottom edges (end).
type Align uint8

// Alignments.
const (
	AlignStart Align = iota
	AlignCenter
	AlignEnd
)

// Gravity is one of the nine points of a rectangle: its corners, the
// middles of its edges and its center, as GdkGravity and Wayland's
// xdg_positioner anchors and gravities name them.
// X and Y are -1 for the left or top, 0 for the middle, 1 for the right or
// bottom.
type Gravity struct{ X, Y int }

// Gravities returns the point of the anchor the flyout is placed at, the
// point of the flyout that goes there, and the offset between them.
func (f *Flyout) Gravities() (anchor, window Gravity, dx, dy int) {
	cross := int(f.Align) - 1 // AlignStart, AlignCenter, AlignEnd: -1, 0, 1
	switch f.Side {
	case SideTop:
		return Gravity{cross, -1}, Gravity{cross, 1}, 0, -f.Gap
	case SideRight:
		return Gravity{1, cross}, Gravity{-1, cross}, f.Gap, 0
	case SideLeft:
		return Gravity{-1, cross}, Gravity{1, cross}, -f.Gap, 0
	default:
		return Gravity{cross, 1}, Gravity{cross, -1}, 0, f.Gap
	}
}

// Place returns where a flyout of the given size goes, its anchor being at
// anchor on the screen, in the work area of the display that work
// returns for anchor. It resolves the placement as GTK places popups on
// X11 (gdk_window_move_to_rect, with GDK_ANCHOR_FLIP, GDK_ANCHOR_SLIDE
// and GDK_ANCHOR_RESIZE) and Wayland compositors do (xdg_positioner): on
// each axis the flyout flips to the other side of the anchor, or to the
// other end of it, where it would not fit and the flipped one would;
// then it slides along the edge it would cross, back into the work area;
// then it shrinks to it. flippedX and flippedY tell whether it flipped.
func (f *Flyout) Place(anchor Rect, size Size, work Rect) (r Rect, flippedX, flippedY bool) {
	ra, wa, dx, dy := f.Gravities()
	r.Width, r.Height = size.Width, size.Height
	r.X, flippedX = flipPosition(work.X, work.Width, anchor.X, anchor.Width, r.Width, ra.X, wa.X, dx)
	r.Y, flippedY = flipPosition(work.Y, work.Height, anchor.Y, anchor.Height, r.Height, ra.Y, wa.Y, dy)
	r.X, r.Width = slideAndResize(work.X, work.Width, r.X, r.Width)
	r.Y, r.Height = slideAndResize(work.Y, work.Height, r.Y, r.Height)
	return r, flippedX, flippedY
}

// flipPosition is GDK's maybe_flip_position on one axis.
func flipPosition(boundsPos, boundsSize, rectPos, rectSize, size, rectSign, windowSign, offset int) (int, bool) {
	primary := rectPos + (1+rectSign)*rectSize/2 + offset - (1+windowSign)*size/2
	fits := func(p int) bool { return p >= boundsPos && p+size <= boundsPos+boundsSize }
	if fits(primary) {
		return primary, false
	}
	secondary := rectPos + (1-rectSign)*rectSize/2 - offset - (1-windowSign)*size/2
	if fits(secondary) {
		return secondary, true
	}
	return primary, false
}

// slideAndResize moves a span back into bounds, its start first when it
// is longer, then cuts what still sticks out.
func slideAndResize(boundsPos, boundsSize, pos, size int) (int, int) {
	if pos+size > boundsPos+boundsSize {
		pos = boundsPos + boundsSize - size
	}
	if pos < boundsPos {
		pos = boundsPos
	}
	if pos+size > boundsPos+boundsSize {
		size = boundsPos + boundsSize - pos
	}
	return pos, max(size, 1)
}

// WorkAreaFor returns the work area of the display a flyout anchored at r
// goes to, as GDK picks it: the one whose work area overlaps r the most,
// else the one nearest to r's center.
func WorkAreaFor(r Rect, displays []Display) Rect {
	best, bestArea := -1, 0
	for i, d := range displays {
		if a := overlap(d.WorkArea, r); a > bestArea {
			best, bestArea = i, a
		}
	}
	if best < 0 {
		cx, cy := r.X+r.Width/2, r.Y+r.Height/2
		bestDist := -1
		for i, d := range displays {
			b := d.Bounds
			dx := max(b.X-cx, 0, cx-(b.X+b.Width-1))
			dy := max(b.Y-cy, 0, cy-(b.Y+b.Height-1))
			if dist := dx*dx + dy*dy; bestDist < 0 || dist < bestDist {
				best, bestDist = i, dist
			}
		}
	}
	if best < 0 {
		return Rect{}
	}
	return displays[best].WorkArea
}

func overlap(a, b Rect) int {
	w := min(a.X+a.Width, b.X+b.Width) - max(a.X, b.X)
	h := min(a.Y+a.Height, b.Y+b.Height) - max(a.Y, b.Y)
	if w <= 0 || h <= 0 {
		return 0
	}
	return w * h
}

// Centered returns where a flyout of the given size goes without an
// anchor: in the middle of the work area, as large as it fits.
func Centered(size Size, work Rect) Rect {
	r := Rect{Width: min(size.Width, work.Width), Height: min(size.Height, work.Height)}
	r.X = work.X + (work.Width-r.Width)/2
	r.Y = work.Y + (work.Height-r.Height)/2
	return r
}

// PrimaryWorkArea returns the work area of the primary display.
func PrimaryWorkArea(displays []Display) Rect {
	for _, d := range displays {
		if d.Primary {
			return d.WorkArea
		}
	}
	if len(displays) > 0 {
		return displays[0].WorkArea
	}
	return Rect{}
}
