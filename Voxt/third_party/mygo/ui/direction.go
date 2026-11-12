package ui

import "os"

// Direction is the direction an interface reads in: left to right, or
// right to left, as Arabic and Hebrew are, as Flutter's TextDirection,
// SwiftUI's LayoutDirection and CSS's direction say.
//
// A window takes the direction of the app's language (package mygo sets
// it), and an element that of the element it is built in, unless it is
// given one (Element.Direction). Right to left, a Row lays its children
// out from the right, the Start of Justify, AlignItems and TextAlign is
// the right, a grid's columns go from the right, the logical edges
// (PaddingStart, MarginStart, Start, AnchorTopStart, ...) are on the
// right, horizontal scrolling starts at the right with its offsets
// counted from there, the vertical scroll bar is on the left, and the
// controls mirror what points along the line: arrows, chevrons, the
// fill of sliders and progress bars, switches, and where popups open.
// Paragraphs without a strongly directional character, as numbers, read
// right to left; the others take the direction of their first strong
// character either way.
//
// As in Godot, the coordinate system is not mirrored: X still grows to
// the right, in Rect, in Draw's painter and in pointer positions, and
// what is physical stays so, as in CSS: Left, Right, Padding's left and
// right, AnchorTopLeft. Text, images and icons are not mirrored; draw
// what must point the other way after reading Context.Direction.
type Direction uint8

const (
	LeftToRight Direction = iota
	RightToLeft
)

func (d Direction) String() string {
	if d == RightToLeft {
		return "rtl"
	}
	return "ltr"
}

// The bits of node.dir.
const (
	// dirRTL marks an element laid out right to left.
	dirRTL uint8 = 1 << iota
	// dirSet marks a direction given to the element itself, which its
	// ancestors' do not change.
	dirSet
)

// The bits of node.logical: which of an element's horizontal edges were
// set by their start or end, rather than their left or right.
const (
	logPadStart uint8 = 1 << iota
	logPadEnd
	logMarginStart
	logMarginEnd
	logBorderStart
	logBorderEnd
	logInsetStart
	logInsetEnd

	logPad    = logPadStart | logPadEnd
	logMargin = logMarginStart | logMarginEnd
	logBorder = logBorderStart | logBorderEnd
	logInset  = logInsetStart | logInsetEnd
)

// rtl reports whether the element is laid out right to left.
func (e *node) rtl() bool { return e.dir&dirRTL != 0 }

// startSide and endSide are the indexes of the element's start and end
// edges in its padding, margin, border and inset: left (3) and right (1),
// or the other way round right to left.
func (e *node) startSide() int {
	if e.rtl() {
		return 1
	}
	return 3
}

func (e *node) endSide() int { return 4 - e.startSide() }

// unlog notes that side (1 right, 3 left) of the edges of group (logPad,
// logInset, ...) was set physically, after its start or end was.
func (e *node) unlog(side int, group uint8) {
	const starts = logPadStart | logMarginStart | logBorderStart | logInsetStart
	if side == e.startSide() {
		e.logical &^= group & starts
	} else {
		e.logical &^= group &^ starts
	}
}

// Direction lays the element and what is inside it out in d, whatever the
// direction around it: Direction(ui.LeftToRight) keeps a code sample, a
// phone number or a media player's controls left to right in a
// right-to-left window, as Flutter's Directionality and SwiftUI's
// layoutDirection environment do. The elements built inside it inherit it,
// as do the popups attached to it.
//
// Give it right after creating the element, before its children, logical
// edges and the parts that read it, as widgets do as they are built:
// elements already inside take it, and logical edges move to the other
// side, but what read the old direction keeps it.
func (e *node) Direction(d Direction) *node {
	e.setDir(d == RightToLeft, true)
	return e
}

// setDir sets the element's direction, given to it or inherited, and
// that of the elements inside it that inherit it, moving their logical
// edges to the other side where it changes.
func (e *node) setDir(rtl, explicit bool) {
	was := e.rtl()
	e.dir = 0
	if rtl {
		e.dir |= dirRTL
	}
	if explicit {
		e.dir |= dirSet
	}
	if was == rtl {
		return
	}
	e.swapLogical()
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.dir&dirSet == 0 {
			ch.setDir(rtl, false)
		}
	}
}

// swapLogical moves the edges set by their start and end to the other
// side, for an element whose direction changed. A side that was set
// physically and then by its start or end loses its physical value: the
// direction should be given before the edges.
func (e *node) swapLogical() {
	if e.logical&logPad != 0 {
		e.pad[1], e.pad[3] = e.pad[3], e.pad[1]
	}
	if e.logical&logMargin != 0 {
		e.margin[1], e.margin[3] = e.margin[3], e.margin[1]
	}
	if e.logical&logBorder != 0 {
		e.border[1], e.border[3] = e.border[3], e.border[1]
	}
	if e.logical&logInset != 0 {
		e.inset[1], e.inset[3] = e.inset[3], e.inset[1]
	}
}

// Direction returns the direction of the elements built now: that of the
// element they go in. Widgets of an app's own read it to draw what points
// along the line, as an arrow forward:
//
//	forward := "M5 2 11 8 5 14"
//	if c.Direction() == ui.RightToLeft {
//		forward = "M11 2 5 8 11 14"
//	}
func (c *context) Direction() Direction {
	if c.parent != nil && c.parent.rtl() {
		return RightToLeft
	}
	return LeftToRight
}

// PaddingStart sets the padding on the side lines start: the left, or
// the right of an element laid out right to left, as CSS's
// padding-inline-start does. PaddingEnd sets the other side's.
func (e *node) PaddingStart(v float32) *node {
	e.pad[e.startSide()] = v
	e.logical |= logPadStart
	return e
}

func (e *node) PaddingEnd(v float32) *node {
	e.pad[e.endSide()] = v
	e.logical |= logPadEnd
	return e
}

// MarginStart and MarginEnd set the margin on the side lines start and
// on the other, as PaddingStart does: Auto takes the free space there, so
// that a Row's child with MarginStart(ui.Auto) goes to its end.
func (e *node) MarginStart(v float32) *node {
	e.margin[e.startSide()] = v
	e.logical |= logMarginStart
	return e
}

func (e *node) MarginEnd(v float32) *node {
	e.margin[e.endSide()] = v
	e.logical |= logMarginEnd
	return e
}

// BorderStartWidth and BorderEndWidth set the width of the border on the
// side lines start and on the other, as PaddingStart does: a line
// between a sidebar and the content is BorderEndWidth(1) on the sidebar.
func (e *node) BorderStartWidth(v float32) *node {
	e.border[e.startSide()] = v
	e.logical |= logBorderStart
	return e
}

func (e *node) BorderEndWidth(v float32) *node {
	e.border[e.endSide()] = v
	e.logical |= logBorderEnd
	return e
}

// Start and End place an Absolute element from the side lines start and
// from the other, as Left and Right do, as CSS's inset-inline-start and
// inset-inline-end: from the left and the right, or the right and the
// left right to left. On an element in its parent's layout, they move it
// as relative positioning does.
func (e *node) Start(v float32) *node {
	e.inset[e.startSide()] = px(v)
	e.logical |= logInsetStart
	return e
}

func (e *node) End(v float32) *node {
	e.inset[e.endSide()] = px(v)
	e.logical |= logInsetEnd
	return e
}

// StartPercent and EndPercent place the element as Start and End do, by
// a percentage of the parent's width.
func (e *node) StartPercent(p float32) *node {
	e.inset[e.startSide()] = percent(p)
	e.logical |= logInsetStart
	return e
}

func (e *node) EndPercent(p float32) *node {
	e.inset[e.endSide()] = percent(p)
	e.logical |= logInsetEnd
	return e
}

// arrows returns the arrow keys that go back and forth along a line of
// the element: Left and Right, or Right and Left right to left, as the
// arrows move among tabs, segments and items laid out from the right.
func (e *node) arrows() (back, forth Key) {
	if e.rtl() {
		return KeyRight, KeyLeft
	}
	return KeyLeft, KeyRight
}

// direction returns the direction of the window: the debugging override
// (MYGO_DIRECTION, or the inspector's), else the one the window was
// given, which package mygo takes from the app's language.
func (rt *engine) direction() bool {
	switch {
	case envDirection != 0:
		return envDirection == 2
	case rt.dirOverride != 0:
		return rt.dirOverride == 2
	}
	return rt.host.rightToLeft()
}

// envDirection is MYGO_DIRECTION, read once: 1 for "ltr", 2 for "rtl", 0
// unset. It lays every window of native UI out in that direction, to try
// an interface right to left without translating it, as GTK's
// GTK_DEBUG=invert-text-dir and Qt's -reverse do.
var envDirection = func() uint8 {
	switch os.Getenv("MYGO_DIRECTION") {
	case "ltr":
		return 1
	case "rtl":
		return 2
	}
	return 0
}()
