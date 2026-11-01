package ui

import "slices"

// ScrollState is how far a scroll container scrolls its content, kept in
// the app's state with TrackScroll.
type ScrollState struct {
	Handle

	// X and Y are how far the content is scrolled left and up, in DIPs:
	// right to left (Direction), X is how far it is scrolled right, from
	// its start at the right. Set them to scroll: the container keeps them
	// within its content, so that 0 shows the start and math.MaxFloat32
	// the end.
	X, Y float32
	// MaxX and MaxY are as far as X and Y go, as the frame laid the content
	// out: Y is MaxY at the end. The container sets them.
	MaxX, MaxY float32

	glide *scrollGlide
}

// scrollGlide is a ScrollTo on its way: to x and y, as a says.
type scrollGlide struct {
	x, y   channel
	to     [2]float32
	a      Animation
	moved  bool // the target changed since the container last took it
	frame  uint64
	placed bool
}

// ScrollTo scrolls the container smoothly to x and y as a says, from where
// it shows its content, as SwiftUI's scrollTo in withAnimation:
// math.MaxFloat32 is the end. Called again on the way, it turns smoothly
// toward the new place; the user scrolling stops it. While the desktop
// asks for less motion, or the frame disables animations, it scrolls at
// once.
//
//	if ui.Button(c, "Back to top").Clicked() {
//		app.feed.ScrollTo(0, 0, ui.Smooth)
//	}
func (s *ScrollState) ScrollTo(x, y float32, a Animation) {
	finiteMotion("scroll", float64(x), float64(y))
	if a.Duration < 0 || a.Delay < 0 {
		panic("ui: animation durations must not be negative")
	}
	if s.glide == nil {
		s.glide = &scrollGlide{}
	}
	s.glide.to, s.glide.a, s.glide.moved = [2]float32{x, y}, a, true
}

// Scrolling reports whether a ScrollTo is on its way.
func (s *ScrollState) Scrolling() bool { return s.glide != nil }

// TrackScroll keeps the scrolling of a scroll container (Scroll,
// ScrollHorizontal, ScrollBoth, List) in *s: the container shows its
// content from s.X and s.Y, and writes there as the user scrolls it. The
// app reads where it is, scrolls it by setting them, and keeps a place
// with a ScrollState of its own for each page the container shows.
//
//	// A log that follows its end, unless the user scrolled up from it.
//	if app.log.Y >= app.log.MaxY {
//		app.log.Y = math.MaxFloat32
//	}
//	ui.Scroll(c).TrackScroll(&app.log).Children(app.lines)
//
// A List keeps its place by its rows instead, and scrolls to a row with
// its ListState.
func (e *node) TrackScroll(s *ScrollState) *node {
	if s != nil {
		wrapElement(e).Bind(&s.Handle)
	}
	e.track = s
	if e.adoptScroll() && e.first != nil {
		// What is built was built for the old place, as List's rows.
		e.c.rt.consumed = true
	}
	return e
}

// ScrollIntoView scrolls the containers around the element as little as
// shows it, once the frame is laid out: an element added in this frame
// comes into view with it. Call it in the frame it should come into view,
// as when it is added or chosen: in every frame, it would keep the user
// from scrolling it away.
func (e *node) ScrollIntoView() *node {
	if !e.hasState() {
		return e
	}
	e.c.reveal = append(e.c.reveal, e)
	return e
}

// adoptScroll takes the offset the app set in the element's ScrollState,
// and reports whether it was a new one.
func (e *node) adoptScroll() bool {
	st, t := e.st, e.track
	st.track = t
	if t != nil && t.glide != nil {
		e.glide(t)
	}
	if t == nil || t.X == float32(st.scrollX) && t.Y == float32(st.scrollY) {
		return false
	}
	st.beginMove(e.c.rt.frame)
	if t.X != float32(st.scrollX) {
		st.scrollX = float64(t.X)
	}
	if t.Y != float32(st.scrollY) {
		st.scrollY = float64(t.Y)
	}
	return true
}

// glide moves the offset of a ScrollTo on its way to the frame's time.
func (e *node) glide(t *ScrollState) {
	g, st, rt, now := t.glide, e.st, e.c.rt, e.c.now
	if !g.placed || g.frame+1 < rt.frame {
		// Starting, or the container was not shown: from where it shows.
		g.x = channel{value: st.scrollX, target: st.scrollX, at: now}
		g.y = channel{value: st.scrollY, target: st.scrollY, at: now}
		g.placed, g.moved = true, true
	}
	g.frame = rt.frame
	if g.moved {
		g.moved = false
		to := [2]float64{max(0, min(float64(g.to[0]), float64(t.MaxX))), max(0, min(float64(g.to[1]), float64(t.MaxY)))}
		if rt.preferences().ReduceMotion || rt.instant() {
			g.x = channel{value: to[0], target: to[0], at: now}
			g.y = channel{value: to[1], target: to[1], at: now}
		} else {
			headFor(&g.x, to[0], g.a, now)
			headFor(&g.y, to[1], g.a, now)
		}
	}
	g.x.advance(now)
	g.y.advance(now)
	t.X, t.Y = float32(g.x.value), float32(g.y.value)
	if g.x.moving || g.x.pending || g.y.moving || g.y.pending {
		rt.animating = true
	} else {
		t.glide = nil
	}
}

// maxScroll returns as far as the element scrolls its content, on the axes
// it scrolls.
func (e *node) maxScroll() (x, y float64) {
	if e.flags&flagScrollX != 0 {
		x = max(0, e.contentW-float64(e.w))
	}
	if e.flags&flagScrollY != 0 {
		y = max(0, e.contentH-float64(e.h))
	}
	return x, y
}

// scrollLeft returns how far the scroll container moves its content to
// the left: its horizontal offset, which, right to left, counts from the
// right, as Flutter's do (CSS's go negative instead).
func (e *node) scrollLeft() float64 {
	if e.flags&flagScrollX != 0 && e.rtl() {
		mx, _ := e.maxScroll()
		return mx - e.st.scrollX
	}
	return e.st.scrollX
}

// followShift returns how far what follows the scroll container sideways
// (ScrollWith) moves to the left: right to left, both are laid out from
// the right, which the offset counts from.
func (e *node) followShift() float64 {
	if e.rtl() {
		return -e.st.scrollX
	}
	return e.st.scrollX
}

// scrollTo moves the content of a scroll container, and the ScrollState
// tracking it.
func (s *state) scrollTo(x, y float64) {
	s.scrollX, s.scrollY = x, y
	if t := s.track; t != nil {
		t.X, t.Y = float32(x), float32(y)
	}
}

// beginMove notes where the offset was before the frame first moves it,
// other than an event does.
func (s *state) beginMove(frame uint64) {
	if s.moveFrame != frame {
		s.moveFrame, s.startX, s.startY = frame, s.scrollX, s.scrollY
	}
}

// movedIn reports whether frame moved the offset: what was built read the
// old one, and the next frame builds with the new.
func (s *state) movedIn(frame uint64) bool {
	return s.moveFrame == frame && (s.scrollX != s.startX || s.scrollY != s.startY)
}

// reveal scrolls element id into view in the next frame, once it is laid
// out.
func (rt *engine) reveal(id uint64) {
	if !slices.Contains(rt.revealIDs, id) {
		rt.revealIDs = append(rt.revealIDs, id)
	}
	rt.requestFrame()
}

// revealAll scrolls the elements of root with the given ids into view, in
// that order.
func revealAll(root *node, ids []uint64) {
	found := make([]*node, len(ids))
	var walk func(e *node)
	walk = func(e *node) {
		if i := slices.Index(ids, e.id); i >= 0 && found[i] == nil {
			found[i] = e
		}
		for ch := e.first; ch != nil; ch = ch.next {
			walk(ch)
		}
	}
	walk(root)
	for _, e := range found {
		if e != nil {
			reveal(e)
		}
	}
}

// reveal scrolls the containers around e, innermost first, as little as
// shows its box. The layout has sized and arranged the boxes, relative to
// their parents, but not placed them yet.
func reveal(e *node) {
	for e.isInline() {
		e = e.parent
	}
	x, y, w, h := e.x, e.y, e.w, e.h
	frame := e.c.rt.frame
	for ch, p := e, e.parent; p != nil; ch, p = p, p.parent {
		if p.scrolls() && ch.flags&flagAbsolute == 0 {
			p.adoptScroll()
			st := p.st
			mx, my := p.maxScroll()
			// Where the box is in the content: a List placed its rows
			// from scrollBase.
			cx, cy := float64(x), float64(y)+p.scrollBase
			rtl := p.flags&flagScrollX != 0 && p.rtl()
			left := st.scrollX
			if rtl {
				left = mx - left
			}
			sx := max(0, min(nearest(left, cx, w, p.border[3], p.w-p.border[1]), mx))
			if rtl {
				// Right to left, offsets count from the right.
				sx = mx - sx
			}
			sy := max(0, min(nearest(st.scrollY, cy, h, p.border[0], p.h-p.border[2]), my))
			if sx != st.scrollX || sy != st.scrollY {
				st.beginMove(frame)
				if p.list != nil {
					// A List places its rows anew from there, building
					// those it lacked, which moves the row holding the box.
					was := ch.y
					p.relayoutList(ch, sy)
					cy = float64(y+ch.y-was) + p.scrollBase
				} else {
					st.scrollTo(sx, sy)
				}
			}
			x = float32(cx - p.scrollLeft())
			y = float32(cy - st.scrollY)
		}
		x += p.x
		y += p.y
	}
}

// nearest returns the offset that shows what spans [pos, pos+size) of a
// container's content in its viewport [lo, hi), scrolled by off, moving as
// little as it can: what is larger than the viewport shows from its start.
func nearest(off, pos float64, size, lo, hi float32) float64 {
	start, end := pos-off, pos+float64(size)-off
	switch {
	case start < float64(lo):
		return off - (float64(lo) - start)
	case end > float64(hi):
		return off + min(end-float64(hi), start-float64(lo))
	}
	return off
}

// RowWidth makes the rows of a List at least w DIPs wide, as the columns
// of a table ask: where they are wider than the list, it scrolls sideways
// too, and its rows grow to its width where they are not. A header above
// the rows goes with them through ScrollWith:
//
//	head := ui.Row(c).Clip().Padding(0, 8).Children(app.columnTitles)
//	list := ui.List(c, &app.rows, n).Padding(0, 8).RowWidth(app.columnsWidth())
//	list.Rows(app.row)
//	head.ScrollWith(list)
func (e *node) RowWidth(w float32) *node {
	e.rowMinW = max(w, 0)
	if e.rowMinW > 0 {
		e.flags |= flagScrollX
	} else {
		e.flags &^= flagScrollX
	}
	return e
}

// ScrollWith moves the element's content sideways as far as scroller, a
// scroll container, scrolls its own, as a table's header goes with its
// rows. Clip the element, so that what goes out of it does not show.
func (e *node) ScrollWith(scroller *node) *node {
	e.followX = scroller
	return e
}
