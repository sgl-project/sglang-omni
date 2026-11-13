package ui

import (
	"math"
	"slices"
	"time"
)

// ElementTransition animates an element (Element.Transition) from what the
// last frame showed to what the layout makes of it in this one: where it
// is in its parent, its size, and its background and border colors. Enter
// and Exit animate it as it appears and as it goes:
//
//	ui.Row(c).Key(item.ID).Transition(ui.ElementTransition{
//		Enter: &ui.Motion{Collapse: true},
//		Exit:  &ui.Motion{Collapse: true},
//	})
type ElementTransition struct {
	// Duration is how long a change takes: 200 ms when zero.
	Duration time.Duration
	// Ease is how the element goes, as for AnimateWith: EaseOut when nil.
	Ease Easing
	// Animation, if set, is how changes go instead of Duration and Ease:
	// with a spring (Snappy, …), a change on the way keeps the element
	// going as it went; with a Delay, changes wait, as the rows of a list
	// entering one after another.
	Animation *Animation
	// Position, Size and Colors choose what animates: where the element is
	// in its parent, its width and height (its content laid out at each
	// size on the way), and its background and border colors. None set
	// animates them all.
	Position, Size, Colors bool
	// Enter, if set, is where the element comes from as it appears in a
	// parent that the last frame showed: it starts there, and goes to its
	// place.
	Enter *Motion
	// Exit, if set, is where the element goes once it is no longer built,
	// while its parent still is: a copy of it, as the last frame showed
	// it, goes there under its siblings, taking neither room, the pointer
	// nor the focus, and is then gone.
	Exit *Motion
	// Match, if set, is an identity the element shares with one that the
	// frame before showed elsewhere, even in another parent, as a photo's
	// thumbnail in a grid and the photo in a detail view: the element
	// appearing comes from where that one was, at its size and in its
	// colors, as SwiftUI's matchedGeometryEffect and Motion's layoutId.
	// Use a value of a type of your own, as a struct of the photo's ID.
	Match any
}

// Motion is where an element appearing comes from, or where one going
// goes, from its place in the layout.
type Motion struct {
	// X and Y move it by as many DIPs.
	X, Y float32
	// Opacity is how opaque it is there: 0, as by default, is
	// transparent, and 1 opaque.
	Opacity float32
	// Collapse makes it nothing along its parent's direction: no height in
	// a column, no width in a row. Its content keeps its layout, clipped.
	Collapse bool
	// Scale is how large it is there, around its TransformOrigin: 0.9 a
	// tenth smaller; 0, as by default, keeps it at its size, as 1 does.
	// Rotate turns it by as many degrees clockwise.
	Scale, Rotate float32
}

// defaultDuration is the Duration of a Transition without one.
const defaultDuration = 200 * time.Millisecond

// Transition animates the element as t asks. The element needs the same
// ID from frame to frame: a Key, or the same place among its siblings.
// Elements that the change moves as well need a Transition of their own to
// move smoothly, the same one to move in step with it, as the rows of a
// list:
//
//	for _, it := range app.items {
//		ui.Row(c).Key(it.ID).Transition(rowTransition).Children(func() {
//			ui.Text(c, it.Title)
//		})
//	}
//
// Changes made by resizing the window do not animate. While the desktop
// asks for less motion (Preferences.ReduceMotion), nothing moves: colors
// still change smoothly, and an Enter or Exit that fades without
// collapsing fades in place.
func (e *node) Transition(t ElementTransition) *node {
	e.c.transitions = append(e.c.transitions, transitionUse{e: e, t: t})
	return e
}

// Transitioning reports whether the element's Transition was moving it,
// or changing its colors, or its pose (AnimatePose) or Draggable was
// moving it, in the frame before: false once it is where it goes.
func (e *node) Transitioning() bool {
	if !e.hasState() {
		return false
	}
	frame := e.c.rt.frame
	if p := e.st.pose; p != nil && p.frame+1 >= frame && p.moving() {
		return true
	}
	if d := e.st.dragRec; d != nil && d.frame+1 >= frame && d.moving() {
		return true
	}
	r := e.st.trec
	return r != nil && r.built+1 >= frame && (r.moving != 0 || r.colorMoving || r.matching)
}

// transitionUse is a call of Transition in the frame being built.
type transitionUse struct {
	e *node
	t ElementTransition
	r *transition
}

// tbox is what a transition moves: an element's place in its parent's
// box, its size, its opacity, and the scale and rotation of a Motion.
type tbox struct{ x, y, w, h, alpha, scale, rot float32 }

// vals returns the components of b, in the order of the channels.
func (b tbox) vals() [7]float32 { return [7]float32{b.x, b.y, b.w, b.h, b.alpha, b.scale, b.rot} }

func boxOf(v [7]float32) tbox { return tbox{v[0], v[1], v[2], v[3], v[4], v[5], v[6]} }

// What a transition moves now.
const (
	movePos uint8 = 1 << iota
	moveSize
	moveAlpha
	moveTurn
	moveAll = movePos | moveSize | moveAlpha | moveTurn
)

// groupOf is what moves each channel of a transition.
var groupOf = [7]uint8{movePos, movePos, moveSize, moveSize, moveAlpha, moveTurn, moveTurn}

// The axes a Motion collapses an element along.
const (
	collapseWidth  = 1
	collapseHeight = 2
)

// transition is what an element's Transition keeps from frame to frame,
// by the element's ID. It outlives the element while a copy of it leaves.
type transition struct {
	t ElementTransition
	// built is the last frame that built the element, and laid the last
	// that laid it out: elem is its element then, and parent the ID of
	// its parent.
	built, laid uint64
	elem        *node
	parent      uint64
	// target is where the layout put the element in frame laid, next
	// where it puts it in the frame being laid out, and shown where it
	// shows: ch move the components of moving (movePos, …) there.
	target, next, shown tbox
	ch                  [7]channel
	moving              uint8
	fresh               bool
	// collapse is the axis a Motion collapses the element along, while it
	// does.
	collapse uint8
	// The background and border colors, which move as the element
	// paints, after its styleFn set them, k of the way from colorFrom.
	colorTarget, colorFrom, colorShown [2]Color
	colorK                             channel
	colorMoving, colorSet              bool
	colorFrame                         uint64
	// ghost is the copy of the element going, which moves from gone,
	// where it showed as it went; opacity is its own, and turn its
	// transform then.
	ghost   *node
	gone    tbox
	opacity float32
	turn    Transform
	// match moves an element appearing where the one it Matches showed:
	// its box in the window, from the box that one had.
	match    [4]channel
	matching bool
}

// animation returns how the transition's changes go.
func (r *transition) animation() Animation {
	if a := r.t.Animation; a != nil {
		return *a
	}
	ease := r.t.Ease
	if ease == nil {
		ease = EaseOut
	}
	return Animation{Duration: r.duration(), Ease: ease}
}

// duration returns how long the transition's changes take, about as long
// for a spring.
func (r *transition) duration() time.Duration {
	if a := r.t.Animation; a != nil {
		if a.Duration > 0 {
			return a.Duration
		}
		if a.Spring.Duration > 0 {
			return a.Spring.Duration
		}
		if a.Spring.Stiffness == 0 && a.Spring.Damping == 0 {
			return defaultSpringDuration
		}
		m, k, _, _ := a.Spring.physics()
		return time.Duration(2 * math.Pi * math.Sqrt(m/k) * float64(time.Second))
	}
	if r.t.Duration > 0 {
		return r.t.Duration
	}
	return defaultDuration
}

// head sends the channels of move from where they are to target, as the
// transition's animation says, from now.
func (r *transition) head(target tbox, move uint8, now time.Time) {
	a := r.animation()
	to := target.vals()
	for i := range r.ch {
		if move&groupOf[i] != 0 {
			headFor(&r.ch[i], float64(to[i]), a, now)
		}
	}
	r.moving |= move
}

// headFor sends ch from where it is now to to, as a says.
func headFor(ch *channel, to float64, a Animation, now time.Time) {
	ch.advance(now)
	if to == ch.goal() && (ch.moving || ch.pending || ch.value == to) {
		return
	}
	switch {
	case a.Delay > 0:
		ch.pending, ch.pendingTo, ch.pendingAt, ch.pendingAnim = true, to, now.Add(a.Delay), a
		ch.pendingAnim.Delay = 0
	default:
		ch.pending = false
		ch.retarget(to, a)
	}
}

// follow shifts what moves that the Transition does not animate, as an
// Enter's place, by as much as the layout moved it since the frame before:
// that follows the layout at once, as it would without the Enter. A size
// scales instead, so that one collapsed opens on in proportion, never
// below nothing as the layout makes it smaller.
func (r *transition) follow(next tbox, props uint8) {
	var own uint8
	if props&propPosition != 0 {
		own |= movePos
	}
	if props&propSize != 0 {
		own |= moveSize
	}
	was, to := r.target.vals(), next.vals()
	for i := range r.ch {
		g := groupOf[i]
		if g&(movePos|moveSize) == 0 || own&g != 0 || r.moving&g == 0 {
			continue
		}
		switch d := to[i] - was[i]; {
		case d == 0:
		case g == moveSize && was[i] > 0:
			r.ch[i].scale(float64(was[i]), float64(to[i]))
		default:
			r.ch[i].shift(float64(d))
		}
	}
}

// rest puts every channel at rest at b.
func (r *transition) rest(b tbox, now time.Time) {
	for i, v := range b.vals() {
		r.ch[i] = channel{value: float64(v), target: float64(v), at: now}
	}
	r.moving, r.collapse = 0, 0
}

// sample moves the channels to now, and returns where they show the
// element going to target: the components not moving there at once.
// The moves over, the transition no longer moves.
func (r *transition) sample(target tbox, now time.Time) tbox {
	to := target.vals()
	var v [7]float32
	var still uint8 = moveAll
	for i := range r.ch {
		ch := &r.ch[i]
		if r.moving&groupOf[i] == 0 {
			*ch = channel{value: float64(to[i]), target: float64(to[i]), at: now}
		} else {
			ch.advance(now)
			if ch.moving || ch.pending {
				still &^= groupOf[i]
			}
		}
		v[i] = float32(ch.value)
	}
	r.moving &^= still
	if r.moving == 0 {
		r.collapse = 0
		return target
	}
	return boxOf(v)
}

// What a transition animates.
const (
	propPosition uint8 = 1 << iota
	propSize
	propColors
)

func (r *transition) properties() uint8 {
	t := &r.t
	if !t.Position && !t.Size && !t.Colors {
		return propPosition | propSize | propColors
	}
	var p uint8
	if t.Position {
		p |= propPosition
	}
	if t.Size {
		p |= propSize
	}
	if t.Colors {
		p |= propColors
	}
	return p
}

func lerp(a, b, k float32) float32 { return a + (b-a)*k }

// moved returns b moved as m says, collapsed along axis, or, without
// motion (still), only faded.
func (b tbox) moved(m *Motion, axis uint8, still bool) tbox {
	b.alpha = max(0, min(m.Opacity, 1))
	if still {
		return b
	}
	b.x += m.X
	b.y += m.Y
	if m.Scale != 0 {
		b.scale = m.Scale
	}
	b.rot = m.Rotate
	switch axis {
	case collapseWidth:
		b.w = 0
	case collapseHeight:
		b.h = 0
	}
	return b
}

// fades reports whether m makes an element appear or go even without
// motion: it fades, and collapses nothing, which would move its siblings.
func (m *Motion) fades() bool { return m.Opacity < 1 && !m.Collapse }

// collapseAxis returns the axis a Motion collapses a child of parent along.
func collapseAxis(m *Motion, parent *node) uint8 {
	switch {
	case !m.Collapse:
		return 0
	case parent.row && parent.list == nil && !parent.grid:
		return collapseWidth
	}
	return collapseHeight
}

// animateLayout moves the elements given a Transition from where the last
// frame showed them to where the layout put them, and the copies of those
// gone with an Exit: after the layout of a frame in a window w×h, while
// boxes are relative to their parents.
func (rt *engine) animateLayout(root *node, w, h float32) {
	c := root.c
	// Changes that resizing the window makes do not animate.
	resized := w != rt.laidW || h != rt.laidH || rt.instant()
	rt.laidW, rt.laidH = w, h
	uses := c.transitions
	rt.exitsBuilt = false
	if len(uses) == 0 && len(rt.trans) == 0 {
		return
	}
	if rt.trans == nil {
		rt.trans = map[uint64]*transition{}
	}
	now := c.now
	still := rt.preferences().ReduceMotion
	// Parents first: one resizing lays out its children anew, which then
	// move themselves.
	uses = c.byDepth(uses)
	n := 0
	for _, u := range uses {
		e := u.e
		if e.parent == nil {
			continue // the root's
		}
		r := rt.trans[e.id]
		if r == nil {
			r = &transition{}
			rt.trans[e.id] = r
		} else if r.built == rt.frame {
			if r.elem == e {
				r.t = u.t // the element's last Transition counts
			}
			continue
		}
		r.fresh = r.laid+1 != rt.frame
		r.t, r.elem, r.parent, r.built = u.t, e, e.parent.id, rt.frame
		r.next = tbox{e.x, e.y, e.w, e.h, 1, 1, 0}
		if g := r.ghost; g != nil {
			// Built again as a copy of it was going: it comes back from
			// where the copy shows, as fast as it went.
			r.ghost, r.fresh, r.collapse = nil, false, 0
			r.head(r.next, moveAll, now)
		}
		e.st.trec = r
		if u.t.Exit != nil {
			rt.exitsBuilt = true
		}
		u.r = r
		uses[n] = u
		n++
	}
	for _, u := range uses[:n] {
		e, r := u.e, u.r
		next := r.next
		switch {
		case r.fresh:
			r.rest(next, now)
			// An element appearing in a parent that was there comes in;
			// whole new parts of the interface, as a page, appear at once.
			if m := r.t.Enter; m != nil && (!still || m.fades()) && !resized && e.st.born == rt.frame && e.parent.st.born != rt.frame {
				var axis uint8
				if !still {
					axis = collapseAxis(m, e.parent)
				}
				r.rest(next.moved(m, axis, still), now)
				r.collapse = axis
				r.head(next, moveAll, now)
			}
		case resized || still && r.moving&^moveAlpha != 0:
			r.rest(next, now)
		default:
			props := r.properties()
			var changed uint8
			if props&propPosition != 0 && (next.x != r.target.x || next.y != r.target.y) {
				changed |= movePos
			}
			if props&propSize != 0 && (next.w != r.target.w || next.h != r.target.h) {
				changed |= moveSize
			}
			if still {
				changed = 0
			}
			if r.laid+1 == rt.frame {
				// Not back as its copy went: that heads for next already.
				r.follow(next, props)
			}
			if changed != 0 {
				r.head(next, changed, now)
			}
		}
		r.target, r.laid = next, rt.frame
		shown := next
		if r.moving != 0 {
			shown = r.sample(next, now)
		}
		if r.moving != 0 {
			rt.animating = true
		} else {
			r.rest(next, now)
		}
		r.shown = shown
		r.apply(e, shown)
	}
	rt.animateExits(root, now, still, resized)
}

// matchRec is where the element of a Match showed in a frame, in the
// window, and its colors.
type matchRec struct {
	frame, id uint64
	box       Rect
	colors    [2]Color
}

// animateMatches moves the elements of a Transition's Match appearing
// from where the element of their Match showed in the frame before, by a
// transform: once the frame's boxes are in the window.
func (rt *engine) animateMatches() {
	for m, rec := range rt.matched {
		if rec.frame+1 < rt.frame {
			delete(rt.matched, m)
		}
	}
	now := rt.c.now
	still := rt.preferences().ReduceMotion
	for _, u := range rt.c.transitions {
		m, e := u.t.Match, u.e
		r := e.st.trec
		if m == nil || r == nil || r.elem != e || r.built != rt.frame {
			continue
		}
		box := Rect{e.x, e.y, e.w, e.h}
		rec := rt.matched[m]
		switch {
		case still || rt.instant():
			r.matching = false
		case r.fresh && rec != nil && rec.id != e.id && rec.frame+1 == rt.frame:
			from := [4]float32{rec.box.X, rec.box.Y, rec.box.W, rec.box.H}
			for i := range r.match {
				r.match[i] = channel{value: float64(from[i]), target: float64(from[i]), at: now}
			}
			r.matching = true
			if rec.colors != ([2]Color{}) {
				// Its colors come from the other's too.
				r.colorSet, r.colorTarget, r.colorShown, r.colorFrame = true, rec.colors, rec.colors, rt.frame-1
			}
		}
		shown := box
		if r.matching {
			a := r.animation()
			to := [4]float32{box.X, box.Y, box.W, box.H}
			var v [4]float32
			moving := false
			for i := range r.match {
				ch := &r.match[i]
				headFor(ch, float64(to[i]), a, now)
				ch.advance(now)
				moving = moving || ch.moving || ch.pending
				v[i] = float32(ch.value)
			}
			if !moving {
				r.matching = false
			} else {
				shown = Rect{v[0], v[1], max(v[2], 0), max(v[3], 0)}
				e.showAt(shown)
				rt.animating = true
			}
		}
		if rec == nil {
			if rt.matched == nil {
				rt.matched = map[any]*matchRec{}
			}
			rec = &matchRec{}
			rt.matched[m] = rec
		}
		if rec.id != e.id {
			rec.colors = [2]Color{}
		}
		rec.frame, rec.id, rec.box = rt.frame, e.id, shown
	}
}

// showAt shows e in the window's box r, its content laid out at r's size
// as a Transition's Size lays it out on the way: once the frame's boxes are
// in the window.
func (e *node) showAt(r Rect) {
	if r.W == e.w && r.H == e.h {
		shift(e, r.X-e.x, r.Y-e.y)
		return
	}
	layoutBox(e, r.W, r.H)
	e.x, e.y = r.X, r.Y
	place(e, 0, 0)
}

// byDepth returns the transitions asked for in the order of their
// elements' depth, those of one depth in the order asked: a counting sort,
// into a slice the context reuses.
func (c *context) byDepth(uses []transitionUse) []transitionUse {
	deepest := 0
	for i := range uses {
		deepest = max(deepest, uses[i].e.depth)
	}
	starts := c.depthStarts
	if cap(starts) < deepest+2 {
		starts = make([]int, deepest+2)
	} else {
		starts = starts[:deepest+2]
		clear(starts)
	}
	for i := range uses {
		starts[uses[i].e.depth+1]++
	}
	for d := 1; d < len(starts); d++ {
		starts[d] += starts[d-1]
	}
	sorted := slices.Grow(c.sortedUses[:0], len(uses))[:len(uses)]
	for _, u := range uses {
		sorted[starts[u.e.depth]] = u
		starts[u.e.depth]++
	}
	c.depthStarts, c.sortedUses = starts, sorted
	return sorted
}

// apply shows e at s.
func (r *transition) apply(e *node, s tbox) {
	if s.w != e.w || s.h != e.h {
		if r.collapse != 0 {
			// Collapsing: the content keeps its layout, clipped.
			e.w, e.h = max(s.w, 0), max(s.h, 0)
			e.flags |= flagClip
		} else {
			layoutBox(e, max(s.w, 0), max(s.h, 0))
		}
	}
	e.x, e.y = s.x, s.y
	if s.alpha < 1 {
		o := float32(1)
		if e.opacitySet {
			o = e.opacity
		}
		e.opacity, e.opacitySet = o*max(s.alpha, 0), true
	}
	if t, ok := s.turn(); ok {
		e.transform = e.transform.Then(t)
	}
}

// turn returns the transform of s's scale and rotation, unless it has
// none.
func (s tbox) turn() (Transform, bool) {
	if s.scale == 1 && s.rot == 0 {
		return Transform{}, false
	}
	return Scaling(s.scale, s.scale).Then(Rotation(s.rot)), true
}

// animateExits moves the copies of the elements gone with an Exit, under
// their parents, and copies those that just went, from the frame before,
// which the engine kept for them.
func (rt *engine) animateExits(root *node, now time.Time, still, instant bool) {
	var byID map[uint64]*node
	for id, r := range rt.trans {
		if r.built == rt.frame {
			continue
		}
		if r.ghost == nil && (r.t.Exit == nil || still && !r.t.Exit.fades() || instant && rt.instant() || r.laid+1 != rt.frame) {
			delete(rt.trans, id)
			continue
		}
		if byID == nil {
			byID = rt.elementsByID(root)
		}
		p := byID[r.parent]
		if p == nil || p.leaving != 0 {
			// Its parent went too.
			delete(rt.trans, id)
			continue
		}
		if r.ghost == nil {
			e := r.elem
			r.ghost = ghostOf(e)
			r.elem = nil
			r.ghost.leaving = 1
			if e.flags&flagAbsolute != 0 {
				r.ghost.leaving = 2
			}
			r.ghost.flags |= flagAbsolute
			r.opacity = 1
			if e.opacitySet {
				r.opacity = e.opacity / max(r.shown.alpha, 0.001)
			}
			r.turn = r.ghost.transform
			r.gone, r.collapse = r.shown, 0
			if !still {
				r.collapse = collapseAxis(r.t.Exit, p)
			}
			// From where it showed, as fast as it went.
			r.head(r.gone.moved(r.t.Exit, r.collapse, still), moveAll, now)
		}
		s := r.sample(r.gone.moved(r.t.Exit, r.collapse, still), now)
		if r.moving == 0 {
			delete(rt.trans, id)
			continue
		}
		g := r.ghost
		g.next = nil
		p.add(g)
		p.ghosts = true
		layoutBox(g, r.gone.w, r.gone.h)
		if s.w != r.gone.w || s.h != r.gone.h {
			g.w, g.h = max(s.w, 0), max(s.h, 0)
			g.flags |= flagClip
		}
		g.x, g.y = s.x, s.y
		g.opacity, g.opacitySet = r.opacity*max(s.alpha, 0), true
		g.transform = r.turn
		if t, ok := s.turn(); ok {
			g.transform = g.transform.Then(t)
		}
		r.shown = s
		rt.animating = true
	}
}

// elementsByID returns the elements of the frame by ID, in a map the
// engine reuses.
func (rt *engine) elementsByID(root *node) map[uint64]*node {
	if rt.byID == nil {
		rt.byID = map[uint64]*node{}
	}
	clear(rt.byID)
	var walk func(e *node)
	walk = func(e *node) {
		rt.byID[e.id] = e
		for c := e.first; c != nil; c = c.next {
			walk(c)
		}
	}
	walk(root)
	return rt.byID
}

// ghostOf copies e, an element of the frame before, and the elements inside
// it, for its exit transition. The copy only shows: it refers to no other
// element nor to the app's state, has states of its own, and is inert.
func ghostOf(e *node) *node {
	g := new(node)
	*g = *e
	st := *e.st
	st.trec = nil
	g.st = &st
	g.parent, g.first, g.last, g.next, g.nchild = nil, nil, nil, nil, 0
	g.shadows, g.cols, g.rows, g.frags = slices.Clone(e.shadows), slices.Clone(e.cols), slices.Clone(e.rows), slices.Clone(e.frags)
	g.track, g.popover, g.list, g.rowsOf, g.overflow = nil, nil, nil, nil, nil
	g.activeDescendant, g.nameFrom, g.field, g.form = nil, nil, nil, nil
	g.followX, g.colFit, g.gridFit = nil, nil, nil
	g.inputFn, g.ghosts = nil, false
	g.poseSet = false // it keeps the pose it had
	g.tl, g.nmeasure, g.leafOK = nil, 0, false
	g.flags = g.flags&^(flagModal|flagPage|flagFocusTarget) | flagInert
	for c := e.first; c != nil; c = c.next {
		if c.leaving == 0 {
			g.add(ghostOf(c))
		}
	}
	return g
}

// animateColors moves e's background and border colors, once its styleFn
// set them, as its Transition asks: even while the desktop asks for less
// motion, as changing colors moves nothing.
func (rt *engine) animateColors(e *node, r *transition) {
	if r.properties()&propColors == 0 {
		return
	}
	now := rt.c.now
	if r.colorFrame != rt.frame {
		target := [2]Color{e.bg, e.borderC}
		switch {
		case !r.colorSet || r.colorFrame+1 != rt.frame:
			// Not painted in the frame before: its colors show at once.
			r.colorMoving = false
		case target != r.colorTarget && rt.instant():
			r.colorMoving = false
		case target != r.colorTarget:
			r.colorFrom, r.colorMoving = r.colorShown, true
			r.colorK = channel{at: now}
			headFor(&r.colorK, 1, r.animation(), now)
		}
		r.colorSet, r.colorTarget, r.colorFrame = true, target, rt.frame
		r.colorShown = target
		if r.colorMoving {
			r.colorK.advance(now)
			if !r.colorK.moving && !r.colorK.pending {
				r.colorMoving = false
			} else {
				k := max(0, min(float32(r.colorK.value), 1))
				r.colorShown = [2]Color{mixPremultiplied(r.colorFrom[0], target[0], k), mixPremultiplied(r.colorFrom[1], target[1], k)}
				rt.animating = true
			}
		}
		if m := r.t.Match; m != nil {
			if rec := rt.matched[m]; rec != nil && rec.id == e.id {
				rec.colors = r.colorShown
			}
		}
	}
	e.bg, e.borderC = r.colorShown[0], r.colorShown[1]
}

// mixPremultiplied returns the color k of the way from a to b, mixed with
// premultiplied alpha, so that a color fading from transparent keeps its
// hue.
func mixPremultiplied(a, b Color, k float32) Color {
	aa, ba := float32(a.A)/255, float32(b.A)/255
	alpha := lerp(aa, ba, k)
	if alpha <= 0 {
		return Color{}
	}
	if a.wide.ok() || b.wide.ok() {
		// Premultiplied, b weighs this much of the straight colors.
		c := mixWide(a, b, k*ba/alpha)
		c.A = alphaByte(alpha)
		return c
	}
	ch := func(x, y uint8) uint8 {
		v := lerp(float32(x)*aa, float32(y)*ba, k) / alpha
		return uint8(max(0, min(v+0.5, 255)))
	}
	return Color{R: ch(a.R, b.R), G: ch(a.G, b.G), B: ch(a.B, b.B), A: alphaByte(alpha)}
}
