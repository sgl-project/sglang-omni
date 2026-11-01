package ui

import (
	"math"
	"time"

	"github.com/egoist/mygo/internal/scene"
)

// DraggableOptions are how an element that the pointer drags moves
// (Element.Draggable), and where it comes to rest once let go.
type DraggableOptions struct {
	// Axis is the direction the element moves in: Horizontal, Vertical,
	// or, when zero, both.
	Axis Orientation
	// Bounds, if set, holds the element's offset within it at rest: from
	// X to X+W across, and from Y to Y+H down. The pointer drags the
	// element past them less and less far, as iOS holds content pulled
	// past its edges (Rubberband), and it goes back within them once let
	// go; Rigid stops it at them.
	Bounds *Rect
	Rigid  bool
	// Snap, if set, chooses where the element comes to rest once let go,
	// from where the throw would carry it (Project, held within Bounds):
	// as the nearest of a sheet's heights or of a pager's pages (Nearest).
	// A Snap returning its point lets the element glide there. Without
	// Snap, the element rests where the pointer lets it go. What Snap
	// returns along an axis the element does not move along is ignored.
	Snap func(Point) Point
	// Animation is how the element goes to rest, and to an offset the app
	// sets: Smooth's spring when zero. A spring starts at the pointer's
	// velocity as it lets go.
	Animation Animation
}

// Nearest returns a Snap (DraggableOptions) that chooses the nearest of
// points, as the heights a sheet rests at:
//
//	Snap: ui.Nearest(ui.Point{}, ui.Point{Y: 200}, ui.Point{Y: 400}),
func Nearest(points ...Point) func(Point) Point {
	if len(points) == 0 {
		panic("ui: Nearest needs points")
	}
	points = append([]Point(nil), points...)
	return func(p Point) Point {
		best, far := points[0], float32(math.Inf(1))
		for _, q := range points {
			dx, dy := q.X-p.X, q.Y-p.Y
			if d := dx*dx + dy*dy; d < far {
				best, far = q, d
			}
		}
		return best
	}
}

// Draggable lets the pointer move the element from its place in the
// layout, by *offset, and returns the element. As the pointer drags it,
// the element follows the pointer, along its parent's axes, and *offset
// with it; once the pointer lets go, *offset is where the element comes to
// rest, as o says, and it goes there. Setting *offset moves the element
// there as well, as o.Animation says. A nil offset is kept in the element.
//
// The element moves after its own transform (Transform) and its pose
// (AnimatePose), and goes to rest in frames that only paint, without
// building the view, as AnimatePose moves it. Dragging reports that the
// pointer drags it, once the pointer moved a few DIPs, Transitioning that
// it moves, and DragOffset where it shows. A press dragging it is no
// click; a press catches it as it moves. While the desktop asks for less
// motion (Preferences.ReduceMotion), it goes to rest at once.
//
//	sheet.Draggable(&app.sheet, ui.DraggableOptions{
//		Axis:   ui.Vertical,
//		Bounds: &ui.Rect{H: 400},
//		Snap:   ui.Nearest(ui.Point{}, ui.Point{Y: 200}, ui.Point{Y: 400}),
//	})
func (e *node) Draggable(offset *Point, o DraggableOptions) *node {
	axes := o.Axis
	if axes&^(Horizontal|Vertical) != 0 {
		panic("ui: Draggable's Axis must be Horizontal, Vertical or both")
	}
	if axes == 0 {
		axes = Horizontal | Vertical
	}
	if b := o.Bounds; b != nil {
		finiteMotion("drag bounds", float64(b.X), float64(b.Y), float64(b.W), float64(b.H))
		if b.W < 0 || b.H < 0 {
			panic("ui: Draggable's Bounds must not have a negative size")
		}
	}
	a := o.Animation
	if a.Duration < 0 || a.Delay < 0 {
		panic("ui: animation durations must not be negative")
	}
	if a.Duration == 0 {
		a.Spring.physics() // panics now for invalid options
	}
	if !e.hasState() {
		return e
	}
	st, rt, now := e.st, e.c.rt, e.c.now
	r := st.dragRec
	if r == nil {
		r = &dragRec{}
		st.dragRec = r
	}
	if offset == nil {
		offset = &r.own
	}
	finiteMotion("drag offset", float64(offset.X), float64(offset.Y))
	if r.frame == 0 || r.frame+1 < rt.frame {
		// Not dragged in the frame before: it shows at its offset at once.
		*r = dragRec{own: r.own}
		r.hold(*offset, now)
	}
	r.frame, r.axes, r.rigid = rt.frame, axes, o.Rigid
	r.bounded = o.Bounds != nil
	if r.bounded {
		r.bounds = *o.Bounds
	}
	e.flags |= flagDraggable
	e.poseSet = true
	r.advance(now)
	shown := r.shown()
	still := rt.preferences().ReduceMotion || rt.instant()
	set := func(p Point) {
		if *offset != p {
			*offset = p
			rt.consumed = true
		}
	}

	held := st.pressed && rt.pressButton == 0 && !e.disabled()
	// A press that began and ended since the last frame.
	ended := !st.pressed && st.released && rt.pressButton == 0
	if !r.pressing && (held || ended) {
		// The press holds the element where it shows.
		r.pressing, r.caught, r.from = true, r.moving(), shown
		r.hold(shown, now)
		if r.caught {
			set(shown)
		}
	}
	space, invertible := e.dragSpace()
	if r.pressing && st.drag.far && invertible {
		// It follows the pointer, from where the press took it.
		d := &st.drag
		x0, y0 := space.Point(d.ox, d.oy)
		x1, y1 := space.Point(d.ox+d.wx, d.oy+d.wy)
		shown = r.band(e, Point{r.from.X + x1 - x0, r.from.Y + y1 - y0})
		r.hold(shown, now)
		set(shown)
	}
	if r.pressing && !held {
		r.pressing = false
		if st.drag.far || r.caught {
			var v Point
			if st.drag.far && ended && invertible {
				d := &st.drag
				v.X, v.Y = space.VectorAt(d.ox+d.wx, d.oy+d.wy, st.releaseWX, st.releaseWY)
				v = flingVelocity(v)
			}
			to := shown
			if o.Snap != nil {
				to = o.Snap(r.clamp(Point{Project(shown.X, v.X), Project(shown.Y, v.Y)}, shown))
				finiteMotion("drag offset", float64(to.X), float64(to.Y))
			}
			to = r.clamp(to, shown)
			set(to)
			r.settle(to, v, a, still, now)
			return e
		}
	}
	if !r.pressing {
		// It goes where the app moves it, within its bounds.
		to := r.clamp(*offset, *offset)
		set(to)
		r.head(to, a, still, now)
	}
	return e
}

// DragOffset returns where Draggable shows the element in this frame,
// from its place in the layout: following the pointer, or on its way to
// rest. Reading it builds the view in every frame the element moves in,
// as AnimateTo does, which the element otherwise moves in without: for
// what follows the element, as a backdrop fading as a sheet goes down.
// Call it after Draggable.
func (e *node) DragOffset() Point {
	if !e.hasState() {
		return Point{}
	}
	r := e.st.dragRec
	if r == nil || r.frame+1 < e.c.rt.frame {
		return Point{}
	}
	r.advance(e.c.now)
	if r.moving() {
		e.c.AnimationFrame()
	}
	return r.shown()
}

// maxFling is the fastest a throw goes, in DIPs per second, as Flutter's
// and Android's flings: moves the window delivers at once, after a slow
// frame, would otherwise throw the element far away.
const maxFling = 8000

// flingVelocity returns v, no faster than maxFling.
func flingVelocity(v Point) Point {
	n := math.Hypot(float64(v.X), float64(v.Y))
	if math.IsNaN(n) || math.IsInf(n, 0) {
		return Point{}
	}
	if n > maxFling {
		k := float32(maxFling / n)
		v = Point{v.X * k, v.Y * k}
	}
	return v
}

// dragSpace returns the map from the window to the axes that the
// element's offset moves it along, those of its parent, as the last frame
// showed them, and whether there is one.
func (e *node) dragSpace() (scene.Affine, bool) {
	if e.parent == nil || e == e.c.overlay || e.parent.st == nil {
		return scene.Affine{}, true
	}
	return e.parent.st.world.Inverse()
}

// dragRec is what Draggable keeps of an element from frame to frame: the
// channels of its offset, the last frame built with it, the axes it
// moves along, its bounds, and the press holding it, which caught it
// moving when caught, and took it at from.
type dragRec struct {
	ch               [2]channel
	frame            uint64
	axes             Orientation
	bounds           Rect
	bounded, rigid   bool
	pressing, caught bool
	from             Point
	// own is the offset of a Draggable without one.
	own Point
}

// dragged reports whether the press of s dragged the element of r, as
// Draggable moves it.
func (r *dragRec) dragged(rt *engine, s *state) bool {
	return r != nil && r.frame+1 >= rt.frame && s.drag.far && rt.pressButton == 0
}

// current reports whether the element of r was built with Draggable in
// the frame rt shows.
func (r *dragRec) current(rt *engine) bool { return r != nil && r.frame == rt.frame }

func (r *dragRec) at() Point { return Point{float32(r.ch[0].value), float32(r.ch[1].value)} }

// shown returns where the element shows: where its channels are, held
// within its bounds when they are rigid.
func (r *dragRec) shown() Point {
	p := r.at()
	if r.rigid {
		p = r.clamp(p, p)
	}
	return p
}

func (r *dragRec) moving() bool {
	return r.ch[0].moving || r.ch[0].pending || r.ch[1].moving || r.ch[1].pending
}

// advance moves the channels on to now.
func (r *dragRec) advance(now time.Time) {
	for i := range r.ch {
		if ch := &r.ch[i]; ch.moving || ch.pending {
			ch.advance(now)
		}
	}
}

// hold puts the channels at rest at p.
func (r *dragRec) hold(p Point, now time.Time) {
	r.ch[0] = channel{value: float64(p.X), target: float64(p.X), at: now}
	r.ch[1] = channel{value: float64(p.Y), target: float64(p.Y), at: now}
}

// clamp returns p held within the bounds along the axes the element
// moves along, and as rest is along the others.
func (r *dragRec) clamp(p, rest Point) Point {
	if r.axes&Horizontal == 0 {
		p.X = rest.X
	} else if r.bounded {
		p.X = max(r.bounds.X, min(p.X, r.bounds.X+r.bounds.W))
	}
	if r.axes&Vertical == 0 {
		p.Y = rest.Y
	} else if r.bounded {
		p.Y = max(r.bounds.Y, min(p.Y, r.bounds.Y+r.bounds.H))
	}
	return p
}

// band returns where the pointer takes the element to p: as from along
// the axes it does not move along, and past its bounds less and less far,
// as on a rubber band as long as its parent, unless they are rigid.
func (r *dragRec) band(e *node, p Point) Point {
	if r.axes&Horizontal == 0 {
		p.X = r.from.X
	}
	if r.axes&Vertical == 0 {
		p.Y = r.from.Y
	}
	if !r.bounded || r.rigid {
		return r.clamp(p, p)
	}
	w, h := e.st.w, e.st.h
	if e.parent != nil && e.parent.st != nil {
		w, h = e.parent.st.w, e.parent.st.h
	}
	b := r.bounds
	if r.axes&Horizontal != 0 {
		p.X = Rubberband(p.X, b.X, b.X+b.W, w)
	}
	if r.axes&Vertical != 0 {
		p.Y = Rubberband(p.Y, b.Y, b.Y+b.H, h)
	}
	return p
}

// settle sends the channels from where the pointer let the element go to
// to, at velocity v.
func (r *dragRec) settle(to, v Point, a Animation, still bool, now time.Time) {
	a.Delay = 0
	tos, vs := [2]float32{to.X, to.Y}, [2]float32{v.X, v.Y}
	for i := range r.ch {
		ch, t := &r.ch[i], float64(tos[i])
		switch {
		case t == ch.value:
			// Already there: a throw along it goes nowhere.
		case still:
			*ch = channel{value: t, target: t, at: now}
		default:
			b := a
			b.Spring.Velocity = vs[i]
			ch.at = now
			ch.retarget(t, b)
		}
	}
}

// head sends the channels from where they are to to, as a says, or
// there at once when still.
func (r *dragRec) head(to Point, a Animation, still bool, now time.Time) {
	tos := [2]float32{to.X, to.Y}
	for i := range r.ch {
		ch, t := &r.ch[i], float64(tos[i])
		switch {
		case still:
			if ch.moving || ch.pending || ch.value != t {
				*ch = channel{value: t, target: t, at: now}
			}
		case t != ch.goal():
			headFor(ch, t, a, now)
		}
	}
}
