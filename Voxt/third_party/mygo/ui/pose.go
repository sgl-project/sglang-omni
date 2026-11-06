package ui

import "github.com/egoist/mygo/internal/scene"

// Pose is where an element shows, from its place in the layout, which
// AnimatePose moves it to: moved by X and Y DIPs, scaled by Scale around
// its TransformOrigin (0 keeps its size, as 1 does), turned by Rotate
// degrees clockwise, and faded by Fade, from 0, opaque, to 1, gone.
type Pose struct {
	X, Y, Scale, Rotate, Fade float32
}

// vals returns the components of p, in the order of a pose's channels.
func (p Pose) vals() [5]float32 {
	s := p.Scale
	if s == 0 {
		s = 1
	}
	return [5]float32{p.X, p.Y, s, p.Rotate, max(0, min(p.Fade, 1))}
}

// AnimatePose moves the element to p as a says, after its own transform
// (Transform), and fades it, and returns the element. The element keeps
// going in frames that only paint, without building the view, as Core
// Animation and the Web Animations API run transforms and opacity: the
// cheapest motion there is, which a view large or slow to build keeps
// smooth. The pointer, carets and overlays follow it as they follow
// transforms. It starts at the first pose. While the desktop asks for less
// motion, it moves at once, and only fades smoothly.
//
//	card.AnimatePose(ui.Pose{Y: lift, Scale: 1.02}, ui.Snappy)
func (e *node) AnimatePose(p Pose, a Animation) *node {
	for _, v := range p.vals() {
		finiteTransform(v)
	}
	if a.Duration < 0 || a.Delay < 0 {
		panic("ui: animation durations must not be negative")
	}
	if a.Duration == 0 {
		a.Spring.physics()
	}
	e.pose, e.poseAnim, e.poseSet = p, a, true
	return e
}

// poseRec is what an element's AnimatePose keeps from frame to frame: the
// channels of its pose, and the last frame that posed it.
type poseRec struct {
	ch    [5]channel
	frame uint64
}

// moving reports whether the pose still moves.
func (r *poseRec) moving() bool {
	for i := range r.ch {
		if r.ch[i].moving || r.ch[i].pending {
			return true
		}
	}
	return false
}

// samplePose heads e's pose for the one the view asked for, as the frame
// commits, and sets where it shows.
func (rt *engine) samplePose(e *node) {
	st, now := e.st, rt.c.now
	to := e.pose.vals()
	r := st.pose
	if r == nil || r.frame+1 < rt.frame {
		// Not posed in the frame before: it shows there at once.
		r = &poseRec{}
		st.pose = r
		for i, v := range to {
			r.ch[i] = channel{value: float64(v), target: float64(v), at: now}
		}
	}
	r.frame = rt.frame
	still, instant := rt.preferences().ReduceMotion, rt.instant()
	for i := range r.ch {
		ch := &r.ch[i]
		if instant || still && i < 4 {
			*ch = channel{value: float64(to[i]), target: float64(to[i]), at: now}
			continue
		}
		headFor(ch, float64(to[i]), e.poseAnim, now)
	}
	rt.posed = append(rt.posed, e)
	rt.showPose(e, r)
}

// showPose sets where e shows as its pose's channels are now, and asks
// for frames that paint while they move.
func (rt *engine) showPose(e *node, r *poseRec) {
	now := rt.c.now
	var v [5]float32
	for i := range r.ch {
		// A channel at rest stays where it is until headFor moves it on,
		// which advances it first.
		if ch := &r.ch[i]; ch.moving || ch.pending {
			ch.advance(now)
		}
		v[i] = float32(r.ch[i].value)
	}
	// Draggable moves it along with its pose.
	d := e.st.dragRec
	if d.current(rt) {
		moving := d.moving()
		d.advance(now)
		if moving && !d.moving() {
			// It came to rest in a frame that only paints: the view
			// builds again, to see it there (Transitioning).
			rt.animating = true
		}
		p := d.shown()
		v[0] += p.X
		v[1] += p.Y
	}
	e.poseM = scene.Affine{}
	if v[0] != 0 || v[1] != 0 || v[2] != 1 || v[3] != 0 {
		e.poseM = scene.Translation(v[0], v[1]).Mul(scene.Rotation(v[3])).Mul(scene.Scaling(v[2], v[2]))
	}
	e.poseFade = max(0, min(v[4], 1))
	if r.moving() || d.current(rt) && d.moving() {
		rt.repainting = true
	}
}

// repose moves the posed elements of the last frame built on to the time
// of a frame that only paints, with what is inside them. It moves every
// pose first, and then sets the transforms to the window again from the
// outermost elements that moved, once for what is inside them: poses
// nested in moving poses would otherwise set those inside again and again.
func (rt *engine) repose() {
	for _, e := range rt.posed {
		e.reposed = false
		d := e.st.dragRec
		if r := e.st.pose; r != nil && (r.moving() || d.current(rt) && d.moving()) {
			rt.showPose(e, r)
			e.reposed = true
		}
	}
	for _, e := range rt.posed {
		if e.reposed && !e.insideReposed() {
			e.updateWorld()
		}
	}
}

// insideReposed reports whether an element around e moved its pose in
// this frame.
func (e *node) insideReposed() bool {
	for p := e.parent; p != nil; p = p.parent {
		if p.reposed {
			return true
		}
		if p == e.c.overlay {
			// The overlay's place does not follow those around it.
			return false
		}
	}
	return false
}

// updateWorld sets the transform from e to the window again, and those of
// the elements inside it, as commitElement sets them.
func (e *node) updateWorld() {
	w := e.transformAt(e.x, e.y)
	if e.parent != nil && e != e.c.overlay {
		w = e.parent.world.Mul(w)
	}
	e.world = w
	if e.st != nil {
		e.st.world = w
	}
	for c := e.first; c != nil; c = c.next {
		c.updateWorld()
	}
}
