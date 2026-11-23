package ui

import (
	"math"
	"slices"
	"time"
)

// Animation says how a value goes to a new target: along an easing curve
// for Duration, or, when Duration is zero, with a spring, which keeps the
// value's velocity when the target changes again on the way. The zero
// Animation is Smooth's spring:
//
//	x := card.AnimateTo("x", target, ui.Snappy)
//	fade := card.AnimateTo("fade", target, ui.Animation{Duration: 150 * time.Millisecond, Ease: ui.EaseInOut})
type Animation struct {
	// Duration and Ease make a curve: Ease, EaseOut when nil, over
	// Duration.
	Duration time.Duration
	Ease     Easing
	// Spring is the spring of an Animation without a Duration.
	Spring SpringOptions
	// Delay is how long the value keeps going as it was after its target
	// changes, before it heads for the new one, as a list's rows do one
	// after another with Delay: time.Duration(i) * 30 * time.Millisecond.
	Delay time.Duration
}

// Springs that settle without bouncing, with a little bounce, and with a
// visible one, as SwiftUI's of the same names; Smooth is the zero
// Animation's.
var (
	Smooth = Animation{Spring: SpringOptions{Duration: defaultSpringDuration}}
	Snappy = Animation{Spring: SpringOptions{Duration: defaultSpringDuration, Bounce: 0.15}}
	Bouncy = Animation{Spring: SpringOptions{Duration: defaultSpringDuration, Bounce: 0.3}}
)

const defaultSpringDuration = 500 * time.Millisecond

// SpringOptions describe a spring, as one perceives it or physically.
//
// Duration and Bounce describe it as SwiftUI and Motion do: Duration is
// about how long it takes to arrive (500 ms when zero), and Bounce how much
// it overshoots, from 0, settling without overshooting, toward 1, bouncing
// for ever; below 0 it arrives more slowly still.
//
// Mass, Stiffness and Damping describe it physically instead, as react
// spring does (1, 170 and 26 when zero), when Stiffness or Damping is set;
// setting them with Duration or Bounce panics.
//
// Precision is how close to its target, in the value's units, the spring
// rests there: 0.001 when zero. Velocity, in value units per second, is the
// velocity the value takes when its target changes in the call, as that
// of a pointer letting go of what it dragged (DragEnded); zero keeps the
// value's own.
type SpringOptions struct {
	Duration                 time.Duration
	Bounce                   float64
	Mass, Stiffness, Damping float64
	Precision                float64
	Velocity                 float32
}

// physics returns the mass, stiffness and damping of o, and the precision.
func (o SpringOptions) physics() (m, k, c, eps float64) {
	finiteMotion("spring", o.Bounce, o.Mass, o.Stiffness, o.Damping, o.Precision, float64(o.Velocity))
	if o.Mass < 0 || o.Stiffness < 0 || o.Damping < 0 || o.Precision < 0 || o.Duration < 0 {
		panic("ui: spring options must not be negative")
	}
	m, eps = o.Mass, o.Precision
	if m == 0 {
		m = 1
	}
	if eps == 0 {
		eps = 0.001
	}
	if o.Stiffness != 0 || o.Damping != 0 {
		if o.Duration != 0 || o.Bounce != 0 {
			panic("ui: a spring takes Duration and Bounce or Stiffness and Damping, not both")
		}
		k, c = o.Stiffness, o.Damping
		if k == 0 {
			k = 170
		}
		if c == 0 {
			c = 26
		}
	} else {
		if o.Bounce <= -1 || o.Bounce >= 1 {
			panic("ui: spring Bounce must be between -1 and 1")
		}
		d := o.Duration
		if d == 0 {
			d = defaultSpringDuration
		}
		// SwiftUI's: the period of the undamped spring is Duration, and
		// the damping ratio 1 - Bounce, or 1 / (1 + Bounce) below 0.
		w := 2 * math.Pi / d.Seconds()
		ratio := 1 - o.Bounce
		if o.Bounce < 0 {
			ratio = 1 / (1 + o.Bounce)
		}
		k, c = w*w*m, 2*ratio*w*m
	}
	if w, a := k/m, c/m; math.IsInf(w, 0) || math.IsInf(a, 0) || w == 0 || a == 0 || math.IsNaN(w) || math.IsNaN(a) {
		panic("ui: spring options exceed numerical range")
	}
	return m, k, c, eps
}

// finiteMotion panics unless the values of what are finite.
func finiteMotion(what string, v ...float64) {
	for _, n := range v {
		if math.IsNaN(n) || math.IsInf(n, 0) {
			panic("ui: " + what + " values must be finite")
		}
	}
}

// channel is a value that animations move, by an element and a key: along
// a curve or a spring, toward target, from value and velocity at time at.
type channel struct {
	value, velocity, target float64
	at                      time.Time
	moving                  bool
	// The spring's mass, stiffness and damping, and its precision.
	m, k, c, eps float64
	// The curve's, which went from from at start.
	curve    bool
	from     float64
	start    time.Time
	duration time.Duration
	ease     Easing
	// A target that the value goes to after a Delay, from when, with
	// pendingAnim.
	pending     bool
	pendingTo   float64
	pendingAt   time.Time
	pendingAnim Animation
	// rested is the frame in which the value came to rest at its target
	// after moving there.
	rested uint64
	// jumped tells that Jump set the value at at.
	jumped bool
}

// goal returns where the value is headed: the target it goes to after a
// delay, or its target.
func (ch *channel) goal() float64 {
	if ch.pending {
		return ch.pendingTo
	}
	return ch.target
}

// retarget heads the value, as it is at ch.at, for target as a says.
func (ch *channel) retarget(target float64, a Animation) {
	ch.target, ch.moving, ch.jumped = target, true, false
	if a.Duration > 0 {
		ch.curve, ch.from, ch.start, ch.duration, ch.ease = true, ch.value, ch.at, a.Duration, a.Ease
		if ch.ease == nil {
			ch.ease = EaseOut
		}
		return
	}
	ch.curve = false
	ch.m, ch.k, ch.c, ch.eps = a.Spring.physics()
	if a.Spring.Velocity != 0 {
		ch.velocity = float64(a.Spring.Velocity)
	}
}

// shift moves the value, and where it goes, by d, as what it is relative
// to moves: it goes on as it went.
func (ch *channel) shift(d float64) {
	ch.value += d
	ch.target += d
	ch.from += d
	ch.pendingTo += d
}

// scale scales the value, where it goes and how fast, from was to to, as
// what it is a part of grows or shrinks: it goes on as it went, in
// proportion. A value at was goes exactly to to.
func (ch *channel) scale(was, to float64) {
	ch.value = ch.value * to / was
	ch.velocity = ch.velocity * to / was
	ch.target = ch.target * to / was
	ch.from = ch.from * to / was
	ch.pendingTo = ch.pendingTo * to / was
}

// advance moves the value to now, heading for a pending target when its
// delay is over on the way.
func (ch *channel) advance(now time.Time) {
	if ch.pending && !now.Before(ch.pendingAt) {
		ch.step(ch.pendingAt)
		ch.pending = false
		ch.retarget(ch.pendingTo, ch.pendingAnim)
	}
	ch.step(now)
}

// step moves the value along its curve or spring to now.
func (ch *channel) step(now time.Time) {
	dt := now.Sub(ch.at).Seconds()
	if dt <= 0 {
		// The same time, or a clock set back.
		ch.at = now
		return
	}
	ch.at = now
	jumped := ch.jumped
	ch.jumped = false
	if !ch.moving {
		// A value that Jump moved keeps going as it did for a moment, for
		// an animation to take over.
		if !jumped || dt > 0.1 {
			ch.velocity = 0
		}
		return
	}
	if ch.curve {
		d := ch.duration.Seconds()
		t := now.Sub(ch.start).Seconds() / d
		if t >= 1 {
			ch.value, ch.velocity, ch.moving = ch.target, 0, false
			return
		}
		t = max(t, 0)
		span := ch.target - ch.from
		ch.value = ch.from + span*float64(ch.ease(float32(t)))
		// The slope of the easing, for a spring taking over.
		const h = 1e-3
		lo, hi := max(t-h, 0), min(t+h, 1)
		ch.velocity = span * float64(ch.ease(float32(hi))-ch.ease(float32(lo))) / (hi - lo) / d
		return
	}
	x, v := solveSpring(ch.value-ch.target, ch.velocity, ch.m, ch.k, ch.c, dt)
	ch.value, ch.velocity = ch.target+x, v
	if math.Abs(x) <= ch.eps && math.Abs(v) <= ch.eps*math.Sqrt(ch.k/ch.m) {
		ch.value, ch.velocity, ch.moving = ch.target, 0, false
	}
}

// solveSpring returns where a damped spring at x from its rest, going at
// v, is after dt, and its velocity there, solved analytically, so that
// frames, however far apart, follow the same trajectory.
func solveSpring(x, v, m, k, c, dt float64) (float64, float64) {
	if x == 0 && v == 0 {
		return 0, 0
	}
	w := math.Sqrt(k / m)
	a := (c / m) / 2
	switch d := a*a - w*w; {
	case math.Abs(d) < 1e-8*w*w:
		b := v + a*x
		decay := math.Exp(-a * dt)
		return (x + b*dt) * decay, (v - a*b*dt) * decay
	case d < 0:
		f := math.Sqrt(-d)
		sn, cs := math.Sincos(f * dt)
		b := (v + a*x) / f
		decay := math.Exp(-a * dt)
		y := x*cs + b*sn
		return y * decay, (f*(-x*sn+b*cs) - a*y) * decay
	default:
		f := math.Sqrt(d)
		r1 := -w * w / (a + f)
		r2 := -a - f
		c1 := (v - r2*x) / (r1 - r2)
		c2 := x - c1
		u, z := c1*math.Exp(r1*dt), c2*math.Exp(r2*dt)
		return u + z, r1*u + r2*z
	}
}

// channel returns the element's channel for key, made at value when
// there is none, and whether it was.
func (e *node) channel(key any, value float64) (*channel, bool) {
	st := e.st
	if st.motions == nil {
		st.motions = map[any]*channel{}
	}
	ch := st.motions[key]
	if ch == nil {
		ch = &channel{value: value, target: value, at: e.c.now}
		st.motions[key] = ch
		return ch, true
	}
	return ch, false
}

// AnimateTo returns a value that goes to target as a says, and keeps
// frames coming while it moves; key tells apart the animated values of
// the element, which Animate, AnimateWith and Spring share. The value
// starts at the first target (Jump sets it elsewhere first). A new target
// on the way, or a new kind of Animation, takes over from where the value
// is: a spring keeps its velocity, even that of a curve. When the desktop
// asks for less motion (Preferences.ReduceMotion), the value goes to its
// target at once.
func (e *node) AnimateTo(key any, target float32, a Animation) float32 {
	finiteMotion("animation", float64(target))
	if a.Duration < 0 || a.Delay < 0 {
		panic("ui: animation durations must not be negative")
	}
	if a.Duration == 0 {
		a.Spring.physics() // panics now for invalid options
	}
	if !e.hasState() {
		return target
	}
	to := float64(target)
	ch, made := e.channel(key, to)
	if made {
		return target
	}
	rt := e.c.rt
	now := e.c.now
	moving := ch.moving || ch.pending
	ch.advance(now)
	if rt.preferences().ReduceMotion || rt.instant() {
		if ch.moving || ch.pending || ch.value != to {
			ch.rested = rt.frame
		}
		ch.value, ch.velocity, ch.target, ch.moving, ch.pending = to, 0, to, false, false
		ch.at = now
		return target
	}
	switch {
	case to != ch.goal() && a.Delay > 0:
		ch.pending, ch.pendingTo, ch.pendingAt, ch.pendingAnim = true, to, now.Add(a.Delay), a
		ch.pendingAnim.Delay = 0
	case to != ch.goal():
		ch.pending = false
		ch.retarget(to, a)
	case ch.moving && !ch.curve && a.Duration == 0 && !ch.pending:
		// The spring's options may change on the way.
		ch.m, ch.k, ch.c, ch.eps = a.Spring.physics()
	}
	if ch.moving || ch.pending {
		e.c.AnimationFrame()
	} else if moving {
		ch.rested = rt.frame
	}
	return float32(ch.value)
}

// Spring returns a value following target with a spring, as AnimateTo
// does with Animation{Spring: options}: Smooth's spring without options.
func (e *node) Spring(key any, target float32, options ...SpringOptions) float32 {
	if len(options) > 1 {
		panic("ui: Spring takes at most one SpringOptions")
	}
	var o SpringOptions
	if len(options) > 0 {
		o = options[0]
	}
	return e.AnimateTo(key, target, Animation{Spring: o})
}

// Jump sets the value animated under key to value at once, as a drag
// moves it under the pointer, and returns it. The value's velocity
// follows the jumps from frame to frame, so that a spring taking over
// from the last (AnimateTo) keeps going the way it went.
func (e *node) Jump(key any, value float32) float32 {
	finiteMotion("animation", float64(value))
	if !e.hasState() {
		return value
	}
	v := float64(value)
	ch, made := e.channel(key, v)
	if made {
		return value
	}
	now := e.c.now
	dt := now.Sub(ch.at).Seconds()
	switch {
	case dt > 0 && dt <= 0.1 && ch.jumped:
		ch.velocity = (v - ch.value) / dt
	case dt > 0:
		ch.advance(now)
		if dt > 0.1 {
			ch.velocity = 0
		}
	}
	ch.value, ch.target, ch.moving, ch.pending, ch.jumped, ch.at = v, v, false, false, true, now
	return value
}

// Velocity returns how fast, in its units per second, the value animated
// under key goes in this frame: zero for one at rest or unknown.
func (e *node) Velocity(key any) float32 {
	if !e.hasState() {
		return 0
	}
	if ch := e.st.motions[key]; ch != nil {
		return float32(ch.velocity)
	}
	return 0
}

// Settled reports whether the value animated under key, by AnimateTo,
// Animate, Spring or Keyframes, rests at its target in this frame: true
// for an unknown key.
func (e *node) Settled(key any) bool {
	if !e.hasState() {
		return true
	}
	if ch := e.st.motions[key]; ch != nil {
		return !ch.moving && !ch.pending
	}
	if m := e.st.colorMotions[key]; m != nil {
		return !m.k.moving && !m.k.pending
	}
	if s := e.st.sequences[key]; s != nil {
		return s.done
	}
	return true
}

// Arrived reports whether the value animated under key came to rest at
// its target in this frame, after moving there: true in that frame only,
// for the view to go on, as by removing what faded out.
func (e *node) Arrived(key any) bool {
	if !e.hasState() {
		return false
	}
	frame := e.c.rt.frame
	if ch := e.st.motions[key]; ch != nil {
		return ch.rested == frame
	}
	if m := e.st.colorMotions[key]; m != nil {
		return m.k.rested == frame
	}
	if s := e.st.sequences[key]; s != nil {
		return s.rested == frame
	}
	return false
}

// InView reports whether some of the element showed in the window in the
// frame before, within the scroll containers and clips around it: to
// animate it in as it scrolls into view, as Motion's whileInView.
func (e *node) InView() bool {
	if !e.hasState() {
		return false
	}
	s := e.st
	return s.seen+1 >= e.c.rt.frame && s.vw > 0 && s.vh > 0
}

// colorMotion is a color that AnimateColor moves from from to to, k of
// the way.
type colorMotion struct {
	from, to, shown Color
	k               channel
}

// AnimateColor returns a color that goes to target as a says, mixed with
// premultiplied alpha as transitions mix colors, so that a color fading
// from transparent keeps its hue; key tells apart the element's animated
// colors. It starts at the first target. As changing colors moves
// nothing, colors still change smoothly while the desktop asks for less
// motion; DisableAnimations shows them at once.
func (e *node) AnimateColor(key any, target Color, a Animation) Color {
	if a.Duration < 0 || a.Delay < 0 {
		panic("ui: animation durations must not be negative")
	}
	if a.Duration == 0 {
		a.Spring.physics()
	}
	if !e.hasState() {
		return target
	}
	st, rt, now := e.st, e.c.rt, e.c.now
	if st.colorMotions == nil {
		st.colorMotions = map[any]*colorMotion{}
	}
	m := st.colorMotions[key]
	if m == nil {
		st.colorMotions[key] = &colorMotion{from: target, to: target, shown: target, k: channel{value: 1, target: 1, at: now}}
		return target
	}
	moving := m.k.moving || m.k.pending
	m.k.advance(now)
	switch {
	case target == m.to:
	case rt.instant():
		m.k = channel{value: 1, target: 1, at: now}
	default:
		m.from = m.shown
		m.k = channel{at: now}
		headFor(&m.k, 1, a, now)
	}
	m.to = target
	if m.k.moving || m.k.pending {
		m.shown = mixPremultiplied(m.from, target, max(0, min(float32(m.k.value), 1)))
		e.c.AnimationFrame()
	} else {
		m.shown = target
		if moving {
			m.k.rested = rt.frame
		}
	}
	return m.shown
}

// Rubberband returns value held within lo and hi as iOS holds content
// dragged past its edges: beyond them, it goes less and less far, as on a
// rubber band, the less the larger dimension is (the size of what is
// dragged, along the drag).
func Rubberband(value, lo, hi, dimension float32) float32 {
	band := func(over float32) float32 {
		if dimension <= 0 {
			return 0
		}
		const c = 0.55
		return (1 - 1/(over*c/dimension+1)) * dimension
	}
	switch {
	case value < lo:
		return lo - band(lo-value)
	case value > hi:
		return hi + band(value-hi)
	}
	return value
}

// Project returns where a value going at velocity, in its units per
// second, comes to rest as iOS's scroll views slow down (their normal
// deceleration rate): where a fling would carry it, to choose the
// target a spring takes it to (SpringOptions.Velocity).
func Project(value, velocity float32) float32 {
	const rate = 0.998 // kept of the velocity each millisecond
	return value + velocity/1000*rate/(1-rate)
}

// Keyframe is a value at At since a sequence starts. The first At must be
// zero, subsequent times strictly increase. Ease controls the segment
// arriving at this frame; nil means Linear.
type Keyframe struct {
	At    time.Duration
	Value float32
	Ease  Easing
}

// RepeatForever repeats a sequence of Keyframes, or a Timeline, for as
// long as the view builds it (KeyframeOptions.Repeat,
// TimelineOptions.Repeat).
const RepeatForever = -1

// KeyframeOptions controls playback.
type KeyframeOptions struct {
	// Run replays the sequence, or a changed one, when it changes: the
	// first time through, it starts from the value shown, so that
	// interrupting it does not jump.
	Run uint64
	// Repeat is how many more times the sequence plays after the first,
	// as Motion's repeat: RepeatForever for ever. Alternate plays every
	// other time backward.
	Repeat    int
	Alternate bool
	// Delay is how long the sequence holds its first value before it
	// plays, and RepeatDelay how long it holds where it got to before it
	// plays again.
	Delay, RepeatDelay time.Duration
}

// Keyframes plays a sequence, holding its last value when done. It starts
// on the first call, with the first frame's value. Keep the element and
// key stable, and change Run when changing the sequence. Only active
// playback asks for animation frames. Reduced motion completes playback at
// its final value, including when the preference changes mid-sequence.
func (e *node) Keyframes(key any, frames []Keyframe, options ...KeyframeOptions) float32 {
	if len(frames) < 2 || frames[0].At != 0 {
		panic("ui: Keyframes needs at least two frames starting at zero")
	}
	for i, f := range frames {
		finiteMotion("keyframe", float64(f.Value))
		if i > 0 && f.At <= frames[i-1].At {
			panic("ui: keyframe times must strictly increase")
		}
	}
	if len(options) > 1 {
		panic("ui: Keyframes takes at most one KeyframeOptions")
	}
	var o KeyframeOptions
	if len(options) > 0 {
		o = options[0]
	}
	if o.Repeat < RepeatForever || o.Delay < 0 || o.RepeatDelay < 0 {
		panic("ui: keyframe Repeat must be RepeatForever or nonnegative, and Delay nonnegative")
	}
	if !e.hasState() {
		return frames[0].Value
	}
	st := e.st
	if st.sequences == nil {
		st.sequences = map[any]*sequence{}
	}
	s := st.sequences[key]
	now := e.c.now
	if s == nil || s.options.Run != o.Run {
		from := frames[0].Value
		if s != nil {
			from, _ = s.sample(now)
		}
		s = &sequence{frames: slices.Clone(frames), options: o, start: now.Add(o.Delay), from: from, value: from}
		st.sequences[key] = s
	}
	rt := e.c.rt
	switch {
	case s.done:
	case rt.preferences().ReduceMotion:
		s.value, s.done, s.rested = s.final(), true, rt.frame
	default:
		s.value, s.done = s.sample(now)
		if s.done {
			s.rested = rt.frame
		} else {
			e.c.AnimationFrame()
		}
	}
	return s.value
}

// playhead folds how long an animation has played into the time it shows:
// where it is within its duration, repeating and alternating as asked, and
// whether it came to rest at its end. iteration is which time through it
// is, for what starts from a value of its own the first time. A duration of
// zero plays on, for an animation whose duration is not known yet.
func playhead(raw, d time.Duration, repeat int, alternate bool, repeatDelay time.Duration) (time.Duration, int64, bool) {
	if raw <= 0 {
		return 0, 0, false
	}
	if d <= 0 {
		return raw, 0, false
	}
	plays := int64(repeat) + 1 // zero when it plays for ever
	period := d + repeatDelay
	if period < d {
		period = d
	}
	iteration, rest := int64(raw/period), raw%period
	if repeat != RepeatForever && (iteration >= plays || iteration == plays-1 && rest >= d) {
		if alternate && plays%2 == 0 {
			return 0, iteration, true // it came back to where it started
		}
		return d, iteration, true
	}
	at := min(rest, d)
	if alternate && iteration%2 == 1 {
		at = d - at
	}
	return at, iteration, false
}

type sequence struct {
	frames  []Keyframe
	options KeyframeOptions
	start   time.Time
	// from is the first value the first time through, which interrupting
	// a sequence starts from.
	from   float32
	value  float32
	done   bool
	rested uint64
}

// plays returns how many times the sequence plays, 0 for ever.
func (s *sequence) plays() int64 {
	if s.options.Repeat == RepeatForever {
		return 0
	}
	return int64(s.options.Repeat) + 1
}

func (s *sequence) final() float32 {
	if s.options.Alternate && s.plays()%2 == 0 {
		return s.frames[0].Value
	}
	return s.frames[len(s.frames)-1].Value
}

func (s *sequence) sample(now time.Time) (float32, bool) {
	if s.done {
		return s.value, true
	}
	elapsed := now.Sub(s.start)
	if elapsed < 0 {
		return s.from, false // delayed
	}
	d := s.frames[len(s.frames)-1].At
	at, iteration, done := playhead(elapsed, d, s.options.Repeat, s.options.Alternate, s.options.RepeatDelay)
	if done {
		return s.final(), true
	}
	for i := 1; i < len(s.frames); i++ {
		b := s.frames[i]
		if at > b.At {
			continue
		}
		a := s.frames[i-1]
		if i == 1 && iteration == 0 {
			a.Value = s.from
		}
		t := float32(at-a.At) / float32(b.At-a.At)
		ease := b.Ease
		if ease == nil {
			ease = Linear
		}
		return lerp(a.Value, b.Value, ease(t)), false
	}
	return s.final(), false
}

// CubicBezier returns the easing of the cubic Bézier curve from (0, 0) to
// (1, 1) through the control points (x1, y1) and (x2, y2), as CSS's
// cubic-bezier(): x1 and x2 within 0 and 1.
func CubicBezier(x1, y1, x2, y2 float32) Easing {
	finiteMotion("easing", float64(x1), float64(y1), float64(x2), float64(y2))
	if x1 < 0 || x1 > 1 || x2 < 0 || x2 > 1 {
		panic("ui: CubicBezier's x1 and x2 must be within 0 and 1")
	}
	cx := 3 * float64(x1)
	bx := 3*float64(x2-x1) - cx
	ax := 1 - cx - bx
	cy := 3 * float64(y1)
	by := 3*float64(y2-y1) - cy
	ay := 1 - cy - by
	curveX := func(t float64) float64 { return ((ax*t+bx)*t + cx) * t }
	return func(x float32) float32 {
		if x <= 0 || x >= 1 {
			return x
		}
		// Newton's method, then bisection where the slope is too flat.
		X := float64(x)
		t := X
		for range 8 {
			dx := (3*ax*t+2*bx)*t + cx
			if math.Abs(dx) < 1e-6 {
				break
			}
			e := curveX(t) - X
			if math.Abs(e) < 1e-7 {
				return float32(((ay*t+by)*t + cy) * t)
			}
			t -= e / dx
		}
		lo, hi := 0.0, 1.0
		t = X
		for range 40 {
			v := curveX(t)
			if math.Abs(v-X) < 1e-7 {
				break
			}
			if v < X {
				lo = t
			} else {
				hi = t
			}
			t = (lo + hi) / 2
		}
		return float32(((ay*t+by)*t + cy) * t)
	}
}
