package ui

import (
	"fmt"
	"time"
)

// A Timeline plays several tweens along one playhead, so that motion that
// belongs together is written, controlled and tested together: a card
// arriving while its title fades in, rows following one another, a small
// bounce after the card lands. It coordinates the motion already in MyGo:
// a tween returns the value it shows in this frame, and the view gives it
// to a transform, a pose or a value, as it would give one of AnimateTo's.
//
//	card := ui.Column(c)
//	tl := card.Timeline("intro", ui.TimelineOptions{Run: app.opened})
//	x := tl.Tween(ui.Tween{From: -200, To: 0, Duration: 400 * time.Millisecond})
//	y := tl.Tween(ui.Tween{From: 20, To: 0, Duration: 300 * time.Millisecond},
//		ui.AfterPrevious(-150*time.Millisecond))
//	card.Translate(x, y)
//
// An element keeps its timelines as it keeps its Keyframes, from frame to
// frame while the view builds it, and they go with it: build a timeline in
// the same order every frame, and change Run to play it again, or a
// changed one. A *Timeline is valid only while the view builds the frame
// it was returned in: ask the element for it again in the next.
type Timeline struct {
	c         *context
	st        *timelineState
	segs      []tweenSeg
	prevStart time.Duration
	prevEnd   time.Duration
}

// A Tween is a value going from From to To along Ease over Duration, which
// a Timeline places and plays. With no Ease it eases out (EaseOut), as an
// Animation does.
type Tween struct {
	From, To float32
	Duration time.Duration
	Ease     Easing
}

// TimelineOptions says how a timeline plays. What it changes applies to the
// frame it is given in, except Run, which plays the timeline over when it
// changes, as KeyframeOptions.Run does.
type TimelineOptions struct {
	// Run restarts the timeline when it changes, so that a view plays it
	// again by counting: the value a callback or a piece of state gives.
	Run uint64
	// Repeat is how many more times the timeline plays after the first,
	// RepeatForever for as long as the view builds it.
	Repeat int
	// Alternate plays every other time backward.
	Alternate bool
	// Delay holds the timeline at its first value for a while before it
	// plays, and RepeatDelay the value it got to between plays.
	Delay, RepeatDelay time.Duration
	// Paused holds the playhead where it is, so that the time a paused
	// timeline spends is not time it played and it goes on from where it
	// held when it is not paused.
	Paused bool
	// Reverse plays the timeline backward from where the playhead is, to
	// its start, as a menu that closes the way it opened; a timeline that
	// Run plays over while Reverse is set plays from its end.
	Reverse bool
	// Seek holds the playhead at a time of the timeline, and Progress at
	// a part of its duration, for a scrubber or a jump: while either is
	// set the timeline does not play, and it goes on from there when it
	// is not. Set one of them at most.
	Seek     *time.Duration
	Progress *float32
}

// Timeline returns the timeline the element keeps under key, playing it
// as o says. Every tween added to it shares one playhead, and returns the
// value it shows in this frame.
//
//	card := ui.Column(c)
//	tl := card.Timeline("intro", ui.TimelineOptions{Run: app.opened})
//	x := tl.Tween(ui.Tween{From: -200, To: 0, Duration: 400 * time.Millisecond})
//	card.Translate(x, 0)
//
// Key tells apart the timelines of the element, as it does its Keyframes,
// and an element asks for each of its timelines once per build. Play the
// timeline again, or a changed one, by changing o.Run. The *Timeline is
// valid while the view builds this frame.
//
// While the desktop asks for less motion (Preferences.ReduceMotion), a
// timeline that would play is at once where it would come to rest, but
// Seek, Progress and Paused still put and hold its playhead, as a drag
// still moves a Draggable element. DisableAnimations does not change a
// timeline, which plays on its own, as Keyframes do: jump it with Seek.
func (e *node) Timeline(key any, o TimelineOptions) *Timeline {
	if o.Repeat < RepeatForever || o.Delay < 0 || o.RepeatDelay < 0 {
		panic("ui: a Timeline Repeat must be RepeatForever or nonnegative, and its delays nonnegative")
	}
	if o.Seek != nil && o.Progress != nil {
		panic("ui: a Timeline takes Seek or Progress, not both")
	}
	if !e.hasState() {
		return nil
	}
	if e.st.timelines == nil {
		e.st.timelines = map[any]*timelineState{}
	}
	st := e.st.timelines[key]
	if st == nil {
		st = &timelineState{}
		e.st.timelines[key] = st
	}
	rt := e.c.rt
	if st.node == e && st.frame == rt.frame && st.pass == rt.pass {
		panic(fmt.Sprintf("ui: an element asked for its Timeline %#v twice in one build: give each timeline a key of its own", key))
	}
	st.node = e
	tl := st.handle
	if tl == nil {
		tl = &Timeline{}
		st.handle = tl
	}
	tl.c, tl.st = e.c, st
	tl.segs = st.segs[:0]
	tl.prevStart, tl.prevEnd = 0, 0
	tl.begin(o)
	return tl
}

// Tween adds tw to the timeline where at places it, following the tween
// before it at once with no placement, and returns the value it shows in
// this frame: its From before its time, its To after it.
//
//	x := tl.Tween(ui.Tween{From: 0, To: 120, Duration: 300 * time.Millisecond})
//	y := tl.Tween(ui.Tween{From: 20, To: 0, Duration: 200 * time.Millisecond},
//		ui.AfterPrevious(-150*time.Millisecond))
func (t *Timeline) Tween(tw Tween, at ...Placement) float32 {
	finiteMotion("tween", float64(tw.From), float64(tw.To))
	if tw.Duration < 0 {
		panic("ui: animation durations must not be negative")
	}
	if t == nil {
		return tw.From
	}
	start := t.place(at)
	seg := tweenSeg{start: start, dur: tw.Duration, from: tw.From, to: tw.To, ease: tw.Ease}
	t.segs = append(t.segs, seg)
	st := t.st
	st.segs = t.segs
	if end := start + tw.Duration; end > st.build {
		st.build = end
		if st.done && !st.over && !st.reverse && !st.holding && end > st.duration {
			// The view arranges a timeline that reaches past the one the
			// playhead came to rest at the end of: it plays on from there,
			// rather than rest, and is unfinished until it has played.
			st.moving = true
			t.c.AnimationFrame()
		}
	}
	t.prevStart, t.prevEnd = start, start+tw.Duration
	return t.show(seg)
}

// Label marks a time of the timeline for a tween to start at with AtLabel.
// With no placement it marks where the tween before it ends.
//
//	tl.Label("intro", ui.AtTime(0))
//	...
//	tl.Tween(ui.Tween{...}, ui.AtLabel("intro", 200*time.Millisecond))
func (t *Timeline) Label(name string, at ...Placement) {
	if t == nil {
		return
	}
	t.st.labels = append(t.st.labels, timelineLabel{name: name, at: t.place(at)})
}

// Duration is how long the timeline plays: the end of its last tween,
// without its repeats.
func (t *Timeline) Duration() time.Duration {
	if t == nil {
		return 0
	}
	return max(t.st.build, t.st.duration)
}

// Time is where the playhead is, from the start of the timeline.
func (t *Timeline) Time() time.Duration {
	if t == nil {
		return 0
	}
	st := t.st
	if st.over {
		if st.atStart {
			return 0
		}
		return t.Duration()
	}
	return min(st.t, t.Duration())
}

// Progress is how far the playhead is along the timeline, from 0 to 1.
func (t *Timeline) Progress() float32 {
	if t == nil {
		return 0
	}
	st, d := t.st, t.Duration()
	if st.over || st.done && d <= 0 {
		if st.atStart {
			return 0
		}
		return 1
	}
	if d <= 0 {
		return 0
	}
	return min(max(float32(st.t)/float32(d), 0), 1)
}

// Settled reports whether the timeline rests, at its end with its repeats
// played out, or at its start while it plays in Reverse, as
// Element.Settled does for a value.
func (t *Timeline) Settled() bool {
	if t == nil || !t.st.done {
		return false
	}
	if t.st.over || t.st.reverse {
		return true
	}
	// A view that arranges a longer timeline than the one the playhead came
	// to rest at the end of has not finished: the frame that plays it is
	// coming.
	return t.st.build <= t.st.duration
}

// Arrived is true in the one frame in which the timeline came to rest, for
// the view to go on, as Element.Arrived is.
func (t *Timeline) Arrived() bool {
	return t != nil && t.st.rested != 0 && t.st.rested == t.c.rt.frame
}

// place resolves where a tween or label starts, in at most one placement.
func (t *Timeline) place(at []Placement) time.Duration {
	if len(at) > 1 {
		panic("ui: a Timeline placement takes at most one Placement")
	}
	var p Placement
	if len(at) == 1 {
		p = at[0]
	}
	var pos time.Duration
	switch p.kind {
	case placeWith:
		pos = t.prevStart + p.at
	case placeAt:
		pos = p.at
	case placeLabel:
		pos = t.label(p.label) + p.at
	default:
		pos = t.prevEnd + p.at
	}
	if pos < 0 {
		// Nothing starts before the timeline does: an overlap larger
		// than what came before it starts at zero.
		pos = 0
	}
	return pos
}

// label returns where a label of this frame's timeline was marked.
func (t *Timeline) label(name string) time.Duration {
	labels := t.st.labels
	for i := len(labels) - 1; i >= 0; i-- {
		if labels[i].name == name {
			return labels[i].at
		}
	}
	panic("ui: a Timeline has no label " + name)
}

// show returns the value a segment shows at this frame's playhead: 0 to 1
// of its time along its Ease, between its From and its To.
func (t *Timeline) show(s tweenSeg) float32 {
	st := t.st
	if st.over {
		// Less motion: every tween is where the timeline comes to rest at
		// once, at its end, or at its start when it plays back to it.
		if st.atStart {
			return s.from
		}
		return s.to
	}
	now := st.t
	if s.dur <= 0 {
		// A tween that takes no time is where it ends as soon as its time
		// comes, at the end of the timeline as anywhere else: the playhead
		// comes to rest on it there.
		if now >= s.start {
			return s.to
		}
		return s.from
	}
	if now <= s.start {
		return s.from
	}
	if now >= s.start+s.dur {
		return s.to
	}
	ease := s.ease
	if ease == nil {
		ease = EaseOut
	}
	return lerp(s.from, s.to, ease(float32(now-s.start)/float32(s.dur)))
}

// begin starts a frame's build of the timeline: at the start of a frame it
// takes what the last one built and moves the playhead, and at the start of
// a pass it starts the segments and labels over, as building always does.
func (t *Timeline) begin(o TimelineOptions) {
	st, rt := t.st, t.c.rt
	if st.frame != rt.frame {
		st.frame, st.pass = rt.frame, rt.pass
		st.duration, st.build = st.build, 0
		st.segs, st.labels = st.segs[:0], st.labels[:0]
		t.segs = st.segs
		t.tick(o)
		return
	}
	if st.pass != rt.pass {
		st.pass = rt.pass
		st.build = 0
		st.segs, st.labels = st.segs[:0], st.labels[:0]
		t.segs = st.segs
	}
	// The clock of this frame moved once already: what the options say now
	// still decides what the pass shows, so a handler that paused, reversed
	// or played the timeline over shows it in the frame it ran in, as a
	// changed value the view builds later shows in the same frame.
	t.cue(o)
}

// tick moves the playhead on by this frame's time, once per frame, in the
// direction the last frame played it.
func (t *Timeline) tick(o TimelineOptions) {
	st, c := t.st, t.c
	if !st.started {
		st.started, st.at, st.run = true, c.now, o.Run
		t.cue(o)
		return
	}
	st.built = true
	dt := c.now.Sub(st.at)
	st.at = c.now
	// Only time the timeline played moves it: a paused, held or finished
	// timeline asks for no frames, so the frame that lets it go carries
	// the whole wait in its delta and would jump the playhead by it.
	if dt > 0 && st.moving {
		if st.reverse {
			// Back to where it starts playing, not into its Delay.
			st.raw = max(st.raw-dt, min(st.raw, o.Delay))
		} else {
			st.raw += dt
		}
	}
	t.cue(o)
}

// cue puts the playhead where this pass's options ask and asks for the frame
// that goes on, as long as the timeline moves.
func (t *Timeline) cue(o TimelineOptions) {
	st, c := t.st, t.c
	if st.run != o.Run {
		// Run plays the timeline over, in the frame it changes in: from its
		// start, or back from its end.
		st.run, st.raw, st.done, st.over, st.rested = o.Run, 0, false, false, 0
		if o.Reverse {
			end, ok := st.end(o)
			if !ok {
				end = st.duration
			}
			st.raw = o.Delay + end
		}
	}
	st.reverse = o.Reverse
	held := false
	if o.Seek != nil {
		st.raw, held = o.Delay+*o.Seek, true
	}
	if o.Progress != nil {
		st.raw, held = o.Delay+time.Duration(float64(*o.Progress)*float64(st.duration)), true
	}
	if held {
		// A scrubber puts the playhead where it asks, also on a timeline
		// that less motion put at once where it comes to rest.
		st.over = false
	} else if !o.Paused && c.rt.preferences().ReduceMotion {
		// Values and poses go to their targets at once, and a sequence
		// ends at its last value: a timeline that would play is over where
		// it would have come to rest, in this frame. What the view asks of
		// its playhead, holding it or putting it somewhere, it still does,
		// as a drag still moves a Draggable element.
		st.over = true
	}
	if st.over {
		// Where Keyframes ends: back at the start when it plays back to it,
		// or alternates an even number of times.
		st.atStart = o.Reverse || o.Alternate && (o.Repeat+1)%2 == 0
		st.t = st.duration
		if st.atStart {
			st.t = 0
		}
		st.done, st.moving, st.holding = true, false, true
		if st.rested == 0 {
			st.rested = c.rt.frame
		}
		return
	}
	// Nothing plays past the end, so that playing back, or a longer
	// timeline, goes on from it at once.
	if end, ok := st.end(o); ok {
		st.raw = min(st.raw, o.Delay+end)
	}
	st.raw = max(st.raw, 0)
	raw := st.raw - o.Delay
	st.t, _, st.done = playhead(raw, st.duration, o.Repeat, o.Alternate, o.RepeatDelay)
	if raw > 0 && st.duration <= 0 {
		// A timeline whose tweens take no time is over as soon as a
		// frame has gone by: the frame it is first built in has no
		// duration yet to be over.
		st.done = st.built
	}
	if o.Reverse {
		st.done = raw <= 0
	}
	st.holding = o.Paused || held
	st.moving = !st.done && !st.holding
	if st.done {
		if st.rested == 0 {
			st.rested = c.rt.frame
		}
		return
	}
	// It moves again, so the next time it comes to rest is an arrival of
	// its own, as Arrived reports.
	st.rested = 0
	if held && !st.built {
		// A part of the duration, or a time, asked for in the frame the
		// timeline is first built in: it has no duration to take a part
		// of yet, so ask for the frame that knows one, rather than sit
		// at the start.
		c.AnimationFrame()
		return
	}
	if st.moving {
		c.AnimationFrame()
	}
}

// end returns how long the timeline plays with its repeats, false when it
// repeats for ever or its duration is not known yet.
func (st *timelineState) end(o TimelineOptions) (time.Duration, bool) {
	if o.Repeat == RepeatForever || st.duration <= 0 {
		return 0, false
	}
	plays := time.Duration(o.Repeat) + 1
	return plays*st.duration + (plays-1)*o.RepeatDelay, true
}

// A Placement says where along the timeline a tween or a label starts:
// relative to the tween before it, at a label, or at a time, as GSAP's
// position parameter. With no placement, what comes next follows the
// tween before it at once, as AfterPrevious(0). Nothing starts before the
// timeline does: a placement before its start is at its start.
//
//	tl.Tween(ui.Tween{...}, ui.AfterPrevious(-150*time.Millisecond)) // overlaps it by 150ms
//	tl.Tween(ui.Tween{...}, ui.WithPrevious(100*time.Millisecond))
//	tl.Tween(ui.Tween{...}, ui.AtLabel("intro", 200*time.Millisecond))
//	tl.Tween(ui.Tween{...}, ui.AtTime(500*time.Millisecond))
type Placement struct {
	kind  placeKind
	at    time.Duration
	label string
}

// placeKind is what a Placement measures from.
type placeKind uint8

const (
	placeAfter placeKind = iota // after the one before it, which is nothing
	placeWith
	placeAt
	placeLabel
)

// AfterPrevious starts a tween offset after the tween before it ends, or
// with a negative offset that long before, to overlap it, as GSAP's "+=" and
// "-=".
func AfterPrevious(offset time.Duration) Placement {
	return Placement{kind: placeAfter, at: offset}
}

// WithPrevious starts a tween offset after the tween before it starts, to
// move something along with it, as GSAP's "<".
func WithPrevious(offset time.Duration) Placement {
	return Placement{kind: placeWith, at: offset}
}

// AtTime starts a tween at time t of the timeline.
func AtTime(t time.Duration) Placement { return Placement{kind: placeAt, at: t} }

// AtLabel starts a tween offset after a Label of the timeline, as rows
// staggered from one label each start i times a step after it:
//
//	for i := range rows {
//		rows[i] = tl.Tween(ui.Tween{From: 0, To: 1, Duration: 200 * time.Millisecond},
//			ui.AtLabel("rows", time.Duration(i)*40*time.Millisecond))
//	}
func AtLabel(name string, offset time.Duration) Placement {
	return Placement{kind: placeLabel, label: name, at: offset}
}

// tweenSeg is one tween as the timeline placed it.
type tweenSeg struct {
	start, dur time.Duration
	from, to   float32
	ease       Easing
}

// timelineState is what a Timeline keeps between frames: how long its
// playhead has run, and the segments and labels its last frame built,
// which this frame's playhead folds around. An element keeps one per key
// (state.timelines), and it goes with the element's state.
type timelineState struct {
	handle   *Timeline
	node     *node         // the element that last asked for it, to catch it asking twice
	frame    uint64        // the frame the playhead last moved in
	pass     int           // the pass it was built in, to build once per pass
	at       time.Time     // the time of that frame's clock
	started  bool          // whether it has a time to measure from
	built    bool          // whether a frame before this one built it
	moving   bool          // whether the last pass played it: the time to the next frame is played
	reverse  bool          // whether the last pass played it backward
	holding  bool          // whether the last pass paused or held it
	run      uint64        // the Run it plays, to restart when it changes
	raw      time.Duration // where the playhead is through its repeats, its Delay included
	duration time.Duration // the last frame's Duration, which folds this one
	build    time.Duration // the Duration this frame is building
	t        time.Duration // where the playhead is
	done     bool          // it came to rest
	over     bool          // less motion: it settled at once
	atStart  bool          // less motion: where it settled is its start
	rested   uint64        // the frame it came to rest in, for Arrived
	segs     []tweenSeg
	labels   []timelineLabel
}

// timelineLabel is a label a frame marked on a timeline.
type timelineLabel struct {
	name string
	at   time.Duration
}
