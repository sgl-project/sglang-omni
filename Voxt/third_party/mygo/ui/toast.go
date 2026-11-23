package ui

import (
	"strconv"
	"time"
)

const (
	// toastTime is how long a toast shows unless it says, and
	// actionToastTime one of ToastAction, which the user needs time to
	// reach.
	toastTime       = 4 * time.Second
	actionToastTime = 8 * time.Second
	// toastLimit is how many toasts show at once, the newest; older ones
	// wait hidden, their time running, as with Base UI's limit.
	toastLimit = 3
)

// Toast is a toast that AddToast shows, and that the view builds with
// ToastViewportBase to give toasts a look of its own.
type Toast struct {
	// ID names the toast: adding one with the ID of a toast showing
	// changes that toast, which shows anew. AddToast makes one up when it
	// is empty.
	ID string
	// Title and Description are what the toast says, which screen readers
	// read out as it shows.
	Title, Description string
	// Type tells what kind of toast it is, as "error" or "success", for
	// the look to tell.
	Type string
	// Timeout is how long the toast shows: 4 seconds when zero, until it
	// is closed when negative. The time stops while the pointer rests on
	// the toasts, the keyboard focus is in them, or the window is in the
	// background.
	Timeout time.Duration
	// Action labels a button that runs OnAction, as Undo after deleting,
	// and closes the toast.
	Action   string
	OnAction func()
	// OnClose runs as the toast closes: its time is over, or the user, its
	// action or CloseToast closed it.
	OnClose func()
	// Data holds what else the look of the toast needs.
	Data any
}

// life is how long the toast shows, 0 until it is closed.
func (t *Toast) life() time.Duration {
	switch {
	case t.Timeout < 0:
		return 0
	case t.Timeout == 0:
		return toastTime
	}
	return t.Timeout
}

// toast is a toast the window holds, under key: it showed for spent,
// and since at while the time of the toasts runs.
type toast struct {
	Toast
	key   uint64
	spent time.Duration
	at    time.Time
}

// Toast shows message near the bottom of the window for a few seconds, as
// the outcome of what the user just did:
//
//	if ui.Button(c, "Save").Clicked() {
//		app.save()
//		c.Toast("Saved")
//	}
//
// A message already showing shows anew; others stack above it, three at
// most. It shows for as long as the pointer rests on it. Screen readers
// read it out. It is AddToast with message as the ID and the title.
func (c *context) Toast(message string) {
	c.AddToast(Toast{ID: message, Title: message})
}

// ToastAction shows message as Toast does, for longer, with a button of
// label that runs action, as Undo after deleting, and closes the toast;
// Tab reaches the button. action runs on the main thread, as the view
// does, which builds the frame again after it.
//
//	app.trash(note)
//	c.ToastAction("Note deleted", "Undo", func() { app.restore(note) })
func (c *context) ToastAction(message, label string, action func()) {
	c.AddToast(Toast{ID: message, Title: message, Timeout: actionToastTime, Action: label, OnAction: action})
}

// AddToast shows t near the bottom of the window, as Toast does, and
// returns its ID, to close it with CloseToast:
//
//	id := c.AddToast(ui.Toast{Title: "Uploading", Timeout: -1})
//	// ...
//	c.CloseToast(id)
//
// Adding a toast with the ID of one showing changes that toast, which
// shows anew. Toasts show with the theme's look, unless the view builds
// them with ToastViewportBase.
func (c *context) AddToast(t Toast) string {
	rt := c.rt
	rt.nextToast++
	if t.ID == "" {
		t.ID = "toast-" + strconv.FormatUint(rt.nextToast, 10)
	}
	ts := toast{Toast: t, key: rt.nextToast, at: c.now}
	if i := rt.toastIndex(t.ID); i >= 0 {
		ts.key = rt.toasts[i].key
		rt.toasts[i] = ts
	} else {
		rt.toasts = append(rt.toasts, ts)
	}
	// Nothing else tells screen readers: the focus stays.
	c.Announce(t.Title)
	c.Announce(t.Description)
	rt.requestFrame()
	return t.ID
}

// CloseToast closes the toast with id, or every toast when id is empty,
// and runs their OnClose.
func (c *context) CloseToast(id string) {
	rt := c.rt
	var closed []func()
	kept := rt.toasts[:0]
	for _, ts := range rt.toasts {
		if id != "" && ts.ID != id {
			kept = append(kept, ts)
		} else if ts.OnClose != nil {
			closed = append(closed, ts.OnClose)
		}
	}
	clear(rt.toasts[len(kept):])
	rt.toasts = kept
	rt.consumed = true
	rt.requestFrame()
	for _, fn := range closed {
		fn()
	}
}

func (rt *engine) toastIndex(id string) int {
	for i := range rt.toasts {
		if rt.toasts[i].ID == id {
			return i
		}
	}
	return -1
}

// toastAge returns how long ts has shown.
func (rt *engine) toastAge(ts *toast, now time.Time) time.Duration {
	if rt.toastsPaused {
		return ts.spent
	}
	return ts.spent + now.Sub(ts.at)
}

// pauseToasts stops the time of the toasts, or lets it run again.
func (rt *engine) pauseToasts(paused bool, now time.Time) {
	if paused == rt.toastsPaused {
		return
	}
	for i := range rt.toasts {
		ts := &rt.toasts[i]
		if paused {
			ts.spent += now.Sub(ts.at)
		} else {
			ts.at = now
		}
	}
	rt.toastsPaused = paused
}

// ToastViewportBase builds the window's toasts without a look, in place of
// the theme's: fn styles viewport, a column covering the window that the
// pointer goes through, which holds them at its bottom, and builds each of
// toasts with ToastBase, giving it the look of its Type and Data. toasts
// are those showing, the newest three, oldest first. Call it once in the
// view, wherever. Toasts in the bottom right corner:
//
//	ui.ToastViewportBase(c, func(viewport ui.Element, toasts []ui.Toast) {
//		viewport.Padding(16).AlignItems(ui.End).Gap(8)
//		for _, t := range toasts {
//			toast := ui.ToastBase(c, t)
//			toast.Root.Row().Gap(12).Padding(10, 14).Radius(8).Background(surface)
//			toast.Root.Children(func() {
//				ui.Text(c, t.Title).Grow(1)
//				toast.CloseButton().Label("Close").Children(func() { ui.Icon(c, x) })
//			})
//		}
//	})
//
// The time of the toasts stops while the pointer rests on viewport, the
// keyboard focus is in it, or the window is in the background. viewport
// stays while no toast shows, for the last one to go with an exit
// transition. It returns viewport.
func coreToastViewportBase(c *context, fn func(viewport *node, toasts []Toast)) *node {
	return toastViewport(c, true, fn)
}

// toastViewportKey keys the toasts' viewport among what the overlay holds.
const toastViewportKey = "mygo.toasts"

// toastViewport is ToastViewportBase, which builds no viewport while no
// toast shows unless keep.
func toastViewport(c *context, keep bool, fn func(viewport *node, toasts []Toast)) *node {
	rt := c.rt
	if rt.toastFrame == rt.frame && rt.toastPass == rt.pass {
		return nil
	}
	rt.toastFrame, rt.toastPass = rt.frame, rt.pass
	now := c.now
	// The toasts whose time is over close.
	var ended []func()
	kept := rt.toasts[:0]
	for _, ts := range rt.toasts {
		if life := ts.life(); life > 0 && rt.toastAge(&ts, now) >= life {
			if ts.OnClose != nil {
				ended = append(ended, ts.OnClose)
			}
			continue
		}
		kept = append(kept, ts)
	}
	clear(rt.toasts[len(kept):])
	rt.toasts = kept
	for _, fn := range ended {
		fn()
	}
	if len(rt.toasts) == 0 {
		rt.toastsPaused = false
		if !keep {
			return nil
		}
	}
	list := rt.toastList[:0]
	for _, ts := range rt.toasts[max(0, len(rt.toasts)-toastLimit):] {
		list = append(list, ts.Toast)
	}
	rt.toastList = list
	var viewport *node
	coreOverlay(c, func() {
		// Keyed, as an element of the engine's own: in the place of the
		// overlay's first child, it would take the identity, and so the
		// hover and the focus, of what a dialog that just closed built
		// there, and stop the toasts' time as if the pointer rested on
		// them, until the pointer moved.
		viewport = coreColumn(c).Key(toastViewportKey).Absolute().Left(0).Top(0).Right(0).Bottom(0).Justify(End).PassThrough()
		rt.pauseToasts(viewport.Hovered() || viewport.FocusWithin() || !rt.windowFocus(), now)
		viewport.Children(func() { fn(viewport, list) })
	})
	// A frame closes the toast whose time is over next.
	if !rt.toastsPaused {
		var next time.Duration
		for i := range rt.toasts {
			ts := &rt.toasts[i]
			if life := ts.life(); life > 0 {
				if left := life - rt.toastAge(ts, now); next == 0 || left < next {
					next = left
				}
			}
		}
		if next > 0 {
			c.After(next)
		}
	}
	return viewport
}

// ToastParts are the parts of a toast without a look, which ToastBase
// makes.
type toastParts struct {
	// Root is the toast, which assistive technology sees as a status, and
	// which Escape closes while the keyboard focus is in it. Give it
	// children.
	Root      *node
	c         *context
	id        string
	age, left time.Duration
}

// ToastBase creates the toast t without a look, in the viewport of
// ToastViewportBase.
func coreToastBase(c *context, t Toast) toastParts {
	rt := c.rt
	p := toastParts{c: c, id: t.ID}
	var key any = t.ID
	if i := rt.toastIndex(t.ID); i >= 0 {
		ts := &rt.toasts[i]
		key, p.age = ts.key, rt.toastAge(ts, c.now)
		if life := ts.life(); life > 0 {
			p.left = life - p.age
		}
	}
	p.Root = coreBox(c).Key(key).Role(RoleStatus)
	// It takes the pointer, which stops the time resting on it.
	p.Root.flags |= flagHover
	if p.Root.Shortcut(0, KeyEscape) {
		c.CloseToast(t.ID)
	}
	return p
}

// Left returns how long the toast shows still, to count down, or 0 for one
// that shows until it is closed. It does not move while the time of the
// toasts stops; ask for frames with Context.AnimationFrame to draw it
// moving.
func (p toastParts) Left() time.Duration { return p.left }

// ActionButton creates a button without a look, as ButtonBase does, that
// runs the toast's OnAction and closes it. Give it children, as the
// toast's Action.
func (p toastParts) ActionButton() *node {
	b := coreButtonBase(p.c)
	if b.Clicked() {
		if i := p.c.rt.toastIndex(p.id); i >= 0 {
			if action := p.c.rt.toasts[i].OnAction; action != nil {
				action()
			}
		}
		p.c.CloseToast(p.id)
	}
	return b
}

// CloseButton creates a button without a look, as ButtonBase does, that
// closes the toast. Give it children, and a Label when they are not text.
func (p toastParts) CloseButton() *node {
	b := coreButtonBase(p.c)
	if b.Clicked() {
		p.c.CloseToast(p.id)
	}
	return b
}

// buildToasts builds the toasts with the theme's look, unless the view
// built them with ToastViewportBase.
func (rt *engine) buildToasts(c *context) {
	t := c.theme
	fill, text := t.inverse()
	toastViewport(c, false, func(viewport *node, toasts []Toast) {
		viewport.Bottom(t.Space(6)).AlignItems(Center).Gap(t.Space(2))
		for _, ts := range toasts {
			p := coreToastBase(c, ts)
			box := p.Root.Row().AlignItems(Center).Gap(t.Space(4)).Padding(t.Space(2.5), t.Space(4)).Radius(t.Space(2)).MaxWidth(c.w - t.Space(12)).
				Background(fill).TextColor(text)
			box.Shadow(0, 6, 20, 0, RGBA(0, 0, 0, 0.25))
			// It fades in, and out at the end.
			const fade = 200 * time.Millisecond
			opacity := float32(1)
			switch {
			case rt.toastsPaused:
			case p.age < fade:
				opacity = float32(p.age) / float32(fade)
				c.AnimationFrame()
			case p.left > 0 && p.left < fade:
				opacity = float32(p.left) / float32(fade)
				c.AnimationFrame()
			case p.left > 0:
				c.After(p.left - fade)
			}
			box.Opacity(opacity)
			box.Children(func() {
				if ts.Description == "" {
					coreText(c, ts.Title)
				} else {
					coreColumn(c).Shrink(1).Gap(t.Space(0.5)).Children(func() {
						coreText(c, ts.Title).FontWeight(600)
						coreText(c, ts.Description).TextColor(text.Alpha(0.75))
					})
				}
				if ts.Action == "" {
					return
				}
				b := p.ActionButton().Padding(t.Space(1), t.Space(2.5)).Radius(t.Radius).TextColor(t.Accent.Mix(text, 0.35)).FontWeight(600)
				b.styleFn = func(b *node) {
					if b.Hovered() {
						b.bg = text.Alpha(0.12)
					}
				}
				b.Children(func() { coreText(c, ts.Action) })
			})
		}
	})
}
