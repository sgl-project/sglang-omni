package ui

// Handle identifies a control across builds. Bind it to an Element on each
// pass. One handle may have separate bindings in several windows. Queries
// take a Context and read committed identity, regardless of build order.
// Use on the UI thread; background work uses Window.Update.
type Handle struct{ state *handleState }

// Ref is an alias for Handle. Bind and Handle are preferred for new code.
type Ref = Handle

type handleState struct {
	bindings []*refState
	pending  bool
}

type refState struct {
	window            *Context
	element           Element
	id                uint64
	requested, closed bool
}

func (r *Handle) init() *handleState {
	if r == nil {
		return nil
	}
	if r.state == nil {
		r.state = &handleState{}
	}
	return r.state
}

func (r *Handle) binding(rt *engine, create bool) *refState {
	if r == nil || rt == nil {
		return nil
	}
	if r.state == nil && !create {
		return nil
	}
	s := r.init()
	for _, b := range s.bindings {
		if b.window == rt.public {
			return b
		}
	}
	if !create {
		return nil
	}
	b := &refState{window: rt.public, requested: s.pending}
	s.pending = false
	s.bindings = append(s.bindings, b)
	rt.refs = append(rt.refs, b)
	return b
}

// Focus requests focus, waiting while the control is hidden. Supply a Context
// to select a window. Without one, a handle must have at most one open window.
// A request made before its first binding waits for that binding.
func (r *Handle) Focus(window ...*Context) {
	if r == nil {
		return
	}
	if len(window) > 1 {
		panic("ui: Handle.Focus accepts one Context")
	}
	var b *refState
	if len(window) == 1 {
		b = r.binding(window[0].runtime(), true)
	} else {
		s := r.init()
		for _, candidate := range s.bindings {
			if candidate.closed || candidate.window.runtime() == nil {
				continue
			}
			if b != nil {
				panic("ui: Handle.Focus needs a Context when bound in several windows")
			}
			b = candidate
		}
		if b == nil {
			s.pending = true
			return
		}
	}
	if b != nil && !b.closed {
		b.requested = true
		if rt := b.window.runtime(); rt != nil {
			rt.requestFrame()
		}
	}
}

// RequestFocus is an alias for Focus.
func (r *Handle) RequestFocus(window ...*Context) { r.Focus(window...) }

// CancelFocus cancels pending focus in the selected window, or every binding.
func (r *Handle) CancelFocus(window ...*Context) {
	if r == nil || r.state == nil {
		return
	}
	if len(window) > 1 {
		panic("ui: Handle.CancelFocus accepts one Context")
	}
	if len(window) == 1 {
		if b := r.binding(window[0].runtime(), false); b != nil {
			b.requested = false
		}
		return
	}
	r.state.pending = false
	for _, b := range r.state.bindings {
		b.requested = false
	}
}

// Bind gives the handle this control's identity in the current window.
func (e Element) Bind(r *Handle) Element {
	n := e.node()
	if n == nil || r == nil {
		return e
	}
	b := r.binding(n.c.rt, true)
	if b.element.Valid() && b.element != e {
		panic("ui: Handle bound to two controls in one build")
	}
	b.element = e
	return e
}

// Ref is an alias for Bind.
func (e Element) Ref(r *Ref) Element { return e.Bind(r) }

// Resolve returns this pass's element. Focus queries do not need Resolve.
func (c *Context) Resolve(ref Handle) Element {
	rt := c.runtime()
	b := (&ref).binding(rt, false)
	if b == nil || b.closed || !b.element.Valid() {
		return Element{}
	}
	return b.element
}

func (r Handle) Focused(c *Context) bool {
	rt := c.runtime()
	b := (&r).binding(rt, false)
	return b != nil && !b.closed && b.id != 0 && rt.focused == b.id && rt.windowFocus()
}

func (r Handle) FocusWithin(c *Context) bool {
	rt := c.runtime()
	b := (&r).binding(rt, false)
	if b == nil || b.closed || b.id == 0 {
		return false
	}
	for s := rt.states[rt.focused]; s != nil; s = rt.states[s.parent] {
		if s.id == b.id {
			return true
		}
		if s.parent == 0 {
			break
		}
	}
	return false
}

func (c *Context) Focused(ref Handle) bool     { return ref.Focused(c) }
func (c *Context) FocusWithin(ref Handle) bool { return ref.FocusWithin(c) }

// OnShortcut declares an action before or after Bind. A control omitted from
// this pass takes no shortcut. Actions run once after bindings are complete.
func (r *Handle) OnShortcut(c *Context, mods Modifiers, key Key, fn func()) {
	rt := c.runtime()
	if rt == nil || !rt.inFrame || r == nil || fn == nil {
		return
	}
	rt.actions = append(rt.actions, action{handle: r, window: c, kind: actionShortcut, mods: mods, key: key, fn: fn})
}

func (rt *engine) applyFocusRequests() {
	for _, b := range rt.refs {
		if b.closed {
			continue
		}
		if n := b.element.lookup(); n != nil {
			b.id = n.id
		}
		if !b.requested {
			continue
		}
		n := b.element.lookup()
		if n == nil || n.c.rt != rt || n.disabled() {
			continue
		}
		b.id = n.id
		b.requested = false
		if rt.focused != b.id {
			n.Focus()
			rt.consumed = true
		}
	}
	rt.applyFocusBindings()
}

// Bounds returns the control's box in the committed frame, in window DIPs.
// It works between builds on the UI thread; hidden or closed controls return
// an empty box. Use this instead of retaining an Element in input callbacks.
func (r Handle) Bounds(c *Context) Rect {
	rt := c.runtime()
	b := (&r).binding(rt, false)
	if b == nil || b.closed || b.id == 0 {
		return Rect{}
	}
	s := rt.states[b.id]
	if s == nil {
		return Rect{}
	}
	return Rect{s.x, s.y, s.w, s.h}
}
