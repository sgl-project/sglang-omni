package ui

import "fmt"

//go:generate go run ../internal/uigen

// Context is the window's stable UI context. Use it on the UI thread while
// building a view. Children share it and temporarily change its parent.
// Publish background results with Window.Update or Services.Invalidate.
type Context struct {
	rt       *engine
	services Services
}

func makeContext(c *context) *Context {
	if c == nil || c.rt == nil {
		return nil
	}
	return c.rt.public
}

func (c *Context) runtime() *engine {
	if c == nil || c.rt == nil || c.rt.closed {
		return nil
	}
	return c.rt
}

func (c *Context) build() *context {
	rt := c.runtime()
	if rt == nil || !rt.inFrame {
		return nil
	}
	return &rt.c
}

// Valid reports whether this context is building a view.
func (c *Context) Valid() bool { return c.build() != nil }

// Invalidate requests a frame from any goroutine.
func (c *Context) Invalidate() {
	if c != nil {
		c.services.Invalidate()
	}
}

// Key assigns the identity of the next control before its state is read.
// A compound widget consumes the key for its outermost element.
func (c *Context) Key(key any) *Context {
	if raw := c.build(); raw != nil {
		raw.nextKey, raw.keySet = key, true
	}
	return c
}

// elementOwner is detached on close, so saved handles retain only this small
// record. Retire it before a 32-bit generation wraps to prevent old aliases.
type elementOwner struct {
	rt         *engine
	generation uint32
}

// Element is a 16-byte, checked handle to a private pooled node. The zero
// value is absent. It expires before the next build pass, even in the same
// frame. Keep a Handle for persistent identity.
type Element struct {
	owner *elementOwner
	slot  uint32
	gen   uint32
}

func wrapElement(n *node) Element {
	if n == nil || n.c == nil || n.c.rt == nil || n.serial <= 0 {
		return Element{}
	}
	return Element{owner: n.c.rt.arena, slot: uint32(n.serial - 1), gen: uint32(n.epoch)}
}

func (rt *engine) nodeAt(slot uint32) *node {
	if uint64(slot) >= uint64(rt.c.used) {
		return nil
	}
	return &rt.c.chunks[int(slot)/chunkSize][int(slot)%chunkSize]
}

func (e Element) lookup() *node {
	if e.owner == nil || e.owner.rt == nil || e.gen != e.owner.generation {
		return nil
	}
	rt := e.owner.rt
	if !rt.inFrame {
		return nil
	}
	return rt.nodeAt(e.slot)
}

func (e Element) node() *node {
	n := e.lookup()
	if n == nil && e.owner != nil && e.owner.rt != nil && e.owner.rt.handleChecks {
		e.expired()
	}
	return n
}

func (e Element) expired() {
	panic(fmt.Sprintf("ui: element from pass %d used in pass %d; keep a ui.Handle instead", e.gen, e.owner.generation))
}

func (e Element) nodeFor(rt *engine) *node {
	if e.owner != rt.arena {
		return nil
	}
	return e.node()
}

// Valid reports whether the element belongs to the active build pass.
// Checking validity does not trigger expired-handle diagnostics.
func (e Element) Valid() bool { return e.lookup() != nil }

// Children builds children with the same context, scoped to this element.
// An absent element skips fn. Expired elements are diagnosed in development.
func (e Element) Children(fn func()) Element {
	if n := e.node(); n != nil && fn != nil {
		n.Children(fn)
	}
	return e
}

// Key keys a container before its children or local state are built.
// For a stateful control, use Context.Key before construction instead.
func (e Element) Key(key any) Element {
	if n := e.node(); n != nil {
		n.Key(key)
	}
	return e
}

// MaterialBuilder builds a material using a checked element.
type MaterialBuilder interface {
	Material
	BuildMaterial(Element) Material
}

func (e Element) Material(m Material) Element {
	if n := e.node(); n != nil {
		if b, ok := m.(MaterialBuilder); ok {
			m = b.BuildMaterial(e)
		}
		n.Material(m)
	}
	return e
}

func (rt *engine) keepPart(p any) uint32 {
	i := uint32(len(rt.parts))
	rt.parts = append(rt.parts, p)
	return i
}

func (r *Router) rtContext() *context {
	if r == nil || r.rt == nil || r.rt.closed || !r.rt.inFrame {
		return nil
	}
	return &r.rt.c
}
