//go:build js

package ui

import (
	"strings"
	"syscall/js"
)

func init() { browserHistory = &tabHistory{} }

// tabHistory is the history of the browser's tab: the router following
// it keeps its location in the URL's fragment, as "#/notes/42", and the
// number of its entry (Router.base + Router.at) in the state of the
// browser's entry, which tells where the browser's back and forward
// buttons go among the router's entries.
type tabHistory struct {
	r *Router
	// popping tells that the router changes for the browser, which has
	// done what the change would ask of it.
	popping  bool
	popstate js.Func
}

// stateKey is the property of history.state holding an entry's number.
const stateKey = "mygoEntry"

// claim has r follow the history when no open window's router does: it
// shows the page of the URL's fragment, if any, and takes over the
// browser's entry.
func (h *tabHistory) claim(r *Router) {
	if h.r == r {
		return
	}
	// A Tester's routers leave the page's history alone, as do those of
	// an app in a frame of another page, whose history the tab's is too.
	if _, ok := r.rt.host.(*windowHost); !ok || !js.Global().Get("top").Equal(js.Global().Get("self")) {
		return
	}
	if h.r != nil && h.r.rt != nil && !h.r.rt.closed {
		return
	}
	h.r = r
	if h.popstate.IsUndefined() {
		h.popstate = js.FuncOf(func(js.Value, []js.Value) any {
			r := h.r
			if r == nil || r.rt == nil || r.rt.closed {
				return nil
			}
			state, loc := historyState(), fragment()
			r.rt.host.post(func() {
				if h.r == r {
					h.popped(state, loc)
				}
			})
			return nil
		})
		js.Global().Call("addEventListener", "popstate", h.popstate)
	}
	if loc := fragment(); loc != "" {
		h.popping = true
		r.Replace(loc)
		h.popping = false
	}
	// After a reload, the browser's entry has its number: the entries
	// before and after it come back as the router's (popped).
	if state := historyState(); state.Type() == js.TypeObject && state.Get(stateKey).Type() == js.TypeNumber {
		r.base = state.Get(stateKey).Int() - r.at
	}
	h.write("replaceState", r)
}

func (h *tabHistory) pushed(r *Router) {
	if h.r == r && !h.popping {
		h.write("pushState", r)
	}
}

func (h *tabHistory) replaced(r *Router) {
	if h.r == r && !h.popping {
		h.write("replaceState", r)
	}
}

// went follows the router's Back and Forward: the browser goes as far,
// and its popstate finds the router there already.
func (h *tabHistory) went(r *Router, n int) {
	if h.r == r && !h.popping {
		js.Global().Get("history").Call("go", n)
	}
}

// write records the router's entry in the browser's current one
// (replaceState) or a new one (pushState).
func (h *tabHistory) write(method string, r *Router) {
	state := map[string]any{stateKey: r.base + r.at}
	js.Global().Get("history").Call(method, state, "", "#"+r.current().loc)
}

// popped shows the router's entry the browser went to: by its number,
// where the router has it; next to the router's first or last entry, as
// after a reload, where the router had none before, a new one there; and
// a location typed into the address bar, as a page pushed.
func (h *tabHistory) popped(state js.Value, loc string) {
	r := h.r
	h.popping = true
	defer func() { h.popping = false }()
	if loc == "" {
		loc = "/"
	}
	if state.Type() != js.TypeObject || state.Get(stateKey).Type() != js.TypeNumber {
		r.Push(loc)
		h.write("replaceState", r)
		return
	}
	k := state.Get(stateKey).Int() - r.base
	r.current()
	switch {
	case k >= 0 && k < len(r.entries) && r.entries[k].loc == loc:
		r.Go(k - r.at)
	case k == -1:
		e := &routeEntry{loc: loc}
		e.pages, e.layouts = shared(r.entries[0], pathOf(loc))
		r.entries = append([]*routeEntry{e}, r.entries...)
		r.base--
		r.at = 0
		r.changed()
	case k == len(r.entries):
		r.at = len(r.entries) - 1
		r.Push(loc)
	default:
		// Out of what the router knows: the entry starts its history anew.
		r.entries, r.at, r.base = []*routeEntry{{loc: loc}}, 0, k+r.base
		r.changed()
	}
}

// historyState returns history.state.
func historyState() js.Value { return js.Global().Get("history").Get("state") }

// fragment returns the location in the URL's fragment, "" without one:
// "#/notes/42" is "/notes/42".
func fragment() string {
	hash := js.Global().Get("location").Get("hash").String()
	loc, ok := strings.CutPrefix(hash, "#")
	if !ok || !strings.HasPrefix(loc, "/") {
		return ""
	}
	return loc
}
