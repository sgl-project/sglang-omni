package gamepad

import (
	"strconv"
	"syscall/js"
)

// Browsers report pads through navigator.getGamepads(), each at an index
// for as long as it is connected. Those with the "standard" mapping have
// their buttons and axes in this package's order; others keep the order
// the browser gives, which fits the first buttons and axes of most.

type jsPad struct {
	index int
	id    string
}

func pollPads() js.Value {
	nav := js.Global().Get("navigator")
	if !nav.Truthy() || !nav.Get("getGamepads").Truthy() {
		return js.Undefined()
	}
	return nav.Call("getGamepads")
}

func scan(known map[string]device) []device {
	list := pollPads()
	if !list.Truthy() {
		return nil
	}
	var found []device
	for i := range list.Length() {
		g := list.Index(i)
		if !g.Truthy() || !g.Get("connected").Bool() {
			continue
		}
		d := &jsPad{index: i, id: g.Get("id").String()}
		if k, ok := known[d.key()]; ok {
			found = append(found, k)
			continue
		}
		found = append(found, d)
	}
	return found
}

func (d *jsPad) key() string  { return strconv.Itoa(d.index) + "\x00" + d.id }
func (d *jsPad) name() string { return d.id }
func (d *jsPad) close()       {}

func (d *jsPad) read(s *State) bool {
	list := pollPads()
	if !list.Truthy() || d.index >= list.Length() {
		return false
	}
	g := list.Index(d.index)
	if !g.Truthy() || !g.Get("connected").Bool() || g.Get("id").String() != d.id {
		return false
	}
	buttons := g.Get("buttons")
	for i := range min(buttons.Length(), int(ButtonCount)) {
		b := buttons.Index(i)
		s.Buttons[i] = float32(b.Get("value").Float())
		if s.Buttons[i] == 0 && b.Get("pressed").Bool() {
			s.Buttons[i] = 1 // a browser that reports only pressed
		}
	}
	axes := g.Get("axes")
	for i := range min(axes.Length(), int(AxisCount)) {
		s.Axes[i] = float32(axes.Index(i).Float())
	}
	return true
}
