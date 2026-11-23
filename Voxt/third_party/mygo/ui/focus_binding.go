package ui

import "reflect"

type focusMember struct {
	value   any
	element Element
	id      uint64
}
type focusField struct {
	field              reflect.Value
	zero, last, wanted any
	pending            bool
	members, committed []focusMember
}

// FocusBind binds desired focus to a comparable value in an app-owned field.
// Assign another value to request that control; its request waits while hidden.
// User focus changes update the field when no request is waiting. The field
// holds desired focus: use FocusedValue to read actual focus while a request
// waits. Each window should use its own field. Values must be unique per field.
func (e Element) FocusBind(field any, value any) Element {
	n := e.node()
	if n == nil {
		return e
	}
	rt := n.c.rt
	f := rt.focusFields[field]
	if f == nil {
		p := reflect.ValueOf(field)
		if !p.IsValid() || p.Kind() != reflect.Pointer || p.IsNil() || !p.Type().Elem().Comparable() {
			panic("ui: FocusBind needs a non-nil pointer to a comparable value")
		}
		if reflect.TypeOf(value) != p.Type().Elem() {
			panic("ui: FocusBind value must match the field type")
		}
		f = &focusField{field: p.Elem(), zero: reflect.Zero(p.Type().Elem()).Interface(), last: p.Elem().Interface(), wanted: p.Elem().Interface()}
		f.pending = f.wanted != f.zero
		if rt.focusFields == nil {
			rt.focusFields = make(map[any]*focusField)
		}
		rt.focusFields[field] = f
	}
	if reflect.TypeOf(value) != f.field.Type() {
		panic("ui: FocusBind value must match the field type")
	}
	for _, m := range f.members {
		if m.value == value {
			panic("ui: FocusBind value bound twice in one window")
		}
	}
	n.Focusable()
	f.members = append(f.members, focusMember{value: value, element: e, id: n.id})
	return e
}

// FocusedValue returns actual focus from the committed window, independently
// of construction order or a pending FocusBind request. Zero means no member
// has focus. Use only on the UI thread.
func FocusedValue[T comparable](c *Context, field *T) T {
	var zero T
	rt := c.runtime()
	if rt == nil || !rt.windowFocus() {
		return zero
	}
	f := rt.focusFields[field]
	if f == nil {
		return zero
	}
	v := rt.focusMemberValue(f.committed, f.zero)
	if v == f.zero {
		return zero
	}
	return v.(T)
}

func (rt *engine) focusMemberValue(members []focusMember, zero any) any {
	for s := rt.states[rt.focused]; s != nil; s = rt.states[s.parent] {
		for _, m := range members {
			if m.id == s.id {
				return m.value
			}
		}
		if s.parent == 0 {
			break
		}
	}
	return zero
}

func (rt *engine) applyFocusBindings() {
	for _, f := range rt.focusFields {
		v := f.field.Interface()
		if v != f.last {
			f.wanted, f.last, f.pending = v, v, v != f.zero
			if v == f.zero && rt.focusMemberValue(f.committed, f.zero) != f.zero {
				rt.focused = 0
				rt.consumed = true
			}
		}
		if f.pending {
			for _, m := range f.members {
				if m.value != f.wanted {
					continue
				}
				n := m.element.lookup()
				if n == nil || n.disabled() {
					break
				}
				if rt.focused != n.id {
					n.Focus()
					rt.consumed = true
				}
				f.pending = false
				break
			}
		}
		if f.pending {
			continue
		}
		actual := rt.focusMemberValue(f.members, f.zero)
		if actual == f.zero && f.wanted != f.zero {
			visible := false
			for _, m := range f.members {
				if m.value == f.wanted {
					visible = true
					break
				}
			}
			if !visible {
				f.pending = true
				continue
			}
		}
		if actual != f.last {
			f.field.Set(reflect.ValueOf(actual))
			f.last, f.wanted = actual, actual
			rt.consumed = true
		}
	}
}

func (rt *engine) commitFocusBindings() {
	for _, f := range rt.focusFields {
		f.committed = append(f.committed[:0], f.members...)
	}
}
