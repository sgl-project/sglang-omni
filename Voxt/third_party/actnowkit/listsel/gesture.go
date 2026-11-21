package listsel

// The pointer gesture: the app reports the primary button's press on a
// row, the pointer's moves while it is held, and the release; the Model
// tells a click (copy, Shift-click, ⌘-click) from a drag (select a range).
// The secondary button (and Control-click on macOS) is the context menu:
// do not report it.

// Press is the primary button going down on row k at (x, y), in any fixed
// coordinates (DIPs), with mods held. Nothing changes until the pointer
// moves (Move) or is released (Release).
func (m *Model[K]) Press(k K, x, y float32, mods Mods) {
	m.press = press[K]{on: true, key: k, x: x, y: y, mods: mods}
	m.drag = false
}

// Move is the pointer at (x, y) while the button is held, over row over
// when ok (not over a row: outside the list, or between rows). Once it is
// more than DragSlop away from the press, the press is a drag: from the
// row pressed to the row under the pointer, the rows are selected (added
// to the selection with Toggle held). Move reports whether the selection
// may have changed: the view should draw again.
func (m *Model[K]) Move(x, y float32, over K, ok bool, order []K) bool {
	if !m.press.on {
		return false
	}
	changed := false
	if !m.drag {
		dx, dy := x-m.press.x, y-m.press.y
		if dx*dx+dy*dy <= DragSlop*DragSlop {
			return false
		}
		m.before = clone(m.sel)
		m.DragStart(m.press.key, m.press.mods&Toggle != 0)
		changed = true
	}
	if ok {
		m.DragOver(over, order)
		changed = true
	}
	return changed
}

// Release is the button going up. After a drag, the rows stay selected
// (None). Else it was a click on the row pressed, with the modifiers held
// as it was pressed (ClickWith): a plain click returns CopyRow and the row
// to copy.
func (m *Model[K]) Release(order []K) (Intent, K) {
	p := m.press
	m.press = press[K]{}
	if m.drag {
		m.DragEnd()
		m.before = nil
		var zero K
		return None, zero
	}
	if !p.on {
		var zero K
		return None, zero
	}
	return m.ClickWith(p.key, p.mods, order), p.key
}

// Cancel gives up the press (Esc while dragging, the pointer lost): a drag
// puts back the selection it started from. It reports whether there was a
// press to give up.
func (m *Model[K]) Cancel() bool {
	if !m.press.on && !m.drag {
		return false
	}
	if m.drag {
		m.sel, m.base = m.before, clone(m.before)
		m.drag = false
	}
	m.press, m.before = press[K]{}, nil
	return true
}
