package main

// The logs view's lines, chosen as in Finder (actnowkit's listsel, the
// ActNow family's log-selection spec, over the methods in
// shell_logs_selection.go): a click copies its line, Shift-click selects
// from that line, ⌘-click adds or takes out one, dragging selects the
// lines it passes, ⌘A every line shown and Esc none. Copying clears the
// selection. The context menu (right-click, Control-click) copies the
// line, the selection or every line shown.
//
// The rows' list takes the pointer (HandleInput) and keeps its events for
// the next build, which reads where the rows were as the user acted
// (Bounds: the frame on screen) once it has built them. The list shows the
// newest line first (shell_logs_order.go); copies are in the log's order,
// the oldest first.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// logsInput takes the rows column's input: the primary button's presses,
// moves and releases (the secondary one is the context menu's), and the
// Edit menu's Copy and Select All, which come as commands on macOS.
func logsInput(s *logSelection, lines []logLineViewState, actions logsActions) func(ui.InputEvent) bool {
	return func(ev ui.InputEvent) bool {
		switch ev.Kind {
		case ui.InputPointerDown, ui.InputPointerUp:
			if ev.Button != 0 {
				return false
			}
		case ui.InputPointerMove:
			if ev.Button != 0 {
				return false // the hover's moves (-1): the rows look at their own
			}
			if n := len(s.events); n > 0 && s.events[n-1].kind == ui.InputPointerMove {
				s.events[n-1].x, s.events[n-1].y = ev.X, ev.Y // the drag only needs the last
				return true
			}
		case ui.InputCommand:
			switch {
			case ev.Text == "copy" && !s.Empty():
				actions.copyLogLines(s.selectedSeqs(lines))
				return true
			case ev.Text == "selectAll" && len(lines) > 0:
				s.selectAllLines(lines)
				return true
			}
			return false
		default:
			return false
		}
		s.events = append(s.events, logPointer{kind: ev.Kind, x: ev.X, y: ev.Y, mods: ev.Mods})
		return true
	}
}

// logsResolve runs the pointer's events against the rows just built
// (s.spans) and reports whether the selection may have changed: the view
// then builds once more to show it. col is the rows' list, for where the
// pointer is now while it presses (the log scrolled under it).
func logsResolve(c *ui.Context, col ui.Element, lines []logLineViewState, actions logsActions, s *logSelection) bool {
	if len(s.events) == 0 && !s.Pressing() {
		return false
	}
	changed := false
	for _, ev := range s.events {
		switch ev.kind {
		case ui.InputPointerDown:
			seq, ok := s.hit(ev.y, false)
			if !ok {
				// Under the last line: nothing to copy.
				changed = changed || !s.Empty()
				s.clearSelection()
				continue
			}
			s.pressAt(seq, ev.x, ev.y, ev.mods&ui.Shift != 0, ev.mods&ui.Cmd != 0)
		case ui.InputPointerMove:
			seq, ok := s.hit(ev.y, true)
			changed = s.moveTo(ev.x, ev.y, seq, ok, lines) || changed
		case ui.InputPointerUp:
			if !s.Pressing() {
				continue
			}
			seq, ok := s.hit(ev.y, true)
			s.moveTo(ev.x, ev.y, seq, ok, lines) // a flick within one frame
			copyRow, k := s.releaseAt(lines)
			changed = true
			if copyRow {
				actions.copyLogLines([]uint64{k})
			}
		}
	}
	s.events = s.events[:0]
	if s.Dragging() {
		// The log may have scrolled under a pointer at rest.
		x, y, _ := col.PointerPosition()
		seq, ok := s.hit(y, true)
		changed = s.moveTo(x, y, seq, ok, lines) || changed
		s.drag.y, s.drag.h = y, col.Bounds().H
	}
	return changed
}

// logsAutoScroll scrolls the list (its ScrollState st) while a drag holds
// the pointer within 24 DIPs of its top or bottom edge (or past them),
// faster farther on: y is the pointer in the list's box and h the box's
// height, as the last frame had them (s.drag). It runs before the list is
// built, which takes the offset; it reports whether the pointer is at an
// edge, for frames to keep coming while it is.
func logsAutoScroll(t *ui.Theme, y, h float32, st *ui.ScrollState) bool {
	if h <= 0 {
		return false
	}
	edge := px(t, 24)
	var dy float32
	switch {
	case y < edge:
		dy = y - edge
	case y > h-edge:
		dy = y - (h - edge)
	default:
		return false
	}
	step := min(max(abs32(dy)/3, 2), 24)
	if dy < 0 {
		step = -step
	}
	st.Y = min(max(st.Y+step, 0), st.MaxY)
	return true
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// logsKeys gives the rows' list ⌘A, ⌘C and Esc while it has the focus
// (on macOS the Edit menu takes ⌘A and ⌘C: logsInput).
func logsKeys(c *ui.Context, col ui.Element, lines []logLineViewState, actions logsActions, s *logSelection) {
	if col.Shortcut(ui.Cmd, ui.KeyA) && len(lines) > 0 {
		s.selectAllLines(lines)
	}
	if col.Shortcut(ui.Cmd, ui.KeyC) && !s.Empty() {
		actions.copyLogLines(s.selectedSeqs(lines))
	}
	// Esc only while there is something to give up: else it is the
	// window's (a dialog, the palette).
	if (s.Pressing() || !s.Empty()) && col.Shortcut(0, ui.KeyEscape) {
		if !s.cancelSelection() {
			s.clearSelection()
		}
		s.events = s.events[:0]
	}
}

// logRowMenu is a row's context menu: this line, the lines selected, every
// line shown.
func logRowMenu(c *ui.Context, row ui.Element, e logLineViewState, lines []logLineViewState, actions logsActions, s *logSelection) {
	row.ContextMenu(func(m *ui.Menu) {
		if m.Item(i18n.T("Copy this line")).Chosen() {
			actions.copyLogLines([]uint64{e.Seq})
		}
		if n := s.Count(); n > 0 {
			if m.Item(i18n.N(n, "Copy %d selected line", "Copy %d selected lines", n)).Shortcut(ui.Cmd, ui.KeyC).Chosen() {
				actions.copyLogLines(s.selectedSeqs(lines))
			}
		}
		if m.Item(i18n.T("Copy all lines shown")).Chosen() {
			actions.copyLogLines(logSeqs(lines))
		}
	})
}

// logsCheck is the logs' screenshot run of -tags logscheck builds
// (logs_check.go, scripts/mac-logs-shots.sh); nil in the others.
var logsCheck func(dir string)
