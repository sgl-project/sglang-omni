package main

// The logs view's lines, chosen as in Finder (actnowkit's listsel, the
// ActNow family's log-selection spec): a click copies its line, Shift-
// click selects from that line, ⌘-click adds or takes out one, dragging
// selects the lines it passes, ⌘A every line shown and Esc none. Copying
// clears the selection. The context menu (right-click, Control-click)
// copies the line, the selection or every line shown.
//
// The rows' list takes the pointer (HandleInput) and keeps its events for
// the next build, which reads where the rows were as the user acted
// (Bounds: the frame on screen) once it has built them. The list shows the
// newest line first (shell_logs_order.go); copies are in the log's order,
// the oldest first.

import (
	"cmp"
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/listsel"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/logs"
)

// logSelection is the logs view's selection, with the app's state.
type logSelection struct {
	m listsel.Model[uint64] // by core.LogEntry.Seq
	// events are the rows column's pointer events since the last build,
	// in its box's coordinates.
	events []logPointer
	// spans are where the rows were in the list's box (the rows it
	// built, top to bottom), as built last, and origin where the list was
	// in the window.
	spans  []logSpan
	origin ui.Rect
	// views are the page's and the panel's lists (shell_logs_order.go).
	views [2]logListView
	// drag is where the pointer was in the list's box, and the box's
	// height, while a drag selects: for logsAutoScroll in the next build.
	drag struct{ y, h float32 }
}

type logPointer struct {
	kind ui.InputKind
	x, y float32
	mods ui.Modifiers
}

type logSpan struct {
	seq         uint64
	top, bottom float32
}

// logSeqs are the keys of lines, in their order.
func logSeqs(lines []core.LogEntry) []uint64 {
	out := make([]uint64, len(lines))
	for i, e := range lines {
		out[i] = e.Seq
	}
	return out
}

// logsInput takes the rows column's input: the primary button's presses,
// moves and releases (the secondary one is the context menu's), and the
// Edit menu's Copy and Select All, which come as commands on macOS.
func (a *app) logsInput(svc ui.Services, lines []core.LogEntry) func(ui.InputEvent) bool {
	return func(ev ui.InputEvent) bool {
		s := &a.logSel
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
			case ev.Text == "copy" && !s.m.Empty():
				a.copyLogLines(svc, listsel.Items(&s.m, lines, entrySeq))
				return true
			case ev.Text == "selectAll" && len(lines) > 0:
				s.m.SelectAll(logSeqs(lines))
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

func entrySeq(e core.LogEntry) uint64 { return e.Seq }

// logsResolve runs the pointer's events against the rows just built
// (s.spans) and reports whether the selection may have changed: the view
// then builds once more to show it. col is the rows' list, for where the
// pointer is now while it presses (the log scrolled under it).
func (a *app) logsResolve(c *ui.Context, col ui.Element, lines []core.LogEntry) bool {
	s := &a.logSel
	if len(s.events) == 0 && !s.m.Pressed() {
		return false
	}
	order := logSeqs(lines)
	changed := false
	for _, ev := range s.events {
		switch ev.kind {
		case ui.InputPointerDown:
			seq, ok := s.hit(ev.y, false)
			if !ok {
				// Under the last line: nothing to copy.
				changed = changed || !s.m.Empty()
				s.m.Clear()
				continue
			}
			var mods listsel.Mods
			if ev.mods&ui.Shift != 0 {
				mods |= listsel.Extend
			}
			if ev.mods&ui.Cmd != 0 {
				mods |= listsel.Toggle
			}
			s.m.Press(seq, ev.x, ev.y, mods)
		case ui.InputPointerMove:
			seq, ok := s.hit(ev.y, true)
			changed = s.m.Move(ev.x, ev.y, seq, ok, order) || changed
		case ui.InputPointerUp:
			if !s.m.Pressed() {
				continue
			}
			seq, ok := s.hit(ev.y, true)
			s.m.Move(ev.x, ev.y, seq, ok, order) // a flick within one frame
			in, k := s.m.Release(order)
			changed = true
			if in == listsel.CopyRow {
				for _, e := range lines {
					if e.Seq == k {
						a.copyLogLines(c.Services(), []core.LogEntry{e})
					}
				}
			}
		}
	}
	s.events = s.events[:0]
	if s.m.Dragging() {
		// The log may have scrolled under a pointer at rest.
		x, y, _ := col.PointerPosition()
		seq, ok := s.hit(y, true)
		changed = s.m.Move(x, y, seq, ok, order) || changed
		s.drag.y, s.drag.h = y, col.Bounds().H
	}
	return changed
}

// hit returns the row at y in the column (a row takes the gap above it).
// Above the first row and under the last, clamp gives those (a drag's
// rows); else there is none.
func (s *logSelection) hit(y float32, clamp bool) (uint64, bool) {
	n := len(s.spans)
	if n == 0 {
		return 0, false
	}
	if y < s.spans[0].top {
		return s.spans[0].seq, clamp
	}
	for _, sp := range s.spans {
		if y < sp.bottom {
			return sp.seq, true
		}
	}
	return s.spans[n-1].seq, clamp
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

// copyLogLines copies lines as the log file has them, the oldest first
// whatever order they show in, says how many, and clears the selection.
func (a *app) copyLogLines(svc ui.Services, lines []core.LogEntry) {
	if len(lines) == 0 {
		return
	}
	lines = slices.SortedFunc(slices.Values(lines), func(x, y core.LogEntry) int { return cmp.Compare(x.Seq, y.Seq) })
	svc.WriteClipboard(listsel.Text(lines, logs.Format))
	a.toast(i18n.N(len(lines), "Copied %d line", "Copied %d lines", len(lines)))
	a.logSel.m.Clear()
}

// logsKeys gives the rows' list ⌘A, ⌘C and Esc while it has the focus
// (on macOS the Edit menu takes ⌘A and ⌘C: logsInput).
func (a *app) logsKeys(c *ui.Context, col ui.Element, lines []core.LogEntry) {
	s := &a.logSel
	if col.Shortcut(ui.Cmd, ui.KeyA) && len(lines) > 0 {
		s.m.SelectAll(logSeqs(lines))
	}
	if col.Shortcut(ui.Cmd, ui.KeyC) && !s.m.Empty() {
		a.copyLogLines(c.Services(), listsel.Items(&s.m, lines, entrySeq))
	}
	// Esc only while there is something to give up: else it is the
	// window's (a dialog, the palette).
	if (s.m.Pressed() || !s.m.Empty()) && col.Shortcut(0, ui.KeyEscape) {
		if !s.m.Cancel() {
			s.m.Clear()
		}
		s.events = s.events[:0]
	}
}

// logRowMenu is a row's context menu: this line, the lines selected, every
// line shown.
func (a *app) logRowMenu(c *ui.Context, row ui.Element, e core.LogEntry, lines []core.LogEntry) {
	svc := c.Services()
	row.ContextMenu(func(m *ui.Menu) {
		s := &a.logSel
		if m.Item(i18n.T("Copy this line")).Chosen() {
			a.copyLogLines(svc, []core.LogEntry{e})
		}
		if n := s.m.Count(); n > 0 {
			if m.Item(i18n.N(n, "Copy %d selected line", "Copy %d selected lines", n)).Shortcut(ui.Cmd, ui.KeyC).Chosen() {
				a.copyLogLines(svc, listsel.Items(&s.m, lines, entrySeq))
			}
		}
		if m.Item(i18n.T("Copy all lines shown")).Chosen() {
			a.copyLogLines(svc, lines)
		}
	})
}

// logsCheck is the logs' screenshot run of -tags logscheck builds
// (logs_check.go, scripts/mac-logs-shots.sh); nil in the others.
var logsCheck func(dir string)
