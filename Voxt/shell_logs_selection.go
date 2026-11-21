package main

// The logs' selection over actnowkit's listsel (the ActNow family's
// log-selection spec). View files may not import actnowkit, so the
// listsel calls live here, behind methods on the selection (the views'
// local state); shell_logs_select.go's views keep the pointer's events
// and call them with plain values.

import (
	"github.com/hehehai/actnowkit/listsel"
)

// logSelection is the logs view's selection, with the app's state.
type logSelection struct {
	m listsel.Model[uint64] // by the line's Seq
	// events are the rows column's pointer events since the last build,
	// in its box's coordinates.
	events []logPointer
	// spans are where the rows were in the list's box (the rows it
	// built, top to bottom), as built last, and origin where the list was
	// in the window.
	spans  []logSpan
	origin uiRect
	// views are the page's and the panel's lists (shell_logs_order.go).
	views [2]logListView
	// drag is where the pointer was in the list's box, and the box's
	// height, while a drag selects: for logsAutoScroll in the next build.
	drag struct{ y, h float32 }
}

type logPointer struct {
	kind uiInputKind
	x, y float32
	mods uiModifiers
}

type logSpan struct {
	seq         uint64
	top, bottom float32
}

// IsSelected reports whether the line is selected (a row's tint).
func (s *logSelection) IsSelected(seq uint64) bool { return s.m.IsSelected(seq) }

// Count is how many lines are selected.
func (s *logSelection) Count() int { return s.m.Count() }

// Dragging reports whether a drag is selecting (the rows' hover waits,
// the list's edges scroll).
func (s *logSelection) Dragging() bool { return s.m.Dragging() }

// Pressing reports whether the pointer is down.
func (s *logSelection) Pressing() bool { return s.m.Pressed() }

// Empty reports whether nothing is selected.
func (s *logSelection) Empty() bool { return s.m.Empty() }

// prune forgets the lines filtered out or gone from the log.
func (s *logSelection) prune(lines []logLineViewState) { s.m.Prune(logSeqs(lines)) }

// selectAllLines selects every line shown.
func (s *logSelection) selectAllLines(lines []logLineViewState) { s.m.SelectAll(logSeqs(lines)) }

// selectedSeqs are the seqs of the lines selected, in their order.
func (s *logSelection) selectedSeqs(lines []logLineViewState) []uint64 {
	return logSeqs(listsel.Items(&s.m, lines, entrySeq))
}

// clearSelection drops the selection.
func (s *logSelection) clearSelection() { s.m.Clear() }

// cancelSelection ends a press or drag without selecting; it reports
// whether there was one (else the caller clears).
func (s *logSelection) cancelSelection() bool { return s.m.Cancel() }

// pressAt starts a click at a row: shift extends from the last click,
// cmd adds or takes out the one row.
func (s *logSelection) pressAt(seq uint64, x, y float32, shift, cmd bool) {
	var mods listsel.Mods
	if shift {
		mods |= listsel.Extend
	}
	if cmd {
		mods |= listsel.Toggle
	}
	s.m.Press(seq, x, y, mods)
}

// moveTo moves the press or drag over a row (ok false: past the rows).
// It reports whether the selection changed.
func (s *logSelection) moveTo(x, y float32, seq uint64, ok bool, lines []logLineViewState) bool {
	return s.m.Move(x, y, seq, ok, logSeqs(lines))
}

// releaseAt ends the press: copyRow is true when a plain click releases
// on its row, which copies it, and seq is that row's.
func (s *logSelection) releaseAt(lines []logLineViewState) (copyRow bool, seq uint64) {
	in, k := s.m.Release(logSeqs(lines))
	return in == listsel.CopyRow, k
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
