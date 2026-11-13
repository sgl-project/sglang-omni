package listsel

import (
	"fmt"
	"slices"
	"strings"
	"testing"
)

// rows returns the keys from..to, as the sequence numbers of log lines.
func rows(from, to int) []int {
	var out []int
	for k := from; k <= to; k++ {
		out = append(out, k)
	}
	return out
}

func check(t *testing.T, m *Model[int], order []int, want ...int) {
	t.Helper()
	got := m.Selected(order)
	if want == nil {
		want = []int{}
	}
	if got == nil {
		got = []int{}
	}
	if !slices.Equal(got, want) {
		t.Fatalf("selected %v, want %v", got, want)
	}
	if m.Count() != len(want) {
		t.Fatalf("count %d, want %d", m.Count(), len(want))
	}
}

func anchor(t *testing.T, m *Model[int], want int) {
	t.Helper()
	if a, ok := m.Anchor(); !ok || a != want {
		t.Fatalf("anchor %d %v, want %d", a, ok, want)
	}
}

func TestZero(t *testing.T) {
	var m Model[int]
	order := rows(1, 5)
	check(t, &m, order)
	if !m.Empty() || m.IsSelected(1) || m.Dragging() || m.Pressed() {
		t.Fatal("zero model not empty")
	}
	if _, ok := m.Anchor(); ok {
		t.Fatal("zero model has an anchor")
	}
	m.Clear()
	m.Prune(order)
	m.DragOver(3, order) // no drag: nothing
	m.DragEnd()
	if m.Cancel() {
		t.Fatal("Cancel without a press")
	}
	if in, _ := m.Release(order); in != None {
		t.Fatal("Release without a press")
	}
	check(t, &m, order)
}

func TestClickCopiesAndAnchors(t *testing.T) {
	var m Model[int]
	order := rows(1, 10)
	if in := m.Click(4); in != CopyRow {
		t.Fatalf("intent %v", in)
	}
	check(t, &m, order) // the row clicked does not show selected
	anchor(t, &m, 4)
	m.ShiftClick(7, order)
	check(t, &m, order, 4, 5, 6, 7)
	anchor(t, &m, 4)
	m.ShiftClick(2, order) // the range replaces the last one
	check(t, &m, order, 2, 3, 4)
	// A plain click drops the selection and moves the anchor.
	if in := m.Click(9); in != CopyRow {
		t.Fatalf("intent %v", in)
	}
	check(t, &m, order)
	anchor(t, &m, 9)
	m.ShiftClick(10, order)
	check(t, &m, order, 9, 10)
}

func TestShiftClickWithoutAnchor(t *testing.T) {
	var m Model[int]
	order := rows(1, 5)
	m.ShiftClick(3, order)
	check(t, &m, order, 3)
	anchor(t, &m, 3)
	m.ShiftClick(5, order)
	check(t, &m, order, 3, 4, 5)
	m.ShiftClick(3, order)
	check(t, &m, order, 3)
}

func TestToggle(t *testing.T) {
	var m Model[int]
	order := rows(1, 10)
	m.ToggleClick(2)
	m.ToggleClick(5)
	m.ToggleClick(8)
	check(t, &m, order, 2, 5, 8)
	anchor(t, &m, 8)
	m.ToggleClick(5)
	check(t, &m, order, 2, 8)
	anchor(t, &m, 5)
	// Shift extends from the last toggled row, keeping the others.
	m.ShiftClick(7, order)
	check(t, &m, order, 2, 5, 6, 7, 8)
	m.ShiftClick(3, order)
	check(t, &m, order, 2, 3, 4, 5, 8)
	// A toggle after a range keeps the range.
	m.ToggleClick(10)
	check(t, &m, order, 2, 3, 4, 5, 8, 10)
	m.ShiftClick(9, order)
	check(t, &m, order, 2, 3, 4, 5, 8, 9, 10)
	// Toggling the only row off empties the selection.
	m.Clear()
	m.ToggleClick(1)
	m.ToggleClick(1)
	check(t, &m, order)
	anchor(t, &m, 1)
}

func TestClickWith(t *testing.T) {
	var m Model[int]
	order := rows(1, 6)
	if in := m.ClickWith(2, 0, order); in != CopyRow {
		t.Fatalf("plain: %v", in)
	}
	if in := m.ClickWith(4, Extend, order); in != None {
		t.Fatalf("shift: %v", in)
	}
	check(t, &m, order, 2, 3, 4)
	if in := m.ClickWith(6, Toggle, order); in != None {
		t.Fatalf("toggle: %v", in)
	}
	check(t, &m, order, 2, 3, 4, 6)
	// Shift wins over Toggle, from the anchor (6).
	m.ClickWith(5, Extend|Toggle, order)
	check(t, &m, order, 2, 3, 4, 5, 6)
}

func TestSelectAllClear(t *testing.T) {
	var m Model[int]
	order := rows(1, 4)
	m.SelectAll(order)
	check(t, &m, order, 1, 2, 3, 4)
	anchor(t, &m, 1) // no anchor before: the first row
	m.Clear()
	check(t, &m, order)
	anchor(t, &m, 1) // Clear keeps the anchor
	m.Click(3)
	m.SelectAll(order)
	anchor(t, &m, 3)
	// After ⌘A, Shift-click selects from the anchor only.
	m.ShiftClick(4, order)
	check(t, &m, order, 3, 4)
	m.SelectAll(nil)
	check(t, &m, order)
	m.Reset()
	if _, ok := m.Anchor(); ok || !m.Empty() {
		t.Fatal("Reset kept something")
	}
}

func TestDragDown(t *testing.T) {
	var m Model[int]
	order := rows(1, 10)
	m.ToggleClick(9) // replaced by a plain drag
	m.Press(3, 10, 30, 0)
	if !m.Pressed() || m.Dragging() {
		t.Fatal("press")
	}
	// Within the slop: still a click.
	if m.Move(12, 32, 3, true, order) {
		t.Fatal("moved within the slop")
	}
	check(t, &m, order, 9)
	if !m.Move(12, 45, 4, true, order) || !m.Dragging() {
		t.Fatal("no drag past the slop")
	}
	check(t, &m, order, 3, 4)
	m.Move(12, 70, 6, true, order)
	check(t, &m, order, 3, 4, 5, 6)
	// Between rows or outside the list: the selection stays.
	m.Move(12, 75, 0, false, order)
	check(t, &m, order, 3, 4, 5, 6)
	// Back up: the range shrinks.
	m.Move(12, 50, 4, true, order)
	check(t, &m, order, 3, 4)
	if in, _ := m.Release(order); in != None {
		t.Fatalf("a drag copied: %v", in)
	}
	if m.Dragging() || m.Pressed() {
		t.Fatal("still dragging")
	}
	check(t, &m, order, 3, 4)
	anchor(t, &m, 3)
	// Shift-click extends from where the drag started.
	m.ShiftClick(7, order)
	check(t, &m, order, 3, 4, 5, 6, 7)
}

func TestDragUp(t *testing.T) {
	var m Model[int]
	order := rows(1, 10)
	m.Press(8, 0, 80, 0)
	m.Move(0, 60, 6, true, order)
	m.Move(0, 20, 2, true, order)
	check(t, &m, order, 2, 3, 4, 5, 6, 7, 8)
	// Past the first row and back down below it.
	m.Move(0, 95, 10, true, order)
	check(t, &m, order, 8, 9, 10)
	m.Release(order)
	check(t, &m, order, 8, 9, 10)
	anchor(t, &m, 8)
	// The slop counts in any direction, sideways too.
	m.Press(1, 0, 0, 0)
	m.Move(5, 0, 1, true, order)
	if !m.Dragging() {
		t.Fatal("no drag sideways")
	}
	check(t, &m, order, 1)
	m.Release(order)
	check(t, &m, order, 1)
}

func TestDragAdds(t *testing.T) {
	var m Model[int]
	order := rows(1, 10)
	m.ToggleClick(1)
	m.Press(5, 0, 50, Toggle)
	m.Move(0, 70, 7, true, order)
	check(t, &m, order, 1, 5, 6, 7)
	m.Release(order)
	check(t, &m, order, 1, 5, 6, 7)
	// A drag of one's own explicitly.
	m.DragStart(9, true)
	m.DragOver(10, order)
	m.DragEnd()
	check(t, &m, order, 1, 5, 6, 7, 9, 10)
	m.DragStart(2, false)
	m.DragOver(3, order)
	m.DragEnd()
	check(t, &m, order, 2, 3)
}

func TestCancel(t *testing.T) {
	var m Model[int]
	order := rows(1, 6)
	m.ToggleClick(6)
	m.Press(1, 0, 0, 0)
	m.Move(0, 30, 3, true, order)
	check(t, &m, order, 1, 2, 3)
	if !m.Cancel() {
		t.Fatal("no drag to cancel")
	}
	check(t, &m, order, 6)
	if in, _ := m.Release(order); in != None {
		t.Fatal("release after cancel did something")
	}
	// A press cancelled before it moves copies nothing.
	m.Press(2, 0, 0, 0)
	if !m.Cancel() {
		t.Fatal("no press to cancel")
	}
	if in, _ := m.Release(order); in != None {
		t.Fatal("release after cancel copied")
	}
	check(t, &m, order, 6)
}

func TestReleaseIsClick(t *testing.T) {
	var m Model[int]
	order := rows(1, 6)
	m.Press(2, 0, 0, 0)
	m.Move(1, 1, 2, true, order) // a jitter
	if in, k := m.Release(order); in != CopyRow || k != 2 {
		t.Fatalf("intent %v %d", in, k)
	}
	anchor(t, &m, 2)
	m.Press(5, 0, 0, Extend)
	if in, _ := m.Release(order); in != None {
		t.Fatal("shift-click copied")
	}
	check(t, &m, order, 2, 3, 4, 5)
	m.Press(3, 0, 0, Toggle)
	m.Release(order)
	check(t, &m, order, 2, 4, 5)
	// The modifiers are the press's: a release with others is no matter
	// here, the Model never sees them.
	m.Press(6, 0, 0, 0)
	if in, k := m.Release(order); in != CopyRow || k != 6 {
		t.Fatalf("intent %v %d", in, k)
	}
	check(t, &m, order)
}

func TestPruneAfterFilter(t *testing.T) {
	var m Model[int]
	all := rows(1, 10)
	m.ShiftClick(2, all)
	m.ShiftClick(8, all)
	check(t, &m, all, 2, 3, 4, 5, 6, 7, 8)
	// A filter shows the even rows only.
	even := []int{2, 4, 6, 8, 10}
	if got := m.Selected(even); !slices.Equal(got, []int{2, 4, 6, 8}) {
		t.Fatalf("selected of the shown %v", got)
	}
	m.Prune(even)
	check(t, &m, all, 2, 4, 6, 8)
	anchor(t, &m, 2)
	// Shift-click spans the rows shown.
	m.ShiftClick(6, even)
	check(t, &m, all, 2, 4, 6)
	// The anchor filtered out: Shift-click starts over.
	odd := []int{1, 3, 5, 7, 9}
	m.Prune(odd)
	check(t, &m, all)
	if _, ok := m.Anchor(); ok {
		t.Fatal("anchor kept though not shown")
	}
	m.ShiftClick(5, odd)
	check(t, &m, all, 5)
	m.ShiftClick(9, odd)
	check(t, &m, all, 5, 7, 9)
}

func TestPruneDuringDrag(t *testing.T) {
	var m Model[int]
	order := rows(1, 10)
	m.Press(3, 0, 0, 0)
	m.Move(0, 50, 5, true, order)
	m.Prune(rows(4, 10)) // the drag's first row scrolled out of the log
	if m.Dragging() || m.Pressed() {
		t.Fatal("drag kept without its first row")
	}
	check(t, &m, order, 4, 5)
	if in, _ := m.Release(order); in != None {
		t.Fatal("release copied")
	}
	// A press whose row went copies nothing.
	m.Press(4, 0, 0, 0)
	m.Prune(rows(5, 10))
	if in, _ := m.Release(order); in != None {
		t.Fatal("release of a pruned row copied")
	}
}

func TestNewRowsAppended(t *testing.T) {
	var m Model[int]
	order := rows(1, 5)
	m.Click(2)
	m.ShiftClick(4, order)
	// New lines come at the end, and the oldest leave a ring of five.
	order = rows(3, 8)
	m.Prune(order)
	check(t, &m, order, 3, 4)
	if _, ok := m.Anchor(); ok {
		t.Fatal("anchor 2 left the log")
	}
	m.ToggleClick(7)
	check(t, &m, order, 3, 4, 7)
	// The keys do not move with the places: the same rows stay selected
	// whatever comes before them.
	order = append([]int{0}, order...)
	check(t, &m, order, 3, 4, 7)
	// Dragging while lines arrive: the range follows the keys.
	m.Clear()
	m.Press(6, 0, 0, 0)
	m.Move(0, 20, 8, true, order)
	order = append(order, 9, 10)
	m.Move(0, 40, 10, true, order)
	check(t, &m, order, 6, 7, 8, 9, 10)
	m.Release(order)
}

func TestStringKeys(t *testing.T) {
	var m Model[string]
	order := []string{"a", "b", "c", "d"}
	m.Click("b")
	m.ShiftClick("d", order)
	if got := m.Selected(order); !slices.Equal(got, []string{"b", "c", "d"}) {
		t.Fatalf("selected %v", got)
	}
}

func TestItemsText(t *testing.T) {
	type line struct {
		seq int
		msg string
	}
	lines := []line{{1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}}
	var m Model[int]
	if Items(&m, lines, func(l line) int { return l.seq }) != nil {
		t.Fatal("items without a selection")
	}
	m.ToggleClick(3)
	m.ToggleClick(1)
	got := Items(&m, lines, func(l line) int { return l.seq })
	text := Text(got, func(l line) string { return fmt.Sprintf("%d %s", l.seq, l.msg) })
	if text != "1 one\n3 three\n" {
		t.Fatalf("text %q", text)
	}
	if Text([]line(nil), func(l line) string { return l.msg }) != "" {
		t.Fatal("text of nothing")
	}
	if got := Text(lines[1:2], func(l line) string { return l.msg }); got != "two" {
		t.Fatalf("one line %q", got)
	}
}

func TestWordings(t *testing.T) {
	for _, w := range []Wording{TextSelected, TextCopy, TextDeselect, TextCopyLine, TextCopySelected, TextCopyShown, TextCopied} {
		if w.EN == "" || w.ZH == "" || strings.Count(w.EN, "%d") != strings.Count(w.ZH, "%d") {
			t.Errorf("wording %+v", w)
		}
	}
}
