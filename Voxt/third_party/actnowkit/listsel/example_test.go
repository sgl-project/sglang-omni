package listsel_test

import (
	"fmt"

	"github.com/hehehai/actnowkit/listsel"
)

type logLine struct {
	Seq uint64 // stable: the log's sequence number
	Msg string
}

// A log view: the pointer's events come with the row under it; a plain
// click copies its line, a drag selects, ⌘C copies the selection.
func Example() {
	lines := []logLine{{10, "start"}, {11, "load model"}, {12, "ready"}, {13, "dictation"}}
	order := make([]uint64, len(lines))
	for i, l := range lines {
		order[i] = l.Seq
	}
	var sel listsel.Model[uint64]

	// A click on "load model".
	sel.Press(11, 5, 20, 0)
	if in, k := sel.Release(order); in == listsel.CopyRow {
		fmt.Println("copy row", k)
	}

	// A drag from "ready" up to "start".
	sel.Press(12, 5, 40, 0)
	sel.Move(5, 30, 11, true, order)
	sel.Move(5, 5, 10, true, order)
	sel.Release(order)
	fmt.Println("selected", sel.Count(), sel.Selected(order))

	// ⌘C: copy them, then clear.
	picked := listsel.Items(&sel, lines, func(l logLine) uint64 { return l.Seq })
	fmt.Print(listsel.Text(picked, func(l logLine) string { return l.Msg }))
	sel.Clear()
	fmt.Println("after copy", sel.Count())
	// Output:
	// copy row 11
	// selected 3 [10 11 12]
	// start
	// load model
	// ready
	// after copy 0
}
