package ui

import (
	"strings"
	"time"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/text"
)

// change is an edit of the text: the runes at at held removed, and hold
// inserted since. A text the app set is a change of the whole text (app),
// which typing after it does not grow.
type change struct {
	at                int
	removed, inserted string
	app               bool
}

// undoStep is what undoing takes back: the changes of an edit, or of
// typing in a row, and where the caret and the selection were before and
// after them.
type undoStep struct {
	changes                         []change
	caret, anchor                   int
	caretAfter, anchorAfter         int
	selectionBefore, selectionAfter editorSelection
}

// before returns the text before the step's changes, from the text after
// them.
func (s *undoStep) before(text string) string {
	if len(s.changes) > 0 && s.changes[0].app {
		return s.changes[0].removed
	}
	for i := len(s.changes) - 1; i >= 0; i-- {
		c := s.changes[i]
		at := runeOffset(text, c.at)
		text = text[:at] + c.removed + text[at+len(c.inserted):]
	}
	return text
}

// setText gives the editor a text the app set. Undoing the last step takes
// it back with the step, as when the app reformats what the user typed:
// the step becomes one change, from the text before it, so that it keeps
// two texts however often the app sets one, as a log does.
func (ed *editor) setText(s string) {
	ed.compositionActive = false
	if n := len(ed.undo); n > 0 {
		step := &ed.undo[n-1]
		before := step.before(ed.buf.s)
		clear(step.changes)
		step.changes = append(step.changes[:0], change{removed: before, inserted: s, app: true})
	}
	clear(ed.redo)
	ed.redo = ed.redo[:0]
	ed.buf.set(s)
	ed.visualSelection, ed.selected = false, nil
	ed.caret = min(ed.caret, ed.buf.n)
	ed.anchor = min(ed.anchor, ed.buf.n)
	if n := len(ed.undo); n > 0 {
		ed.undo[n-1].caretAfter, ed.undo[n-1].anchorAfter = ed.caret, ed.anchor
		ed.undo[n-1].selectionAfter = ed.editorSelection.clone()
	}
}

// record starts a step of undo before an edit; typing in a row makes one
// step.
func (ed *editor) record(typing bool) {
	now := ed.now()
	if typing && ed.coalesce && now.Sub(ed.lastEdit) < time.Second && len(ed.undo) > 0 {
		ed.lastEdit = now
		return
	}
	ed.undo = append(ed.undo, undoStep{caret: ed.caret, anchor: ed.anchor, selectionBefore: ed.editorSelection.clone()})
	if len(ed.undo) > 200 {
		ed.undo[0] = undoStep{}
		ed.undo = ed.undo[1:]
	}
	clear(ed.redo)
	ed.redo = ed.redo[:0]
	ed.coalesce = typing
	ed.lastEdit = now
}

// replace replaces the runes from start to end with s, as the last step
// of undo notes, and puts the caret after it.
func (ed *editor) replace(start, end int, s string) {
	if !ed.multiline {
		s = strings.Map(func(r rune) rune {
			if r == '\n' || r == '\r' {
				return ' '
			}
			return r
		}, s)
	}
	if n := len(ed.undo); n > 0 {
		step := &ed.undo[n-1]
		removed := ed.buf.slice(start, end)
		// A short deletion must not keep the entire old document alive.
		// Deleting it whole already needs all of its bytes for undo.
		if !ed.buf.indexed && len(removed) < ed.buf.byteLen() {
			removed = strings.Clone(removed)
		}
		c := change{at: start, removed: removed, inserted: s}
		// Typing grows the text the step inserted.
		grown := false
		if k := len(step.changes); k > 0 && c.removed == "" {
			if last := &step.changes[k-1]; !last.app && last.at+utf8.RuneCountInString(last.inserted) == start {
				last.inserted += s
				grown = true
			}
		}
		if !grown {
			step.changes = append(step.changes, c)
		}
	}
	ed.edit(start, end, s)
	ed.caret = start + utf8.RuneCountInString(s)
	ed.anchor = ed.caret
	ed.caretAffinity, ed.anchorAffinity = text.Downstream, text.Downstream
	ed.visualSelection, ed.selected = false, nil
	if n := len(ed.undo); n > 0 {
		ed.undo[n-1].caretAfter, ed.undo[n-1].anchorAfter = ed.caret, ed.anchor
		ed.undo[n-1].selectionAfter = ed.editorSelection.clone()
	}
}

// takeBack undoes the last step of undo, or redoes the last undone with
// redo.
func (ed *editor) takeBack(redo bool) {
	from, to := &ed.undo, &ed.redo
	if redo {
		from, to = to, from
	}
	n := len(*from)
	if n == 0 {
		return
	}
	step := (*from)[n-1]
	(*from)[n-1] = undoStep{}
	*from = (*from)[:n-1]
	if redo {
		for _, c := range step.changes {
			ed.edit(c.at, c.at+utf8.RuneCountInString(c.removed), c.inserted)
		}
		ed.caret, ed.anchor = step.caretAfter, step.anchorAfter
		ed.editorSelection = step.selectionAfter.clone()
	} else {
		for i := len(step.changes) - 1; i >= 0; i-- {
			c := step.changes[i]
			ed.edit(c.at, c.at+utf8.RuneCountInString(c.inserted), c.removed)
		}
		ed.caret, ed.anchor = step.caret, step.anchor
		ed.editorSelection = step.selectionBefore.clone()
	}
	*to = append(*to, step)
	ed.coalesce = false
	if ed.area != nil {
		ed.area.reveal = true
	}
}
