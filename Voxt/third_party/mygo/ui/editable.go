package ui

import (
	"strings"
	"time"
)

// editState is the state of an EditableText: whether it is being edited,
// the text typed, whether the input was just shown, and when a click on
// the text of a chosen row starts editing it.
type editState struct {
	editing, fresh bool
	draft          string
	after          time.Time
}

// renameDelay is how long after a click on the text of a chosen row
// editing starts, unless a second click makes it a double click.
const renameDelay = 500 * time.Millisecond

// EditableText shows *value as a text that the user edits in place, as a
// file's name in Finder: a double click on it, or Enter while it has the
// focus, shows a text input of it, with the text before its extension
// selected. Enter keeps what was typed, as moving the focus away does, and
// Escape goes back; Changed reports a new value.
//
// In a List's or a Table's rows, the list keeps the focus and the text
// takes none: Enter on macOS, and F2, edit the text of the row chosen,
// while a double click on the row is Submitted; a click on the text of a
// row already chosen edits it, once a second click would no longer make a
// double click.
//
//	ui.Table(c, &app.files, cols, len(app.names), func(row, col int) {
//		if ui.EditableText(c, &app.names[row]).Changed() {
//			app.rename(row)
//		}
//	})
func coreEditableText(c *context, value *string) *node {
	t := c.theme
	box := coreRow(c).AlignItems(Center).MinWidth(0)
	box.widget = "EditableText"
	st := coreLocal(box, "edit", func() editState { return editState{} })
	row := c.row
	start := false
	if row != nil && row.f != nil {
		s := row.f.s
		s.editablesNow = true
		if s.editAsked && s.editKey == row.key {
			s.editAsked, start = false, true
		}
		box.flags |= flagHover
		switch {
		case row.double || !row.f.chosen(row.i, row.key):
			st.after = time.Time{}
		case row.clicked && row.chosen && box.Hovered():
			st.after = c.now.Add(renameDelay)
			c.After(renameDelay)
		case !st.after.IsZero() && !c.now.Before(st.after):
			st.after, start = time.Time{}, true
		}
	} else {
		box.Focusable()
		if !st.editing && (box.DoubleClicked() || box.Shortcut(0, KeyEnter)) {
			start = true
		}
	}
	if start && !st.editing {
		st.editing, st.fresh, st.draft = true, true, *value
		c.rt.consumed = true
	}
	if !st.editing {
		box.Children(func() { coreText(c, *value).SingleLine() })
		return box
	}
	// done ends editing, keeping what was typed or not, and gives the focus
	// back to the list or to the text.
	done := func(keep bool) {
		st.editing = false
		if keep && st.draft != *value {
			*value = st.draft
			box.st.markChanged()
		}
		if row != nil && row.f != nil {
			row.f.owner.Focus()
		} else {
			box.Focus()
		}
		c.rt.focusVisible = true
		c.rt.consumed = true
	}
	var in *node
	box.Children(func() {
		// Where the text was, within its padding and border.
		pad := t.Space(0.5)
		in = coreTextInputBase(c, &st.draft).Grow(1).MinWidth(t.Space(10)).Padding(pad, pad*2).
			Margin(-pad-1, -pad*2-1).Radius(t.Space(1)).Background(t.Background).Border(1, t.Accent).
			TextColor(t.Text)
	})
	box.afterInput(func() {
		focused := c.rt.focused == in.id
		switch {
		case st.fresh:
			st.fresh = false
			in.Focus()
			// The name before its extension, as Finder selects.
			ed := in.st.editor
			end := ed.buf.n
			if dot := strings.LastIndexByte(st.draft, '.'); dot > 0 {
				end = len([]rune(st.draft[:dot]))
			}
			ed.anchor, ed.caret = 0, end
		case in.Submitted():
			done(true)
		case in.Shortcut(0, KeyEscape):
			done(false)
		case !focused:
			// The focus went elsewhere: what was typed stays, and the focus
			// with it.
			st.editing = false
			if st.draft != *value {
				*value = st.draft
				box.st.markChanged()
			}
			c.rt.consumed = true
		}
	})
	return box
}
