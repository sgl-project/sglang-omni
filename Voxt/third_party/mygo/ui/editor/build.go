package editor

import (
	"slices"
	"time"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/ui"
)

// View creates a code editor of st's buffer, as Zed's: syntax
// highlighting, several carets and selections, soft wraps, folding,
// finding and replacing, completions, diagnostics, and the commands and
// key bindings of code editors (Command, DefaultBindings). Give it a
// size, as Fill or Grow; with FitContent it is as high as its lines.
//
// It lays out and draws only the lines in view, shaping each row once
// and keeping it while frames draw it, whatever the size of the text: a
// buffer of a million lines scrolls as one of ten.
func View(c *ui.Context, st *State) ui.Element {
	if st.Buffer == nil {
		st.Buffer = NewBuffer("")
	}
	return st.ed.build(c, st.Buffer, st.options())
}

// build builds the element of an editor for a frame.
func (ed *editor) build(c *ui.Context, buf *Buffer, o options) ui.Element {
	ed.texts = l10n.In(c.Locale())
	ed.init(buf, o, c.Theme())
	ed.services = c.Services()
	buf.watch(ed.services)
	ed.sync()
	ed.syncGitDiff()
	ed.frameNow = c.Now()
	// Code reads left to right in a right-to-left interface too, as in
	// editors: its gutter on the left, its lines from the left.
	e := ui.Box(c).Direction(ui.LeftToRight).Focusable().FocusRing(false).Clip()
	// Focused is false while the window has not the keyboard.
	focused := e.Focused()
	if focused != ed.focused {
		ed.focused = focused
		ed.blinkStart = c.Now()
		if !focused {
			ed.complete.close()
			ed.gesture.active = false
		}
	}
	if ed.refocus {
		ed.refocus = false
		e.Focus()
	}
	label := o.label
	switch {
	case label != "":
	case o.diffSide == diffNewSide:
		label = ed.texts.NewText()
	case o.diffSide == diffOldSide:
		label = ed.texts.OldText()
	default:
		label = ed.texts.CodeEditor()
	}
	e.Role(ui.RoleTextField).Label(label).Access(ed.accessibility)
	if !o.readOnly {
		e.HandleTextInput(ed).InputMethod(!o.noInputMethod)
		if o.noInputMethod && ed.markedActive {
			// Input methods turned off leave what they composed.
			ed.UnmarkText()
		}
	}
	e.HandleInput(ed.input)
	switch ed.region {
	case regionText:
		e.Cursor(ui.CursorText)
	default:
		e.Cursor(ui.CursorDefault)
	}
	e.Draw(ed.paint)
	e.ContextMenu(ed.menu)
	if o.fitContent {
		// As high as the lines, at the width of the last frame.
		w := ed.r.W
		if w <= 0 {
			w = 600
		}
		ed.layout(w, ed.viewH, max(1, ed.view.scale))
		ed.view.measureWraps(0, time.Second)
		e.Height(float32(ed.view.heights.total()))
		ed.scrollY = 0
	} else if ed.r.W > 0 {
		// Scroll to what moved before building what depends on it.
		ed.layout(ed.r.W, ed.r.H, max(1, ed.view.scale))
		if ed.o.sharedScroll == nil {
			ed.restoreScroll()
			ed.doReveal()
		}
	}
	// Hovering: ask what the text under the resting pointer is. The
	// editor hears no move once the pointer has left it, so its hover
	// tells: what the pointer passed over on its way out asks nothing, and
	// a hover it asked for goes, unless the pointer went into it.
	h := &ed.hover
	outside := !e.Hovered() && !ed.gesture.active && !ed.scrollbar.active
	if outside && ed.pointerIn {
		ed.pointerIn = false
		ed.hoverLine = -1
		h.pending, h.restOff = false, -1
	}
	if h.pending && ed.pointerIn && !ed.gesture.active {
		if wait := 600*time.Millisecond - c.Now().Sub(h.restAt); wait > 0 {
			c.After(wait)
		} else {
			h.pending = false
			ed.requestHover(h.restOff)
			h.byPointer = true
		}
	}
	e.Children(func() {
		// Popups attach to boxes at the caret, or at the text hovered.
		anchor := func(key string, at int) (ui.Element, bool) {
			cr, ok := ed.caretRect(at)
			if !ok {
				return ui.Element{}, false
			}
			return ui.Box(c).Key(key).Absolute().Left(ed.textX+cr.X-ed.scrollX).Top(float32(float64(cr.Y)-ed.scrollY)).Size(1, cr.H).PassThrough(), true
		}
		var caretAnchor, hoverAnchor ui.Element
		var hasCaret, hasHover bool
		if ed.complete.open() {
			caretAnchor, hasCaret = anchor("code caret", ed.complete.at)
		}
		if h.shown {
			hoverAnchor, hasHover = anchor("code hover anchor", h.start)
		}
		if ds, ok := ed.o.decor.(*diffSide); ok {
			ds.buildBlocks(c, ed)
		}
		if ed.find.open {
			ed.buildFindBar(c)
		}
		if ed.goTo.open {
			ed.buildGoToLine(c)
		}
		if hasCaret {
			ed.buildCompletions(c, caretAnchor)
		}
		if hasHover {
			ed.buildHover(c, hoverAnchor)
		}
	})
	// Once the hover has said whether the pointer or the keyboard is in
	// it: the hover goes a grace after the pointer left the text hovered,
	// or the editor, unless it went into the hover.
	if h.shown {
		switch {
		case h.over || h.focused:
			h.hideAt = time.Time{}
		case outside && h.byPointer && h.hideAt.IsZero():
			h.hideAt = c.Now().Add(hoverHideDelay)
		}
		if !h.hideAt.IsZero() {
			if wait := h.hideAt.Sub(c.Now()); wait > 0 {
				c.After(wait)
			} else {
				ed.closeHover()
				c.Invalidate()
			}
		}
	}
	ed.needsBuild = false
	return e
}

// accessibility describes the editor to assistive technology, while it
// reads the window: a text area with its text, and the newest selection
// in runes.
func (ed *editor) accessibility(n *platform.AccessNode) {
	n.States |= platform.AccessMultiline
	if ed.o.readOnly {
		n.States |= platform.AccessReadOnly
	}
	r := &ed.view.text
	if r.Len() > 4<<20 {
		// Assistive technology gets the lines in view of a very large text.
		n.Value = ed.slice(r.LineStart(ed.anchorLine), r.LineEnd(min(r.LineCount()-1, ed.anchorLine+200)))
		return
	}
	n.Value = r.String()
	p := ed.primary()
	n.SelStart = utf8.RuneCountInString(n.Value[:min(p.start(), len(n.Value))])
	n.SelEnd = n.SelStart + utf8.RuneCountInString(n.Value[min(p.start(), len(n.Value)):min(p.end(), len(n.Value))])
	if n.Placeholder == "" {
		n.Placeholder = ed.o.placeholder
	}
}

// menu is the editor's context menu: the editing commands, folding, and
// the app's items.
func (ed *editor) menu(m *ui.Menu) {
	tx := ed.texts
	has := slices.ContainsFunc(ed.sels, func(s sel) bool { return !s.empty() })
	ro := ed.o.readOnly
	if ed.o.onDefinition != nil {
		if m.Item(tx.GoToDefinition()).Shortcut(0, ui.KeyF12).Chosen() {
			ed.run(GoToDefinition)
		}
		m.Separator()
	}
	if m.Item(tx.Cut()).Shortcut(ui.Cmd, ui.KeyX).Disabled(ro).Chosen() {
		ed.run(Cut)
	}
	if m.Item(tx.Copy()).Shortcut(ui.Cmd, ui.KeyC).Chosen() {
		ed.run(Copy)
	}
	if m.Item(tx.Paste()).Shortcut(ui.Cmd, ui.KeyV).Disabled(ro).Chosen() {
		ed.run(Paste)
	}
	m.Separator()
	if m.Item(tx.SelectAll()).Shortcut(ui.Cmd, ui.KeyA).Chosen() {
		ed.run(SelectAll)
	}
	if !ro {
		if m.Item(tx.ToggleComment()).Shortcut(ui.Cmd, ui.KeySlash).Chosen() {
			ed.run(ToggleComment)
		}
	}
	m.Submenu(tx.Selection(), func(m *ui.Menu) {
		if m.Item(tx.SelectNextOccurrence()).Shortcut(ui.Cmd, ui.KeyD).Chosen() {
			ed.run(SelectNext)
		}
		if m.Item(tx.SelectAllOccurrences()).Shortcut(ui.Cmd|ui.Shift, ui.KeyL).Chosen() {
			ed.run(SelectAllMatches)
		}
		if m.Item(tx.ExpandSelection()).Chosen() {
			ed.run(SelectLarger)
		}
		if m.Item(tx.SplitIntoLines()).Disabled(!has).Chosen() {
			ed.run(SplitSelectionIntoLines)
		}
	})
	if !ro {
		m.Submenu(tx.Transform(), func(m *ui.Menu) {
			for _, it := range []struct {
				label string
				cmd   Command
			}{{tx.UpperCase(), Uppercase}, {tx.LowerCase(), Lowercase}, {tx.TitleCase(), TitleCase},
				{tx.SnakeCase(), SnakeCase}, {tx.KebabCase(), KebabCase}, {tx.CamelCase(), CamelCase}, {tx.PascalCase(), PascalCase},
				{"", NoCommand}, {tx.SortLines(), SortLines}, {tx.ReverseLines(), ReverseLines}, {tx.UniqueLines(), UniqueLines},
				{tx.JoinLines(), JoinLines}, {tx.TrimTrailingWhitespace(), TrimTrailingWhitespace}} {
				if it.cmd == NoCommand {
					m.Separator()
					continue
				}
				if m.Item(it.label).Chosen() {
					ed.run(it.cmd)
				}
			}
		})
	}
	if ed.o.folding {
		m.Submenu(tx.Folding(), func(m *ui.Menu) {
			if m.Item(tx.Fold()).Chosen() {
				ed.run(Fold)
			}
			if m.Item(tx.Unfold()).Chosen() {
				ed.run(Unfold)
			}
			if m.Item(tx.FoldAll()).Chosen() {
				ed.run(FoldAll)
			}
			if m.Item(tx.UnfoldAll()).Chosen() {
				ed.run(UnfoldAll)
			}
		})
	}
	if ed.o.contextMenu != nil {
		m.Separator()
		ed.o.contextMenu(m)
	}
}
