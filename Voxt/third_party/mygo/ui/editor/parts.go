package editor

import (
	"unicode/utf8"

	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/ui"
)

// The parts of an editor around its text: its find bar, its go to line
// bar, its completions and its hovers. The editor places each, as an
// element with the behavior the part needs, and a render function styles
// and fills it, given a handle to what the part shows and does, as ui's
// bases are styled: the State's (RenderFindBar and the others), else the
// editor's own look (DefaultFindBar and the others), which an app's can
// call to wrap.

// FindBar is an editor's find bar, as RenderFindBar builds it: the query and
// its options, which the matches follow as they change, and what the bar
// does.
type FindBar struct {
	c  *ui.Context
	ed *editor
}

// Query and Replacement are the texts of the bar's fields, to bind inputs
// to, with QueryInput and ReplacementInput.
func (f *FindBar) Query() *string       { return &f.ed.find.query }
func (f *FindBar) Replacement() *string { return &f.ed.find.replace }

// CaseSensitive, WholeWord, Regexp and InSelection are the options of the
// search, to bind toggles to: matching case, whole words only, the query
// as a regular expression (RE2, as package regexp's), and only within the
// selections there were as the option was turned on.
func (f *FindBar) CaseSensitive() *bool { return &f.ed.find.caseSens }
func (f *FindBar) WholeWord() *bool     { return &f.ed.find.wholeWord }
func (f *FindBar) Regexp() *bool        { return &f.ed.find.regex }
func (f *FindBar) InSelection() *bool   { return &f.ed.find.inSelection }

// Replacing reports whether the bar replaces too, as Cmd+Alt+F opens it,
// and SetReplacing shows or hides replacing, which CanReplace reports a
// read-only editor never does.
func (f *FindBar) Replacing() bool      { return f.ed.find.replacing && !f.ed.o.readOnly }
func (f *FindBar) SetReplacing(on bool) { f.ed.find.replacing = on }
func (f *FindBar) CanReplace() bool     { return !f.ed.o.readOnly }

// Matches returns how many matches there are, and the index of the one
// selected, -1 without matches.
func (f *FindBar) Matches() (count, current int) {
	fd := &f.ed.find
	fd.search(f.ed)
	if len(fd.matches) == 0 {
		return 0, -1
	}
	return len(fd.matches), fd.current
}

// Status says what was found, as "3 of 12", "No results", or "Invalid"
// for a regular expression that is not one (Err); nothing without a
// query.
func (f *FindBar) Status() string { return f.ed.find.status(l10n.In(f.c.Locale())) }

// Err is the error of a query that is not a valid regular expression.
func (f *FindBar) Err() error { return f.ed.find.err }

// Next and Previous select the next and previous match after the caret,
// round the ends; SelectAll selects every match, with a caret each, and
// closes the bar.
func (f *FindBar) Next()      { f.ed.find.step(f.ed, 1) }
func (f *FindBar) Previous()  { f.ed.find.step(f.ed, -1) }
func (f *FindBar) SelectAll() { f.ed.find.selectAll(f.ed) }

// Replace replaces the match selected and selects the next; ReplaceAll
// replaces every match, as one step of undo. A regular expression's
// replacement expands $1 and ${name}.
func (f *FindBar) Replace()    { f.ed.find.replaceOne(f.ed) }
func (f *FindBar) ReplaceAll() { f.ed.find.replaceAll(f.ed) }

// Close closes the bar, and gives the keyboard back to the text.
func (f *FindBar) Close() {
	f.ed.find.open = false
	f.ed.refocus = true
}

// QueryInput makes in the field of the query: it takes the focus with
// the query selected as the bar opens, and Enter goes to the next match,
// Shift+Enter to the previous, Alt+Enter selects them all, and Escape
// closes the bar. It returns in.
func (f *FindBar) QueryInput(in ui.Element) ui.Element {
	fd := &f.ed.find
	if fd.focusQuery {
		fd.focusQuery = false
		in.Focus().SetTextSelection(0, utf8.RuneCountInString(fd.query))
	}
	mods := f.c.Modifiers()
	switch {
	case in.Submitted() && mods&ui.Shift != 0:
		f.Previous()
	case in.Submitted() && mods&ui.Alt != 0:
		f.SelectAll()
	case in.Submitted():
		f.Next()
	case in.Shortcut(0, ui.KeyEscape):
		f.Close()
	}
	return in
}

// ReplacementInput makes in the field of the replacement: Enter replaces
// the match selected, Cmd+Enter all of them, and Escape closes the bar.
// It returns in.
func (f *FindBar) ReplacementInput(in ui.Element) ui.Element {
	switch {
	case in.Submitted() && f.c.Modifiers()&ui.Cmd != 0:
		f.ReplaceAll()
	case in.Submitted():
		f.Replace()
	case in.Shortcut(0, ui.KeyEscape):
		f.Close()
	}
	return in
}

// DefaultFindBar builds the editor's own find bar into bar, as Zed's:
// the query with its options, the count of matches and the buttons going
// through them, and the replacement below it while replacing.
func DefaultFindBar(c *ui.Context, bar ui.Element, f *FindBar) {
	t := c.Theme()
	tx := l10n.In(c.Locale())
	bar.Gap(t.Space(1)).Padding(t.Space(1.5)).Background(t.Surface).Border(1, t.Border).Radius(t.Radius).
		Shadow(0, 2, 8, 0, ui.RGBA(0, 0, 0, 0.18))
	field := func(value *string, placeholder string) ui.Element {
		return ui.TextInputBase(c, value).Width(t.Space(50)).Placeholder(placeholder).Padding(t.Space(1), t.Space(2)).
			Background(t.Background).Border(1, t.Border).Radius(t.Radius - 2).FontSize(t.FontSize - 1).Label(placeholder)
	}
	button := func(label, tip string) ui.Element {
		b := ui.ButtonBase(c).Padding(t.Space(0.5), t.Space(1.5)).Radius(t.Radius - 2).Label(tip).Tooltip(tip).KeepFocus()
		b.Children(func() { ui.Text(c, label).FontSize(t.FontSize - 1) })
		return b
	}
	bar.Children(func() {
		ui.Row(c).Gap(t.Space(1)).AlignItems(ui.Center).Children(func() {
			f.QueryInput(field(f.Query(), tx.Find()))
			toggle := func(on *bool, label, tip string) {
				b := ui.ToggleBase(c, on).Padding(t.Space(0.5), t.Space(1.5)).Radius(t.Radius - 2).Label(tip).Tooltip(tip).KeepFocus()
				if *on {
					b.Background(t.Accent.Alpha(0.2)).TextColor(t.Accent)
				}
				b.Children(func() { ui.Text(c, label).FontSize(t.FontSize - 1) })
			}
			toggle(f.CaseSensitive(), "Aa", tx.MatchCase())
			toggle(f.WholeWord(), "ab", tx.MatchWholeWord())
			toggle(f.Regexp(), ".*", tx.UseRegularExpression())
			toggle(f.InSelection(), "[ ]", tx.FindInSelection())
			ui.Text(c, f.Status()).FontSize(t.FontSize - 2).TextColor(t.TextMuted).FontFeatures("tnum").MinWidth(t.Space(16)).Role(ui.RoleStatus)
			if button("↑", tx.PreviousMatch()).Clicked() {
				f.Previous()
			}
			if button("↓", tx.NextMatch()).Clicked() {
				f.Next()
			}
			if f.CanReplace() && button("⇄", tx.Replace()).Clicked() {
				f.SetReplacing(!f.Replacing())
			}
			if button("✕", tx.Close()).Clicked() {
				f.Close()
			}
		})
		if f.Replacing() {
			ui.Row(c).Gap(t.Space(1)).AlignItems(ui.Center).Children(func() {
				f.ReplacementInput(field(f.Replacement(), tx.Replace()))
				if ui.Button(c, tx.Replace()).Padding(t.Space(0.5), t.Space(2)).FontSize(t.FontSize - 1).Clicked() {
					f.Replace()
				}
				if ui.Button(c, tx.ReplaceAll()).Padding(t.Space(0.5), t.Space(2)).FontSize(t.FontSize - 1).Clicked() {
					f.ReplaceAll()
				}
			})
		}
	})
}

// buildFindBar places the find bar over the editor's top right corner,
// and builds it.
func (ed *editor) buildFindBar(c *ui.Context) {
	f := &ed.find
	f.update(ed)
	t := c.Theme()
	bar := ui.Box(c).Key("code find").Absolute().Top(t.Space(1)).Right(t.Space(4)).
		Role(ui.RoleToolbar).Label(l10n.In(c.Locale()).Find()).Cursor(ui.CursorDefault).HandleInput(shield)
	render := ed.o.renderFindBar
	if render == nil {
		render = DefaultFindBar
	}
	render(c, bar, &FindBar{c: c, ed: ed})
	// What the bar typed or toggled shows at once.
	f.update(ed)
	if isMac() {
		switch {
		case c.Shortcut(ui.Cmd, ui.KeyG):
			f.step(ed, 1)
		case c.Shortcut(ui.Shift|ui.Cmd, ui.KeyG):
			f.step(ed, -1)
		}
	} else {
		switch {
		case c.Shortcut(0, ui.KeyF3):
			f.step(ed, 1)
		case c.Shortcut(ui.Shift, ui.KeyF3):
			f.step(ed, -1)
		}
	}
}

// GoToLineBar is an editor's go to line bar, as RenderGoToLine builds it:
// line[:column], or +n and -n lines from the caret, which the editor
// scrolls to as it is typed, as Zed's.
type GoToLineBar struct {
	c  *ui.Context
	ed *editor
}

// Text is the text of the bar's field, to bind an input to, with Input.
func (g *GoToLineBar) Text() *string { return &g.ed.goTo.text }

// Input makes in the bar's field: it takes the focus as the bar opens,
// Enter goes to the line, and Escape closes the bar where the editor
// was. It returns in.
func (g *GoToLineBar) Input(in ui.Element) ui.Element {
	gt := &g.ed.goTo
	if gt.focus {
		gt.focus = false
		in.Focus()
	}
	switch {
	case in.Submitted():
		g.Go()
	case in.Shortcut(0, ui.KeyEscape):
		g.Close()
	}
	return in
}

// Cursor is where the newest caret is, and LineCount how many lines there
// are, as the bar tells.
func (g *GoToLineBar) Cursor() Pos    { return posOf(g.ed.view.text, g.ed.primary().head) }
func (g *GoToLineBar) LineCount() int { return g.ed.view.text.LineCount() }

// Target is the line and column typed, from 0, and whether the text is
// one.
func (g *GoToLineBar) Target() (Pos, bool) {
	line, col, ok := g.ed.goTo.target(g.ed)
	return Pos{line, col}, ok
}

// Go moves the caret to the line typed and closes the bar; Close closes
// it where the editor was.
func (g *GoToLineBar) Go() {
	ed := g.ed
	if line, col, ok := ed.goTo.target(ed); ok {
		off := offsetOf(ed.view.text, Pos{line, col})
		ed.goTo.open = false
		ed.setSels([]sel{ed.newSel(off, off)}, revealCenter)
		ed.refocus = true
	}
}
func (g *GoToLineBar) Close() { g.ed.goTo.close(g.ed) }

// DefaultGoToLine builds the editor's own go to line bar into bar: the
// field, and where the caret is.
func DefaultGoToLine(c *ui.Context, bar ui.Element, g *GoToLineBar) {
	t := c.Theme()
	tx := l10n.In(c.Locale())
	bar.Row().Gap(t.Space(2)).AlignItems(ui.Center).Padding(t.Space(1.5), t.Space(2)).Background(t.Surface).
		Border(1, t.Border).Radius(t.Radius).Shadow(0, 2, 8, 0, ui.RGBA(0, 0, 0, 0.18))
	bar.Children(func() {
		g.Input(ui.TextInputBase(c, g.Text()).Width(t.Space(60)).Padding(t.Space(1), t.Space(2)).Background(t.Background).
			Border(1, t.Border).Radius(t.Radius - 2).Label(tx.GoToLine()).Placeholder(tx.LineRange(g.LineCount())))
		cur := g.Cursor()
		ui.Text(c, tx.CurrentLine(cur.Line+1, g.LineCount(), cur.Col+1)).
			FontSize(t.FontSize - 2).TextColor(t.TextMuted)
	})
}

// buildGoToLine places the go to line bar in the middle of the editor's
// top, builds it, and shows the line typed.
func (ed *editor) buildGoToLine(c *ui.Context) {
	t := c.Theme()
	g := &ed.goTo
	ui.Row(c).Key("code go to line").Absolute().Top(t.Space(1)).Left(0).Right(0).Justify(ui.Center).PassThrough().Children(func() {
		bar := ui.Box(c).Role(ui.RoleDialog).Label(l10n.In(c.Locale()).GoToLine()).Cursor(ui.CursorDefault).HandleInput(shield)
		render := ed.o.renderGoToLine
		if render == nil {
			render = DefaultGoToLine
		}
		render(c, bar, &GoToLineBar{c: c, ed: ed})
	})
	if !g.open {
		return
	}
	if line, col, ok := g.target(ed); ok && line != g.preview {
		g.preview = line
		if cr, ok := ed.caretRect(offsetOf(ed.view.text, Pos{line, col})); ok {
			ed.setScroll(float64(cr.Y) - float64(ed.viewH-ed.view.lineHeight)/2)
		}
	}
}

// CompletionMenu is an editor's list of completions, as
// RenderCompletions builds it: the completions matching the word typed,
// best first, and the one chosen, which the arrows move and Enter or Tab
// inserts.
type CompletionMenu struct {
	c  *ui.Context
	ed *editor
	// chosen and closing are what the list was asked while it was built,
	// done once it is, so that the rows after the one clicked still
	// build: the item chosen, -1 for none.
	chosen  int
	closing bool
}

// Items returns the completions shown, best first.
func (m *CompletionMenu) Items() []Completion {
	cc := &m.ed.complete
	out := make([]Completion, len(cc.filtered))
	for i, idx := range cc.filtered {
		out[i] = cc.items[idx]
	}
	return out
}

// Selected is the index in Items of the completion chosen.
func (m *CompletionMenu) Selected() int { return m.ed.complete.selected }

// Prefix is the word typed, which the completions match.
func (m *CompletionMenu) Prefix() string { return m.ed.complete.prefix }

// Item makes the row of completion i of Items: selected while chosen,
// scrolled into view as the arrows choose it, and inserting it when
// clicked. Style it and build its content as the row of a list.
func (m *CompletionMenu) Item(i int) ui.Element {
	cc := &m.ed.complete
	it := cc.items[cc.filtered[i]]
	row := ui.ButtonBase(m.c).Key(i).FocusRing(false).Role(ui.RoleListItem).Label(it.Label).Selected(i == cc.selected)
	if i == cc.selected && cc.scrollTo {
		row.ScrollIntoView()
	}
	if row.Clicked() {
		m.Choose(i)
	}
	return row
}

// Choose inserts completion i of Items in place of the word typed, and
// closes the list; Close closes it. Both happen once the list is built.
func (m *CompletionMenu) Choose(i int) {
	if i >= 0 && i < len(m.ed.complete.filtered) {
		m.chosen = i
	}
}
func (m *CompletionMenu) Close() { m.closing = true }

// Font is the editor's font, LineHeight the height of its lines, and
// Theme its colors, to show code as the editor does.
func (m *CompletionMenu) Font() ui.Font       { return m.ed.view.font }
func (m *CompletionMenu) LineHeight() float32 { return m.ed.view.lineHeight }
func (m *CompletionMenu) Theme() *Theme       { return m.ed.theme }

// Icon returns a letter or symbol for a kind of completion, and Token the
// kind of token it is, whose color tints it, as the default list shows.
func (k CompletionKind) Icon() string {
	switch k {
	case CompletionKeyword:
		return "k"
	case CompletionFunction, CompletionMethod:
		return "ƒ"
	case CompletionVariable:
		return "x"
	case CompletionField:
		return "·"
	case CompletionType:
		return "T"
	case CompletionModule:
		return "{}"
	case CompletionConstant:
		return "π"
	case CompletionSnippet:
		return "⌘"
	}
	return "≡"
}

func (k CompletionKind) Token() TokenKind {
	switch k {
	case CompletionKeyword:
		return TokenKeyword
	case CompletionFunction, CompletionMethod:
		return TokenFunction
	case CompletionField:
		return TokenProperty
	case CompletionType, CompletionModule:
		return TokenType
	case CompletionConstant:
		return TokenConstant
	}
	return TokenText
}

// maxCompletionRows is how many completions the default list shows.
const maxCompletionRows = 200

// DefaultCompletions builds the editor's own list of completions into
// panel, as Zed's: each with its kind, label and detail, nine rows high,
// and the documentation of the one chosen below them.
func DefaultCompletions(c *ui.Context, panel ui.Element, m *CompletionMenu) {
	t := c.Theme()
	ct := m.Theme()
	font := m.Font()
	lh := m.LineHeight()
	panel.Background(t.Surface).Border(1, t.Border).Radius(t.Radius).Shadow(0, 4, 16, 0, ui.RGBA(0, 0, 0, 0.2)).
		Padding(t.Space(1)).MaxHeight(lh*9 + t.Space(2)).MinWidth(t.Space(60)).MaxWidth(t.Space(150))
	items := m.Items()
	panel.Children(func() {
		ui.Scroll(c).Key("rows").MaxHeight(lh * 9).Children(func() {
			for i, it := range items {
				if i >= maxCompletionRows {
					break
				}
				row := m.Item(i).Row().Justify(ui.Start).Gap(t.Space(2)).AlignItems(ui.Center).Padding(t.Space(0.5), t.Space(1.5)).
					Radius(t.Radius - 2).Height(lh + 2)
				if i == m.Selected() {
					row.Background(t.Accent.Alpha(0.18))
				}
				row.Children(func() {
					ui.Text(c, it.Kind.Icon()).Font("monospace").FontSize(font.Size - 1).TextColor(ct.TokenColor(it.Kind.Token())).Width(t.Space(3))
					ui.Text(c, it.Label).Font(font.Family).FontSize(font.Size).TextColor(ct.Text).SingleLine().Shrink(1)
					if it.Detail != "" {
						ui.Text(c, it.Detail).FontSize(font.Size - 1).TextColor(t.TextMuted).SingleLine().Shrink(1).Grow(1).TextAlign(ui.End)
					}
				})
			}
		})
		if s := m.Selected(); s < len(items) && items[s].Documentation != "" {
			ui.Box(c).Height(1).Background(t.Border).Margin(t.Space(1), 0)
			ui.Text(c, items[s].Documentation).FontSize(t.FontSize-1).TextColor(t.TextMuted).MaxWidth(t.Space(140)).Padding(t.Space(0.5), t.Space(1.5))
		}
	})
}

// buildCompletions places the list of completions below the word being
// completed, above the window's content, and builds it.
func (ed *editor) buildCompletions(c *ui.Context, anchor ui.Element) {
	cc := &ed.complete
	ui.Overlay(c, func() {
		panel := ui.Box(c).Key("code completions").AttachTo(anchor, ui.AnchorBottomLeft, ui.AnchorTopLeft).
			Role(ui.RoleList).Label(l10n.In(c.Locale()).Completions()).Cursor(ui.CursorDefault).KeepFocus().HandleInput(shield)
		render := ed.o.renderCompletions
		if render == nil {
			render = DefaultCompletions
		}
		m := &CompletionMenu{c: c, ed: ed, chosen: -1}
		render(c, panel, m)
		switch {
		case m.chosen >= 0:
			cc.selected = m.chosen
			ed.acceptCompletion()
			c.Invalidate()
		case m.closing:
			cc.close()
			c.Invalidate()
		}
	})
	cc.scrollTo = false
}

// HoverCard is what an editor shows of the text under the pointer, as
// RenderHover builds it: the diagnostics there, and what the app gave
// ShowHover.
type HoverCard struct {
	c  *ui.Context
	ed *editor
}

// Diagnostics are the diagnostics of the text hovered.
func (h *HoverCard) Diagnostics() []Diagnostic { return h.ed.hover.diags }

// Text is the text ShowHover gave, and Content builds what its function
// builds, when it gave one.
func (h *HoverCard) Text() string { return h.ed.hover.text }
func (h *HoverCard) Content() {
	if b := h.ed.hover.build; b != nil {
		b(h.c)
	}
}

// Range is the text hovered, from start to end.
func (h *HoverCard) Range() (start, end int) { return h.ed.hover.start, h.ed.hover.end }

// Font is the editor's font, and Theme its colors.
func (h *HoverCard) Font() ui.Font { return h.ed.view.font }
func (h *HoverCard) Theme() *Theme { return h.ed.theme }

// DefaultHover builds the editor's own hover into panel, as Zed's: the
// diagnostics with bars of their severity's color, then the text in the
// editor's font and what the app builds, all of it selectable.
func DefaultHover(c *ui.Context, panel ui.Element, h *HoverCard) {
	t := c.Theme()
	ct := h.Theme()
	panel.Background(t.Surface).Border(1, t.Border).Radius(t.Radius).Shadow(0, 4, 16, 0, ui.RGBA(0, 0, 0, 0.2)).
		Padding(t.Space(2)).Gap(t.Space(1.5)).MaxWidth(t.Space(150)).Margin(0, 0, t.Space(1), 0).Selectable()
	panel.Children(func() {
		for _, d := range h.Diagnostics() {
			ui.Row(c).Gap(t.Space(1.5)).Children(func() {
				ui.Box(c).Width(3).Radius(1.5).Background(ct.SeverityColor(d.Severity)).AlignSelf(ui.Stretch)
				msg := d.Message
				if d.Source != "" {
					msg += " (" + d.Source + ")"
				}
				ui.Text(c, msg).FontSize(t.FontSize - 1).Shrink(1)
			})
		}
		if text := h.Text(); text != "" {
			ui.Text(c, text).Font(h.Font().Family).FontSize(h.Font().Size).TextColor(ct.Text)
		}
		h.Content()
	})
}

// buildHover places the hover above the text hovered, above the window's
// content, and builds it.
func (ed *editor) buildHover(c *ui.Context, anchor ui.Element) {
	h := &ed.hover
	if !h.shown || h.build == nil && h.text == "" && len(h.diags) == 0 {
		return
	}
	ui.Overlay(c, func() {
		// The hover takes the keyboard as it is pressed, for its text to
		// be selected and copied; Escape gives it back to the editor.
		panel := ui.Box(c).Key("code hover").AttachTo(anchor, ui.AnchorTopLeft, ui.AnchorBottomLeft).
			Role(ui.RoleTooltip).HandleInput(shield)
		h.over, h.focused = panel.Hovered(), panel.FocusWithin()
		if panel.Shortcut(0, ui.KeyEscape) {
			ed.closeHover()
		}
		render := ed.o.renderHover
		if render == nil {
			render = DefaultHover
		}
		render(c, panel, &HoverCard{c: c, ed: ed})
	})
}

// shield takes no input, but keeps the presses on what an editor shows
// over its text, as its find bar, from the editor around it: the input
// goes to the innermost element taking it as it comes, which then leaves
// it to the buttons and fields inside, as they take it.
func shield(ui.InputEvent) bool { return false }
