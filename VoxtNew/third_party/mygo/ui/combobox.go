package ui

import (
	"slices"
	"strings"
)

// ComboboxParts are the parts of a combobox without a look: ComboboxBase
// makes its text input, Popup its popup, and Item the options in it.
type comboboxParts struct {
	// Input is the text input, whose text filters the options: style it,
	// and name it with Label.
	Input *node
	c     *context
	text  *string
	open  *bool
	// anchor is what the popup shows below: the input, or the field
	// around it.
	anchor *node
	// highlight is the index of the option the pointer or the arrows are
	// on, -1 for none; values are the options of the last frame that
	// showed them, next those of this frame, and items their elements.
	// pointer is where the pointer was when it last highlighted an option.
	// typed is set while the text is what the user typed since the popup
	// opened, which Filtering reports. shown is the option last scrolled
	// into view as the highlight moved to it.
	highlight *int
	values    *[]string
	next      []string
	items     []*node
	pointer   *[2]float32
	typed     *bool
	shown     *string
	chosen    string
	isChosen  bool
}

// ComboboxBase creates a combobox without a look: a text input editing
// *text, whose popup of options, built with Popup and Item, opens below it
// as the user types, clicks it or presses Down. Up and Down move among the
// options, Enter or a click chooses one, which Chosen reports, and
// Escape, or the input losing the focus, closes the popup. Assistive
// technology sees a combo box, and follows the option the arrows are on.
//
//	cb := ui.ComboboxBase(c, &app.query)
//	cb.Input.Padding(6, 10).Border(1, gray).Label("Font")
//	cb.Popup(func(panel ui.Element) {
//		panel.Padding(4).Background(white).Border(1, gray)
//		for _, f := range fonts {
//			item := cb.Item(f).Padding(6, 10)
//			if item.Highlighted() {
//				item.Background(blue).TextColor(white)
//			}
//			item.Children(func() { ui.Text(c, f) })
//		}
//	})
//	if f, ok := cb.Chosen(); ok {
//		app.font = f
//	}
//
// Combobox, Autocomplete and TokenField are built on it.
func coreComboboxBase(c *context, text *string) *comboboxParts {
	return comboboxBase(c, text, true)
}

// comboboxBase is ComboboxBase, which with first highlights the first
// option as the user types, for Enter to choose.
func comboboxBase(c *context, text *string, first bool) *comboboxParts {
	in := coreTextInputBase(c, text)
	in.widget, in.role = "Combobox", RoleComboBox
	in.flags |= flagClickable
	p := &comboboxParts{
		Input: in, anchor: in, c: c, text: text,
		open:      coreLocal(in, "open", func() bool { return false }),
		highlight: coreLocal(in, "highlight", func() int { return -1 }),
		values:    coreLocal(in, "values", func() []string { return nil }),
		pointer:   coreLocal(in, "pointer", func() [2]float32 { return [2]float32{} }),
		typed:     coreLocal(in, "typed", func() bool { return false }),
		shown:     coreLocal(in, "shown", func() string { return "" }),
	}
	in.afterInput(func() {
		p.chosen, p.isChosen = "", false
		rt := c.rt
		typedFirst := -1
		if first {
			typedFirst = 0
		}
		open := func(o, typed bool, highlight int) {
			if *p.open != o || *p.typed != typed || *p.highlight != highlight {
				rt.consumed = true
			}
			*p.open, *p.typed, *p.highlight = o, typed, highlight
			*p.pointer = [2]float32{rt.pointerX, rt.pointerY}
		}
		n := len(*p.values)
		switch {
		case !in.Focused():
			if *p.open {
				open(false, false, -1)
			}
		case in.Changed():
			open(true, true, typedFirst)
		case in.Clicked() && !*p.open:
			open(true, false, -1)
		}
		switch {
		case in.Shortcut(0, KeyDown):
			if !*p.open {
				open(true, *p.typed, 0)
			} else if n > 0 {
				*p.highlight = min(*p.highlight+1, n-1)
				rt.consumed = true
			}
		case in.Shortcut(0, KeyUp):
			if !*p.open {
				open(true, *p.typed, -2) // the last, once known
			} else if n > 0 {
				*p.highlight = max(*p.highlight-1, 0)
				rt.consumed = true
			}
		}
		// Escape closes the popup; while it is closed, Escape is the
		// dialog's.
		if *p.open && in.Shortcut(0, KeyEscape) {
			open(false, false, -1)
		}
		if in.Submitted() && *p.open {
			if h := *p.highlight; h >= 0 && h < n {
				p.choose((*p.values)[h])
			}
		}
	})
	in.expanded = *p.open
	return p
}

func (p *comboboxParts) choose(v string) {
	p.chosen, p.isChosen = v, true
	p.Input.st.submitted = false
	*p.open, *p.typed, *p.highlight = false, false, -1
	p.c.rt.consumed = true
}

// Chosen returns the option the user chose in this frame, by a click or
// Enter: ask it after Popup, which takes the input of the options it
// built as it returns, so the choice can be handled in the same build pass.
func (p *comboboxParts) Chosen() (string, bool) {
	p.c.rt.applyInputs()
	return p.chosen, p.isChosen
}

// Open reports whether the popup shows.
func (p *comboboxParts) Open() bool { return *p.open }

// SetOpen opens or closes the popup.
func (p *comboboxParts) SetOpen(o bool) {
	if *p.open != o {
		*p.open, *p.typed, *p.highlight = o, false, -1
		p.c.rt.consumed = true
	}
}

// Filtering reports whether the text is what the user typed since the
// popup opened, which the options should match; it is not when the popup
// opened by a click or Down, which shows them all.
func (p *comboboxParts) Filtering() bool { return *p.typed }

// Popup shows the popup below the input, at least as wide, while it is
// open: fn styles the panel and builds the options with Item. It returns
// the panel, or nil while closed or without options.
func (p *comboboxParts) Popup(fn func(panel *node)) *node {
	c := p.c
	p.next, p.items = p.next[:0], p.items[:0]
	if !*p.open {
		*p.values, *p.shown = (*p.values)[:0], ""
		return nil
	}
	b := p.anchor.Bounds()
	var panel *node
	coreOverlay(c, func() {
		panel = coreBox(c).MinWidth(b.W).Role(RoleList).AttachTo(p.anchor, AnchorBottomStart, AnchorTopStart)
		// Pressing an option keeps the focus in the input.
		panel.flags |= flagClickable | flagKeepFocus
		panel.choosesItems = true
		panel.Children(func() { fn(panel) })
	})
	*p.values = append((*p.values)[:0], p.next...)
	n := len(p.next)
	if *p.highlight == -2 || *p.highlight >= n {
		*p.highlight = n - 1
		c.rt.consumed = true
	}
	for i, item := range p.items {
		item.setPos, item.setSize = i+1, n
	}
	if h := *p.highlight; h >= 0 && h < n {
		p.Input.activeDescendant = p.items[h]
	}
	if n == 0 {
		panel.flags |= flagInvisible // nothing to show
		return nil
	}
	return panel
}

// Item creates an option: a row that is Highlighted when the pointer or
// the arrows are on it, and that chooses value when clicked. The option
// the highlight moves to scrolls into view, once, so that the wheel
// scrolls on past it.
func (p *comboboxParts) Item(value string) *node {
	c := p.c
	i := len(p.next)
	p.next = append(p.next, value)
	item := coreRow(c).Role(RoleListItem)
	item.flags |= flagClickable | flagHover | flagChoosable
	p.items = append(p.items, item)
	if pt := [2]float32{c.rt.pointerX, c.rt.pointerY}; item.Hovered() && pt != *p.pointer {
		*p.pointer = pt
		if *p.highlight != i {
			*p.highlight = i
			c.rt.consumed = true
		}
	}
	item.highlighted = *p.highlight == i
	// Into view as the highlight moves to it, and not again: in every
	// frame, it would pull the popup back as the wheel scrolls away from
	// it, the pointer, which highlights, staying still.
	if item.highlighted && *p.shown != value {
		*p.shown = value
		item.ScrollIntoView()
	}
	// The option the arrows are on is the one assistive technology reads
	// as chosen, as in a list box.
	item.checked = 1 + int8(b2f(item.highlighted))
	item.afterInput(func() {
		if item.Clicked() {
			p.choose(value)
		}
	})
	return item
}

// matching returns the options containing query, ignoring case, those
// starting with it first; all of them for a blank query.
func matching(options []string, query string) []string {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return options
	}
	var starts, contains []string
	for _, o := range options {
		switch l := strings.ToLower(o); {
		case strings.HasPrefix(l, q):
			starts = append(starts, o)
		case strings.Contains(l, q):
			contains = append(contains, o)
		}
	}
	return append(starts, contains...)
}

// styleOptions builds the popup of a combobox, showing options, with the
// theme's look.
func styleOptions(c *context, p *comboboxParts, options []string) *node {
	t := c.theme
	return p.Popup(func(panel *node) {
		stylePanel(c, panel)
		panel.MaxHeight(t.Space(70))
		panel.flags |= flagScrollY | flagHover
		for _, opt := range options {
			item := p.Item(opt).Padding(t.Space(1.5), t.Space(2.5)).Radius(t.Radius)
			if item.Highlighted() {
				item.Background(t.Accent).TextColor(t.AccentText)
			}
			item.Children(func() { coreText(c, opt).SingleLine() })
		}
	})
}

// field creates the box around a field's input, with the look of a text
// input, which builds the input with fn; a click in it focuses the input.
// Its Label names the input.
func field(c *context, fn func() *node) (*node, *node) {
	t := c.theme
	f := coreRow(c).AlignItems(Center).Gap(t.Space(1.5)).Padding(t.Space(1.5), t.Space(2.5)).
		Radius(t.Radius).Background(t.Surface).Border(1, t.Border).Role(RoleNone).Cursor(CursorText)
	f.flags |= flagClickable | flagHover
	var in *node
	f.Children(func() { in = fn() })
	in.nameFrom = f
	f.styleFn = func(f *node) { inputBorder(t, f, in) }
	if f.Clicked() {
		in.Focus()
	}
	return f, in
}

// chevronDown draws the arrow of a field that opens options.
func chevronDown(c *context) *node {
	t := c.theme
	return coreBox(c).Size(t.Space(2.5), t.Space(2.5)).Shrink(0).Draw(func(p *Painter, r Rect) {
		var path Path
		path.MoveTo(r.X+r.W*0.1, r.Y+r.H*0.3).LineTo(r.X+r.W*0.5, r.Y+r.H*0.7).LineTo(r.X+r.W*0.9, r.Y+r.H*0.3)
		p.StrokePath(&path, 1.5, t.TextMuted)
	})
}

// Combobox creates a combobox choosing *selected among options: a field
// showing the choice, whose text filters the options below it as the user
// types, those starting with it first, while a click or Down shows them
// all; a click showing them selects the choice, for typing to replace.
// Up and Down move among them and Enter chooses, as does a click; Escape
// closes them, and the text goes back to the choice when the field loses
// the focus. Changed reports a new choice. Name it with Label.
//
//	ui.Combobox(c, &app.font, fonts).Label("Font")
func coreCombobox(c *context, selected *string, options []string) *node {
	t := c.theme
	var p *comboboxParts
	var text *string
	f, in := field(c, func() *node {
		text = coreLocal(c.parent, "combobox", func() string { return *selected })
		p = comboboxBase(c, text, true)
		p.Input.Grow(1).MinWidth(t.Space(25))
		arrow := chevronDown(c)
		arrow.flags |= flagClickable | flagKeepFocus
		arrow.afterInput(func() {
			if arrow.Clicked() {
				p.SetOpen(!p.Open())
				p.Input.Focus()
				if p.Open() {
					// As a click in the text opening them does.
					p.Input.st.editor.selectAll()
				}
			}
		})
		return p.Input
	})
	p.anchor = f
	// Whether the popup showed before this frame's input.
	wasOpen := p.Open()
	shown := options
	if p.Filtering() {
		shown = matching(options, *text)
	}
	styleOptions(c, p, shown)
	f.afterInput(func() {
		// A click opening the options selects the choice, as choosing
		// one leaves it, so that the next keystroke replaces it and
		// filters, instead of appending to it and matching nothing.
		// Clicks while they show place the caret.
		if in.Clicked() && !wasOpen && p.Open() && !p.Filtering() {
			in.st.editor.selectAll()
		}
		if v, ok := p.Chosen(); ok {
			if *selected != v {
				*selected = v
				f.st.markChanged()
			}
			*text = v
			in.st.editor.setText(v)
			in.st.editor.selectAll()
		} else if !in.Focused() && !p.Open() && *text != *selected {
			// What was typed matched no choice.
			*text = *selected
			in.st.editor.setText(*selected)
		}
	})
	return f
}

// Autocomplete creates a text input editing *value, which suggests below
// it those of suggestions containing what was typed, those starting with
// it first: Up and Down move among them and Enter takes one, as does a
// click, while Enter on none is Submitted. Changed reports a change,
// typed or taken. Name it with Label.
//
//	ui.Autocomplete(c, &app.city, cities).Label("City")
func coreAutocomplete(c *context, value *string, suggestions []string) *node {
	p := comboboxBase(c, value, false)
	in := p.Input
	t := c.theme
	in.Padding(t.Space(1.5), t.Space(2.5)).Radius(t.Radius).Background(t.Surface).Border(1, t.Border)
	in.styleFn = func(e *node) { inputBorder(t, e, e) }
	var shown []string
	if q := strings.TrimSpace(*value); q != "" && p.Filtering() {
		for _, s := range matching(suggestions, q) {
			if !strings.EqualFold(s, *value) {
				shown = append(shown, s)
			}
		}
	}
	styleOptions(c, p, shown)
	in.afterInput(func() {
		if v, ok := p.Chosen(); ok {
			if *value != v {
				*value = v
				in.st.markChanged()
			}
			in.st.editor.setText(v)
			in.st.editor.caret, in.st.editor.anchor = in.st.editor.buf.n, in.st.editor.buf.n
		}
	})
	return in
}

// SearchField creates a field for searching, editing *query: a magnifying
// glass, the text, and a button clearing it while it holds any, as Escape
// does then. Enter is its Submitted, and Changed reports a change. It
// shows Search while empty, and assistive technology sees a search field;
// name it with Label.
//
//	if ui.SearchField(c, &app.query).Label("Search mail").Changed() {
//		app.filter()
//	}
func coreSearchField(c *context, query *string) *node {
	t := c.theme
	f, in := field(c, func() *node {
		magnifier(c)
		in := coreTextInputBase(c, query).Grow(1).MinWidth(t.Space(15)).Placeholder(c.texts().Search())
		in.search = true
		if *query != "" {
			clear := coreButtonBase(c).Size(t.Space(4), t.Space(4)).Radius(t.Space(2)).Background(t.TextMuted.Alpha(0.5)).Label(c.texts().Clear())
			clear.flags &^= flagFocusable // as AppKit's: no stop of Tab
			clear.flags |= flagKeepFocus
			clear.role = RoleButton
			clear.Draw(func(p *Painter, r Rect) {
				k := r.W * 0.3
				var x Path
				x.MoveTo(r.X+k, r.Y+k).LineTo(r.X+r.W-k, r.Y+r.H-k).MoveTo(r.X+r.W-k, r.Y+k).LineTo(r.X+k, r.Y+r.H-k)
				p.StrokePath(&x, 1.5, t.Background)
			})
			clear.afterInput(func() {
				if clear.Clicked() || in.Shortcut(0, KeyEscape) {
					*query = ""
					in.st.editor.setText("")
					in.st.markChanged()
					in.Focus()
					c.rt.consumed = true
				}
			})
		}
		return in
	})
	f.Padding(t.Space(1.5), t.Space(2))
	f.afterInput(func() {
		if in.Changed() {
			f.st.markChanged()
		}
		if in.Submitted() {
			f.st.markSubmitted()
		}
	})
	return f
}

// searchIcon is a magnifying glass, MingCute's search icon (Apache License
// 2.0, https://github.com/Richard9394/MingCute/blob/main/LICENSE).
var searchIcon = MustParseSVG([]byte(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M14.5 14.5L20 20m-4-10a6 6 0 1 1-12 0a6 6 0 0 1 12 0Z"/></svg>`))

// magnifier shows the magnifying glass of a field for searching, in the
// muted color of the theme's text, no higher than a line of the field's
// text, so that the field is as high as a text input.
func magnifier(c *context) *node {
	t := c.theme
	return coreIcon(c, searchIcon).Size(t.Space(3.75), t.Space(3.75)).Shrink(0).TextColor(t.TextMuted)
}

// TokenField creates a field of tokens, as of tags or the recipients of a
// mail: *tokens shows as chips, each with a button taking it out, before
// an input where Enter or a comma adds what was typed, and Backspace in
// it while empty takes out the last. The suggestions containing what was
// typed show below it, those not already tokens: Up and Down move among
// them, and Enter or a click adds one. Changed reports a change; name it
// with Label.
//
//	ui.TokenField(c, &app.tags, allTags).Label("Tags")
func coreTokenField(c *context, tokens *[]string, suggestions []string) *node {
	t := c.theme
	var p *comboboxParts
	var query *string
	changed := false
	add := func(s string) {
		s = strings.TrimSpace(s)
		if s != "" && !slices.Contains(*tokens, s) {
			*tokens = append(*tokens, s)
			changed = true
		}
	}
	f, in := field(c, func() *node {
		for i := 0; i < len(*tokens); i++ {
			tok := (*tokens)[i]
			chip := coreRow(c).Key(tok).AlignItems(Center).Gap(t.Space(1)).Padding(t.Space(0.25), 0).PaddingStart(t.Space(2)).PaddingEnd(t.Space(1)).
				Radius(t.Radius).Background(t.SurfacePressed).Shrink(0)
			chip.Children(func() {
				coreText(c, tok).SingleLine()
				x := coreButtonBase(c).Size(t.Space(3.5), t.Space(3.5)).Radius(t.Space(1)).Label(c.texts().RemoveToken(tok))
				x.flags &^= flagFocusable
				x.flags |= flagKeepFocus
				x.role = RoleButton
				x.styleFn = func(x *node) {
					if x.Hovered() {
						x.bg = t.SurfaceHover
					}
				}
				x.Draw(func(pt *Painter, r Rect) {
					k := r.W * 0.3
					var path Path
					path.MoveTo(r.X+k, r.Y+k).LineTo(r.X+r.W-k, r.Y+r.H-k).MoveTo(r.X+r.W-k, r.Y+k).LineTo(r.X+k, r.Y+r.H-k)
					pt.StrokePath(&path, 1.25, t.TextMuted)
				})
				x.afterInput(func() {
					if x.Clicked() {
						if at := slices.Index(*tokens, tok); at >= 0 {
							*tokens = slices.Delete(*tokens, at, at+1)
							changed = true
							c.rt.consumed = true
						}
					}
				})
			})
		}
		// The input keeps its identity, whatever tokens come before it.
		var in *node
		coreBox(c).Key("input").Grow(1).MinWidth(t.Space(20)).Children(func() {
			query = coreLocal(c.parent, "query", func() string { return "" })
			p = comboboxBase(c, query, false)
			in = p.Input
			if len(suggestions) == 0 {
				in.role = RoleAuto // a text field
			}
			in.st.editor.leaveEmptyBackspace = true
			in.afterInput(func() {
				if in.Shortcut(0, KeyBackspace) && len(*tokens) > 0 {
					*tokens = (*tokens)[:len(*tokens)-1]
					changed = true
				}
			})
		})
		return in
	})
	f.Wrap()
	p.anchor = f

	var shown []string
	if q := strings.TrimSpace(*query); q != "" && p.Filtering() {
		for _, s := range matching(suggestions, q) {
			if !slices.Contains(*tokens, s) {
				shown = append(shown, s)
			}
		}
	}
	styleOptions(c, p, shown)
	f.afterInput(func() {
		if strings.Contains(*query, ",") {
			parts := strings.Split(*query, ",")
			for _, s := range parts[:len(parts)-1] {
				add(s)
			}
			*query = parts[len(parts)-1]
		}
		if v, ok := p.Chosen(); ok {
			add(v)
			*query = ""
		} else if in.Submitted() {
			add(*query)
			*query = ""
		}
		if changed {
			f.st.markChanged()
			c.rt.consumed = true
		}
	})
	return f
}
