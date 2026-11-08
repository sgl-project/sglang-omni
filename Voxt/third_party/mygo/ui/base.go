package ui

import "fmt"

// Bases are widgets without a look. A base handles the pointer, the
// keyboard and the focus, and tells assistive technology what it is,
// and leaves every color, size and shape to the elements it returns,
// which take the element methods: the styled widgets, such as Button and
// Checkbox, are bases with the theme's look. Build a design of your own
// on them:
//
//	cb := ui.CheckboxBase(c, &app.sync).Gap(8)
//	cb.Children(func() {
//		box := ui.Box(c).Size(18, 18).Radius(5).Border(1.5, gray)
//		if app.sync {
//			box.Background(green)
//		}
//		ui.Text(c, "Sync")
//	})

// ButtonBase creates a button without a look: a row centering its
// children that takes the keyboard focus, and reports Clicked for the
// pointer, Enter and Space. Style it and give it children; Button is
// ButtonBase with the theme's look.
func coreButtonBase(c *context) *node {
	b := coreRow(c).Center().Focusable().Shrink(0)
	b.flags |= flagClickable | flagHover
	return b
}

// CheckboxBase creates a check box without a look: a row that toggles
// *checked when clicked, or with Space while it has the focus, which
// Changed reports, and that assistive technology sees as a check box,
// checked or not. Draw its box from *checked; Checkbox is CheckboxBase
// with the theme's look.
func coreCheckboxBase(c *context, checked *bool) *node {
	if g := c.checks; g != nil {
		g.boxes = append(g.boxes, checked)
	}
	return toggle(c, checked, RoleCheckBox, "Checkbox")
}

// SwitchBase creates a switch without a look: CheckboxBase, which
// assistive technology sees as a switch. Switch is SwitchBase with the
// theme's look.
func coreSwitchBase(c *context, on *bool) *node { return toggle(c, on, RoleSwitch, "Switch") }

func toggle(c *context, on *bool, role Role, widget string) *node {
	e := coreRow(c).Focusable().Shrink(0)
	e.flags |= flagClickable | flagHover | flagToggle
	e.widget, e.role = widget, role
	*valueBinding[*bool](e) = on
	e.onValueInput(toggleInput)
	e.checked = 1 + int8(b2f(*on))
	return e
}

// RadioBase creates a radio button without a look: a row that selects
// value into *selected when clicked, or with Space while it has the
// focus, which Changed reports, and that assistive technology sees as a
// radio button, on when *selected is value. Radio is RadioBase with the
// theme's look.
func coreRadioBase[T comparable](c *context, selected *T, value T) *node {
	e := coreRow(c).Focusable().Shrink(0)
	e.flags |= flagClickable | flagHover | flagToggle
	e.widget, e.role = "Radio", RoleRadio
	*valueBinding[radioValue[T]](e) = radioValue[T]{selected: selected, value: value}
	e.onValueInput(radioInput[T])
	e.checked = 1 + int8(b2f(*selected == value))
	return e
}

// SliderBase creates a slider without a look: dragging across its
// content box, within its padding, sets *value between lo and hi, as do
// the arrows, Home and End while it has the focus, by a hundredth of the
// range unless Step says otherwise; Changed reports a new value, and
// assistive technology sees a slider of that range. Draw its track and
// thumb where (*value-lo)/(hi-lo) puts them, and pad it by half the
// thumb's width to keep the thumb inside it. Right to left
// (Context.Direction), lo is at the right of the content box, as the
// arrows go: draw it mirrored.
//
//	s := ui.SliderBase(c, &app.volume, 0, 100).Height(24).PaddingX(12)
//	s.Draw(func(p *ui.Painter, r ui.Rect) {
//		x := r.X + 12 + (r.W-24)*float32(app.volume/100)
//		p.Fill(ui.Rect{X: r.X + 12, Y: r.Y + 10, W: r.W - 24, H: 4}, gray, 2)
//		p.Fill(ui.Rect{X: x - 12, Y: r.Y, W: 24, H: 24}, blue, 12)
//	})
//
// Vertical makes it go up from lo at the bottom. Slider is SliderBase with
// the theme's look.
func coreSliderBase(c *context, value *float64, lo, hi float64) *node {
	s := sliderBase(c, value, lo, hi, 0)
	s.widget = "SliderBase" // which Vertical takes
	return s
}

// sliderSettings are what Step and Vertical say of a slider in this
// build, and what they said in the build before, which the input since
// applies by: the slider the user dragged.
type sliderSettings struct {
	now, shown sliderOptions
}

type sliderOptions struct {
	step     float64
	vertical bool
}

type sliderKey struct{}

// sliderBase creates a SliderBase whose values are lo and the multiples of
// step from it, any for 0.
func sliderBase(c *context, value *float64, lo, hi, step float64) *node {
	s := coreBox(c).Focusable()
	s.flags |= flagDraggable | flagHover
	s.widget = "Slider"
	set := coreLocal(s, sliderKey{}, func() sliderSettings { return sliderSettings{} })
	set.shown, set.now = set.now, sliderOptions{}
	*valueBinding[sliderValue](s) = sliderValue{value: value, lo: lo, hi: hi, step: step}
	s.onValueInput(sliderInput)
	if step <= 0 {
		step = (hi - lo) / 100
	}
	s.accStep = step
	s.role, s.hasRange, s.accRange = RoleSlider, true, [3]float64{lo, hi, *value}
	return s
}

// Step makes the values of a slider lo and the multiples of step from it,
// which the arrows move between, as a StepSlider's. Of a range of your own
// (Range), it tells assistive technology how far the keys move the value.
func (e *node) Step(step float64) *node {
	if step <= 0 {
		return e
	}
	e.accStep = step
	if e.widget == "Slider" || e.widget == "SliderBase" {
		coreLocal(e, sliderKey{}, func() sliderSettings { return sliderSettings{} }).now.step = step
	}
	return e
}

// Vertical makes a SliderBase go up, from lo at the bottom of its content
// box to hi at the top, and tells assistive technology so. Pad it by half
// the thumb's height, rather than its width. Slider, drawn across, stays
// so.
func (e *node) Vertical() *node {
	if e.widget == "SliderBase" {
		coreLocal(e, sliderKey{}, func() sliderSettings { return sliderSettings{} }).now.vertical = true
		e.vertical = true
	}
	return e
}

// TabsParts are the parts of a tab list without a look: TabsBase makes
// the list, and Tab its tabs.
type tabsParts struct {
	// List holds the tabs: build them in its children.
	List     *node
	c        *context
	selected *int
	n        int
	// follow is set when the keys chose a tab, which then takes the
	// focus.
	follow *bool
}

// TabsBase creates a list of n tabs without a look, of which *selected is
// the index of the one chosen. A click chooses a tab, as do the arrows,
// Home and End while one has the keyboard focus, which follows the
// choice; List's Changed reports a new choice. Build the tabs in List
// with Tab:
//
//	tabs := ui.TabsBase(c, &app.tab, len(names))
//	tabs.List.Gap(4).Children(func() {
//		for i, name := range names {
//			tab := tabs.Tab(i).Padding(6, 12).Radius(6)
//			if i == app.tab {
//				tab.Background(c.Theme().Surface)
//			}
//			tab.Children(func() { ui.Text(c, name) })
//		}
//	})
//
// Tabs is TabsBase with the theme's look.
func coreTabsBase(c *context, selected *int, n int) tabsParts {
	list := coreRow(c).Shrink(0).Role(RoleTabList).FocusGroup(Horizontal)
	list.widget = "Tabs"
	if n > 0 {
		*selected = max(0, min(*selected, n-1))
	}
	return tabsParts{List: list, c: c, selected: selected, n: n, follow: coreLocal(list, "follow", func() bool { return false })}
}

// Tab creates tab i of the list: a row that takes the focus and chooses
// i when clicked. Style it from whether i is the one chosen.
func (p tabsParts) Tab(i int) *node {
	c := p.c
	tab := coreRow(c).Focusable().Shrink(0).Role(RoleTab)
	tab.widget = "Tab"
	tab.flags |= flagClickable | flagHover
	tab.afterInput(func() {
		choose := func(i int, keys bool) {
			i = (i%p.n + p.n) % p.n
			if i != *p.selected {
				*p.selected = i
				p.List.st.markChanged()
				c.rt.consumed = true
			}
			*p.follow = keys
		}
		if tab.Clicked() {
			choose(i, false)
		}
		back, forth := tab.arrows()
		switch {
		case tab.Shortcut(0, forth), tab.Shortcut(0, KeyDown):
			choose(i+1, true)
		case tab.Shortcut(0, back), tab.Shortcut(0, KeyUp):
			choose(i-1, true)
		case tab.Shortcut(0, KeyHome):
			choose(0, true)
		case tab.Shortcut(0, KeyEnd):
			choose(p.n-1, true)
		}
	})
	on := i == *p.selected
	if on && *p.follow {
		tab.Focus()
		c.rt.focusVisible = true
		*p.follow = false
	}
	tab.checked = 1 + int8(b2f(on))
	return tab
}

// SelectParts are the parts of a select without a look: SelectBase makes
// its trigger, Popup its popup and Item the options in it.
type selectParts[T comparable] struct {
	// Trigger opens and closes the popup: give it children showing the
	// choice.
	Trigger  *node
	c        *context
	selected *T
	open     *bool
	// highlight is the index of the option the pointer or the arrows are
	// on, -1 until the popup finds the chosen one; values are the
	// options of the last frame that showed them, and next those of this
	// frame. pointer is where the pointer was when it last highlighted an
	// option: it highlights another once it moves. shown is the option
	// last scrolled into view as the highlight moved to it, while shows
	// says whether it is one.
	highlight *int
	values    *[]T
	next      []T
	pointer   *[2]float32
	shown     *T
	shows     *bool
}

// SelectBase creates a select without a look: a trigger that opens a
// popup of options, which choose *selected. With the focus on the
// trigger, the arrows open the popup and move among its options, Enter
// and Space choose one, and Escape closes it; the trigger's Changed
// reports a new choice. Give the trigger children showing the choice, and
// build the options in Popup with Item:
//
//	sel := ui.SelectBase(c, &app.size)
//	sel.Trigger.Padding(6, 10).Border(1, gray).Children(func() {
//		ui.Text(c, app.size)
//	})
//	sel.Popup(func(panel ui.Element) {
//		panel.Padding(4).Background(white).Border(1, gray)
//		for _, size := range sizes {
//			item := sel.Item(size).Padding(6, 10)
//			if item.Highlighted() {
//				item.Background(blue).TextColor(white)
//			}
//			item.Children(func() { ui.Text(c, size) })
//		}
//	})
//
// Select is SelectBase with the theme's look.
func coreSelectBase[T comparable](c *context, selected *T) *selectParts[T] {
	b := coreButtonBase(c)
	b.widget, b.role, b.accValue = "Select", RolePopUpButton, fmt.Sprint(*selected)
	s := &selectParts[T]{
		Trigger: b, c: c, selected: selected,
		open:      coreLocal(b, "open", func() bool { return false }),
		highlight: coreLocal(b, "highlight", func() int { return -1 }),
		values:    coreLocal(b, "values", func() []T { return nil }),
		pointer:   coreLocal(b, "pointer", func() [2]float32 { return [2]float32{} }),
		shown:     coreLocal(b, "shown", func() T { var zero T; return zero }),
		shows:     coreLocal(b, "shows", func() bool { return false }),
	}
	b.afterInput(func() {
		open := func(o bool) {
			*s.open, *s.highlight, *s.shows = o, -1, false
			*s.pointer = [2]float32{c.rt.pointerX, c.rt.pointerY}
			c.rt.consumed = true
		}
		if b.Clicked() {
			open(!*s.open)
		}
		n := len(*s.values)
		move := func(i int) {
			*s.highlight = max(0, min(i, n-1))
			c.rt.consumed = true
		}
		if !*s.open {
			if b.Shortcut(0, KeyDown) || b.Shortcut(0, KeyUp) {
				open(true)
			}
		} else if n > 0 {
			switch {
			case b.Shortcut(0, KeyDown):
				move(*s.highlight + 1)
			case b.Shortcut(0, KeyUp):
				move(*s.highlight - 1)
			case b.Shortcut(0, KeyHome):
				move(0)
			case b.Shortcut(0, KeyEnd):
				move(n - 1)
			case b.Shortcut(0, KeyEnter), b.Shortcut(0, KeySpace):
				if h := *s.highlight; h >= 0 && h < n {
					s.choose((*s.values)[h])
				}
			}
		}
	})
	b.expanded = *s.open
	return s
}

func (s *selectParts[T]) choose(v T) {
	if *s.selected != v {
		*s.selected = v
		s.Trigger.st.markChanged()
	}
	*s.open = false
	s.c.rt.consumed = true
}

// Open reports whether the popup shows.
func (s *selectParts[T]) Open() bool { return *s.open }

// Highlight moves the highlight to the option of value while the popup
// shows, as a select of your own may do as the user types the option's
// first letters; Enter then chooses it.
func (s *selectParts[T]) Highlight(value T) {
	for i, v := range *s.values {
		if v == value {
			if *s.highlight != i {
				*s.highlight = i
				s.c.rt.consumed = true
			}
			return
		}
	}
}

// Popup shows the popup below the trigger, at least as wide, while it is
// open: fn styles the panel and builds the options in it with Item.
// Clicking outside it or pressing Escape closes it. It returns the panel,
// or nil when the popup is closed.
func (s *selectParts[T]) Popup(fn func(panel *node)) *node {
	b := s.Trigger
	return popover(s.c, b, s.open, true, func(panel *node) {
		panel.MinWidth(b.Bounds().W)
		s.next = s.next[:0]
		fn(panel)
		*s.values = append((*s.values)[:0], s.next...)
	})
}

// Item creates an option choosing value: a row that is Highlighted when
// the pointer or the arrows are on it, and chooses value and closes the
// popup when clicked. The option the highlight moves to scrolls into view,
// once, so that the wheel scrolls on past it.
func (s *selectParts[T]) Item(value T) *node {
	c := s.c
	i := len(s.next)
	s.next = append(s.next, value)
	item := coreRow(c)
	item.flags |= flagClickable | flagHover
	if *s.highlight < 0 && value == *s.selected {
		*s.highlight = i
	}
	if p := [2]float32{c.rt.pointerX, c.rt.pointerY}; item.Hovered() && p != *s.pointer {
		// The pointer moved onto it, which moves the highlight; the
		// options built before learn it as the frame builds again.
		*s.pointer = p
		if *s.highlight != i {
			*s.highlight = i
			c.rt.consumed = true
		}
	}
	item.highlighted = *s.highlight == i
	if item.highlighted && (!*s.shows || *s.shown != value) {
		*s.shown, *s.shows = value, true
		item.ScrollIntoView()
	}
	item.checked = 1 + int8(b2f(value == *s.selected))
	item.afterInput(func() {
		if item.Clicked() {
			s.choose(value)
		}
	})
	return item
}

// Highlighted reports whether an option of a select is the one the pointer
// or the arrows are on.
func (e *node) Highlighted() bool { return e.highlighted }

// PopoverBase shows a panel without a look below anchor while *open is
// true: fn styles the panel and builds its content. Pressing outside the
// panel and the anchor, or Escape, sets *open to false, which Dismissed
// reports; the press goes on to what is under the pointer, as with the
// web's popovers. It starts at the anchor's start: its left, or its right
// right to left (Direction). Where there is
// no room below the anchor, the panel shows above it; a top margin keeps
// it apart from the anchor on either side. fn may place it elsewhere with
// AttachTo, as after the anchor:
//
//	panel.AttachTo(anchor, ui.AnchorEnd, ui.AnchorStart).MarginStart(6)
//
// It returns the panel, or nil while closed; Popover is PopoverBase with
// the theme's look.
func corePopoverBase(c *context, anchor *node, open *bool, fn func(panel *node)) *node {
	return popover(c, anchor, open, false, fn)
}

// popover shows the panel of a PopoverBase, over a backdrop taking the
// presses outside it with modal, as a select's popup: a menu of the
// system's takes them too.
func popover(c *context, anchor *node, open *bool, modal bool, fn func(panel *node)) *node {
	if !*open {
		return nil
	}
	var panel *node
	dismissed := false
	coreOverlay(c, func() {
		if modal {
			back := coreBox(c).Absolute().Left(0).Top(0).Right(0).Bottom(0)
			back.flags |= flagClickable
			back.popover = anchor
			if back.Clicked() {
				*open = false
				dismissed = true
			}
		}
		panel = coreBox(c).Role(RolePopup).AttachTo(anchor, AnchorBottomStart, AnchorTopStart)
		// Presses on it stay in it.
		panel.flags |= flagClickable
		panel.Children(func() { fn(panel) })
		if panel.OverlayShortcut(0, KeyEscape) || !modal && panel.PressedOutside() {
			*open = false
			dismissed = true
		}
		panel.dismissed = dismissed
	})
	return panel
}

// DialogBase shows a dialog without a look over the window while *open is
// true: fn styles the backdrop covering the window, which centers the
// panel, and the panel, and builds the panel's content. Clicking the
// backdrop or pressing Escape sets *open to false, which Dismissed
// reports. It returns the panel, or nil while closed; Modal is DialogBase
// with the theme's look.
func coreDialogBase(c *context, open *bool, fn func(backdrop, panel *node)) *node {
	if !*open {
		return nil
	}
	var panel *node
	coreOverlay(c, func() {
		back := coreBox(c).Absolute().Left(0).Top(0).Right(0).Bottom(0).Center().Modal()
		back.flags |= flagClickable
		dismissed := back.Clicked() || back.OverlayShortcut(0, KeyEscape)
		if dismissed {
			*open = false
		}
		back.Children(func() {
			panel = coreBox(c).Role(RoleDialog)
			panel.dismissed = dismissed
			panel.flags |= flagClickable
			panel.Children(func() { fn(back, panel) })
		})
	})
	return panel
}

// TextInputBase creates a single-line text input without a look, editing
// *value: TextInput without its padding, background, border and
// corners.
func coreTextInputBase(c *context, value *string) *node { return textInputBase(c, value, false) }

// TextAreaBase creates a multi-line text input without a look, editing
// *value: TextArea without its padding, background, border and corners.
func coreTextAreaBase(c *context, value *string) *node { return textInputBase(c, value, true) }
