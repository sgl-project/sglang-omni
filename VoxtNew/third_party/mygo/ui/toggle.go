package ui

// buttonStyle is how the buttons being built look: as on their own, in a
// toolbar, or as the segments of a ToggleGroup or Segmented.
type buttonStyle uint8

const (
	ownButtons buttonStyle = iota
	toolbarButtons
	segmentButtons
)

// ToggleBase creates a toggle button without a look: a row that turns *on
// over when clicked, or with Space while it has the focus, which Changed
// reports, and that assistive technology sees as a toggle button, pressed
// while *on. Toggle is ToggleBase with the theme's look.
func coreToggleBase(c *context, on *bool) *node {
	return toggle(c, on, RoleToggleButton, "Toggle").Center()
}

// Toggle creates a button that stays pressed while *on, as Bold in an
// editor's toolbar: a click presses it or lets it go, as does Space while
// it has the focus, and Changed reports a change. Give it other content,
// such as an icon, with Children and an empty label, and name it with
// Label:
//
//	ui.Toggle(c, &app.bold, "").Label("Bold").Children(func() { ui.Icon(c, boldIcon) })
func coreToggle(c *context, on *bool, label string) *node {
	b := coreToggleBase(c, on)
	styleButton(c, b, false)
	if *on {
		b.Background(pressedColor(c))
		if c.buttons == segmentButtons {
			b.Shadow(0, 1, 2, 0, RGBA(0, 0, 0, 0.12))
		}
	}
	if label != "" {
		b.Children(func() { coreText(c, label).SingleLine() })
	}
	return b
}

// pressedColor is the face of a toggle that is on, and of the segment
// chosen: raised from the track of a group, sunk elsewhere.
func pressedColor(c *context) Color {
	t := c.theme
	switch {
	case c.buttons != segmentButtons:
		return t.SurfacePressed
	case t.Dark:
		return t.SurfacePressed
	}
	return t.Background
}

// segmentTrack creates the track of a ToggleGroup or Segmented, which
// holds its segments.
func segmentTrack(c *context) *node {
	t := c.theme
	g := coreRow(c).Shrink(0).AlignItems(Stretch).Padding(t.Space(0.5)).Gap(t.Space(0.5)).
		Radius(t.Radius).Background(t.Surface).Border(1, t.Border)
	g.FocusGroup(Horizontal | Vertical)
	return g
}

// ToggleGroup creates a row of the toggles and buttons that fn builds,
// drawn joined as the segments of one control, as Bold, Italic and
// Underline. They are one stop of Tab, among which the arrows move the
// focus, as in a toolbar; name the group with Label.
//
//	ui.ToggleGroup(c, func() {
//		ui.Toggle(c, &app.bold, "B")
//		ui.Toggle(c, &app.italic, "I")
//	}).Label("Style")
func coreToggleGroup(c *context, fn func()) *node {
	g := segmentTrack(c)
	g.widget = "ToggleGroup"
	saved := c.buttons
	c.buttons = segmentButtons
	g.Children(fn)
	c.buttons = saved
	return g
}

// RadioGroup creates a column of the radio buttons that fn builds (Radio,
// RadioBase), which are one stop of Tab: Tab goes to the one chosen, and
// the arrows choose among them, as do Home and End. Assistive technology
// sees a radio group; name it with Label. Make it a row with Row.
//
//	ui.RadioGroup(c, func() {
//		for _, size := range []string{"Small", "Medium", "Large"} {
//			ui.Radio(c, &app.size, size, size)
//		}
//	}).Label("Size")
func coreRadioGroup(c *context, fn func()) *node {
	g := coreColumn(c).Gap(c.theme.Space(2))
	g.widget, g.role = "RadioGroup", RoleRadioGroup
	g.FocusGroup(Horizontal | Vertical)
	g.groupSelects = true
	g.Children(fn)
	return g
}

// SegmentedParts are the parts of a segmented control without a look:
// SegmentedBase makes its track, and Segment its segments.
type segmentedParts struct {
	// Track holds the segments: build them in its children.
	Track    *node
	c        *context
	selected *int
	n        int
}

// SegmentedBase creates a segmented control of n segments without a look,
// of which *selected is the index of the one chosen: a radio group (see
// RadioGroup) whose radio buttons are the segments, built in Track with
// Segment. Track's Changed reports a new choice. Segmented is
// SegmentedBase with the theme's look.
func coreSegmentedBase(c *context, selected *int, n int) segmentedParts {
	g := coreRow(c).Shrink(0)
	g.widget, g.role = "Segmented", RoleRadioGroup
	g.FocusGroup(Horizontal | Vertical)
	g.groupSelects = true
	if n > 0 {
		*selected = max(0, min(*selected, n-1))
	}
	return segmentedParts{Track: g, c: c, selected: selected, n: n}
}

// Segment creates segment i: a radio button choosing i. Style it from
// whether i is the one chosen, and give it children.
func (p segmentedParts) Segment(i int) *node {
	s := coreRadioBase(p.c, p.selected, i).Center()
	s.segment = true
	s.afterInput(func() {
		if s.Changed() {
			p.Track.st.markChanged()
		}
	})
	return s
}

// Segmented creates a segmented control showing labels, of which
// *selected is the index of the one chosen, as a switch between views: a
// click chooses a segment, as do the arrows while it has the focus, and
// Changed reports a new choice. Assistive technology sees a radio group;
// name it with Label. For icons, build the segments with SegmentedBase.
//
//	ui.Segmented(c, &app.view, "List", "Grid").Label("View")
func coreSegmented(c *context, selected *int, labels ...string) *node {
	t := c.theme
	parts := coreSegmentedBase(c, selected, len(labels))
	g := parts.Track
	g.AlignItems(Stretch).Padding(t.Space(0.5)).Gap(t.Space(0.5)).Radius(t.Radius).Background(t.Surface).Border(1, t.Border)
	saved := c.buttons
	c.buttons = segmentButtons
	g.Children(func() {
		for i, label := range labels {
			s := parts.Segment(i)
			styleButton(c, s, false)
			if i == *selected {
				s.Background(pressedColor(c)).Shadow(0, 1, 2, 0, RGBA(0, 0, 0, 0.12))
			}
			s.Children(func() { coreText(c, label).SingleLine() })
		}
	})
	c.buttons = saved
	return g
}
