package ui

import (
	"runtime"
)

// checkGroup is a CheckboxGroup being built: the values of its check boxes.
type checkGroup struct {
	boxes []*bool
}

// CheckboxGroup creates a check box over the check boxes that fn builds,
// indented below it, as a list of permissions: it is checked when they all
// are, mixed when some are, and a click, or Space while it has the focus,
// checks them all, or none when they all are. Changed reports a click.
// Assistive technology sees a check box, mixed while some are checked,
// and a group of the others named by label.
//
//	ui.CheckboxGroup(c, "Notifications", func() {
//		ui.Checkbox(c, &app.mail, "Mail")
//		ui.Checkbox(c, &app.calendar, "Calendar")
//	})
func coreCheckboxGroup(c *context, label string, fn func()) *node {
	t := c.theme
	g := coreColumn(c).Gap(t.Space(2)).Shrink(0)
	g.widget = "CheckboxGroup"
	var head *node
	var group *checkGroup
	g.Children(func() {
		head = coreRow(c).Focusable().Shrink(0).Gap(t.Space(2)).FocusRing(false).Role(RoleCheckBox)
		head.flags |= flagClickable | flagHover | flagToggle
		head.widget = "CheckboxGroup"
		head.flags |= flagClickable
		var box *node
		head.Children(func() {
			box = coreBox(c).Size(t.Space(4), t.Space(4)).Radius(t.Space(1)).Shrink(0)
			coreText(c, label)
		})
		saved := c.checks
		group = &checkGroup{}
		c.checks = group
		boxes := coreColumn(c).Gap(t.Space(2)).PaddingStart(t.Space(6)).Role(RoleGroup).Label(label)
		boxes.Children(fn)
		c.checks = saved
		on := 0
		for _, b := range group.boxes {
			if *b {
				on++
			}
		}
		all := on == len(group.boxes) && on > 0
		head.afterInput(func() {
			if head.Clicked() {
				for _, b := range group.boxes {
					*b = !all
				}
				g.st.markChanged()
				c.rt.consumed = true
			}
		})
		state := int8(1)
		switch {
		case all:
			state = 2
		case on > 0:
			state = 3
		}
		head.checked = state
		paintCheckbox(c, head, box, state)
	})
	return g
}

// paintCheckbox styles the box of a check box, off (1), on (2) or mixed
// (3), as Checkbox does.
func paintCheckbox(c *context, row, box *node, state int8) {
	t := c.theme
	if state > 1 {
		box.Background(t.Accent)
	} else {
		box.Background(t.Background).Border(1, t.Border.Mix(t.Text, 0.25))
	}
	box.DrawOver(func(p *Painter, r Rect) {
		switch state {
		case 2:
			p.StrokePath(checkPath(r), t.Space(0.5), t.AccentText)
		case 3:
			p.Fill(Rect{r.X + r.W*0.25, r.Y + r.H/2 - 1, r.W * 0.5, 2}, t.AccentText, 1)
		}
		if row.FocusVisible() {
			p.FocusRing(r, box.radius)
		}
	})
	box.styleFn = func(box *node) {
		if state == 1 && row.Hovered() {
			box.borderC = t.Accent
		}
	}
}

// Breadcrumbs creates a path of items, as Finder's path bar and AppKit's
// path control, separated by chevrons: the items but the last are links,
// which a click chooses, as Enter does, setting *chosen to its index, which
// Changed reports. Long items shorten with an ellipsis as room runs short.
// Assistive technology sees links in a group; name it with Label.
//
//	path := []string{"Macintosh HD", "Users", "ada", "Documents"}
//	if ui.Breadcrumbs(c, path, &app.chosen).Label("Path").Changed() {
//		app.open(path[:app.chosen+1])
//	}
func coreBreadcrumbs(c *context, items []string, chosen *int) *node {
	t := c.theme
	e := coreRow(c).AlignItems(Center).Gap(t.Space(0.5)).MinWidth(0).Role(RoleGroup)
	e.widget = "Breadcrumbs"
	e.Children(func() {
		for i, item := range items {
			if i > 0 {
				sep := coreBox(c).Size(t.Space(3), t.Space(3)).Shrink(0).Role(RoleNone)
				rtl := sep.rtl()
				sep.Draw(func(p *Painter, r Rect) {
					cx, cy, d := r.X+r.W/2, r.Y+r.H/2, r.W/6
					if rtl {
						// Pointing left, along the path.
						d = -d
					}
					var path Path
					path.MoveTo(cx-d, cy-2*abs32(d)).LineTo(cx+d, cy).LineTo(cx-d, cy+2*abs32(d))
					p.StrokePath(&path, 1.2, t.TextMuted)
				})
			}
			last := i == len(items)-1
			b := coreRow(c).Key(i).Padding(t.Space(0.5), t.Space(1.5)).Radius(t.Radius).MinWidth(t.Space(6)).Shrink(1)
			b.Children(func() { coreText(c, item).SingleLine() })
			if last {
				// Where the path is: no link.
				b.Shrink(0.5).FontWeight(500)
				continue
			}
			b.Focusable().Role(RoleLink).TextColor(t.TextMuted)
			b.flags |= flagClickable | flagHover
			b.afterInput(func() {
				if b.Clicked() {
					*chosen = i
					e.st.markChanged()
				}
			})
			b.styleFn = func(b *node) {
				if b.Hovered() {
					b.bg = t.SurfaceHover
				}
			}
		}
	})
	return e
}

// AlertDismissed is what AlertDialog returns in the frame Escape closes the
// alert, choosing no button.
const AlertDismissed = -2

// AlertDialog shows an alert over the window while *open is true, as
// AppKit's: title, message, and buttons, the last of which is the default,
// in the accent color, with the focus, which Enter clicks. Escape closes the
// alert choosing no button, as canceling does, whatever the buttons say; a
// click outside the alert does nothing. It returns the index of the button
// clicked, in the frame it is, which closes the alert, AlertDismissed in
// the frame Escape closes it, and -1 otherwise. An alert that must be
// answered stays open by setting *open back to true on AlertDismissed.
// Assistive technology sees an alert named by title and described by
// message.
//
//	switch ui.AlertDialog(c, &app.asking, "Delete “Notes”?", "You can't undo this.", "Cancel", "Delete") {
//	case 1:
//		app.delete()
//	}
func coreAlertDialog(c *context, open *bool, title, message string, buttons ...string) int {
	if !*open {
		return -1
	}
	t := c.theme
	chosen, dismissed := -1, false
	coreOverlay(c, func() {
		back := coreBox(c).Absolute().Left(0).Top(0).Right(0).Bottom(0).Center().Background(RGBA(0, 0, 0, 0.4)).Modal()
		dismissed = back.OverlayShortcut(0, KeyEscape)
		back.Children(func() {
			panel := coreColumn(c).Width(t.Space(75)).MaxWidth(c.w - t.Space(10)).Padding(t.Space(5)).Gap(t.Space(3)).Radius(t.Space(2.5)).Background(t.Background).Role(RoleAlertDialog)
			panel.Label(title)
			panel.description = message
			panel.Shadow(0, 10, 30, 0, RGBA(0, 0, 0, 0.3))
			panel.Children(func() {
				coreText(c, title).Bold().FontSize(t.FontSize + 1)
				if message != "" {
					coreText(c, message).TextColor(t.TextMuted)
				}
				coreRow(c).Gap(t.Space(2)).Justify(End).Margin(t.Space(2), 0, 0, 0).Children(func() {
					for i, label := range buttons {
						var b *node
						if i == len(buttons)-1 {
							b = corePrimaryButton(c, label).AutoFocus()
						} else {
							b = coreButton(c, label)
						}
						if b.Clicked() {
							chosen = i
						}
					}
				})
			})
		})
	})
	if dismissed && chosen < 0 {
		chosen = AlertDismissed
	}
	if chosen != -1 {
		*open = false
		c.rt.consumed = true
	}
	return chosen
}

// FindBar creates a bar for finding text while *open is true, as Safari's
// and Xcode's: a field of *query, which takes the focus as the bar opens,
// how many matches there are of matches, previous and next buttons
// choosing *current among them, and Done. Enter in the field goes to the
// next match, Shift+Enter to the previous, as do Cmd+G and Shift+Cmd+G on
// macOS, F3 and Shift+F3 elsewhere, round the ends; Escape and Done close
// the bar. The app finds: it counts the matches of *query and shows match
// *current, which Changed reports moving. Assistive technology hears the
// count as it changes. While closed, it returns an element showing
// nothing.
func coreFindBar(c *context, open *bool, query *string, matches int, current *int) *node {
	t := c.theme
	if !*open {
		// Nothing, which Changed and the rest can ask.
		e := coreBox(c).Absolute()
		e.flags |= flagInvisible
		return e
	}
	// Keyed, for its state to go as it closes, to open anew.
	bar := coreRow(c).Key("find bar").Gap(t.Space(2)).Padding(t.Space(1.5), t.Space(3)).AlignItems(Center).Background(t.Surface).
		BorderWidth(0, 0, 1, 0).BorderColor(t.Border).Shrink(0).Role(RoleToolbar).Label(c.texts().Find())
	bar.widget = "FindBar"
	// Its state goes as it closes: true in the frame it opens.
	opening := coreLocal(bar, "opening", func() bool { return true })
	*current = max(0, min(*current, matches-1))
	step := func(d int) {
		if matches > 0 {
			*current = (*current + d + matches) % matches
			bar.st.markChanged()
			c.rt.consumed = true
		}
	}
	bar.afterInput(func() {
		mac := runtime.GOOS == "darwin"
		switch {
		case mac && c.Shortcut(Cmd, KeyG), !mac && c.Shortcut(0, KeyF3):
			step(1)
		case mac && c.Shortcut(Shift|Cmd, KeyG), !mac && c.Shortcut(Shift, KeyF3):
			step(-1)
		}

	})
	bar.Children(func() {
		f, in := field(c, func() *node {
			magnifier(c)
			in := coreTextInputBase(c, query).Grow(1).MinWidth(t.Space(25)).Placeholder(c.texts().Find())
			in.search = true
			return in
		})
		f.Label(c.texts().Find()).Grow(1).MaxWidth(t.Space(90))
		if *opening {
			// Opening the bar focuses its field, with what was found before
			// chosen, ready to type over.
			*opening = false
			in.Focus()
			if ed := in.st.editor; ed != nil {
				ed.selectAll()
			}
		}
		in.afterInput(func() {
			switch {
			case in.Submitted() && in.st.submitMods&Shift != 0:
				step(-1)
			case in.Submitted():
				step(1)
			case in.Shortcut(0, KeyEscape):
				*open = false
				c.rt.consumed = true
			}

		})
		status := c.texts().NoMatches()
		switch {
		case *query == "":
			status = ""
		case matches > 0:
			status = c.texts().Matches(*current+1, matches)
		}
		coreText(c, status).TextColor(t.TextMuted).FontFeatures("tnum").MinWidth(t.Space(18)).Role(RoleStatus)
		for _, b := range []struct {
			label string
			d     int
		}{{c.texts().Previous(), -1}, {c.texts().Next(), 1}} {
			btn := coreButton(c, "").Padding(t.Space(1.5), t.Space(2)).Label(b.label).Disabled(matches == 0)
			up := b.d < 0
			btn.Children(func() {
				coreBox(c).Size(t.Space(3), t.Space(3)).Shrink(0).Draw(func(p *Painter, r Rect) {
					cx, cy, d := r.X+r.W/2, r.Y+r.H/2, r.W/4
					var path Path
					if up {
						path.MoveTo(cx-d, cy+d/2).LineTo(cx, cy-d/2).LineTo(cx+d, cy+d/2)
					} else {
						path.MoveTo(cx-d, cy-d/2).LineTo(cx, cy+d/2).LineTo(cx+d, cy-d/2)
					}
					p.StrokePath(&path, 1.5, t.Text)
				})
			})
			btn.afterInput(func() {
				if btn.Clicked() {
					step(b.d)
				}
			})
		}
		done := coreButton(c, c.texts().Done())
		done.afterInput(func() {
			if done.Clicked() {
				*open = false
				c.rt.consumed = true
			}
		})
	})
	return bar
}
