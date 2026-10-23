package ui

// toolbarItem is a control a toolbar had no room for, which its overflow
// menu shows.
type toolbarItem struct {
	id                    uint64
	label                 string
	toggle, checked, menu bool
	disabled, startsGroup bool
}

// Toolbar creates a row of the controls that fn builds, as along the top
// of a window: buttons, toggles, toggle groups, segmented controls and
// menu buttons, which take no face of their own until hovered. It is one
// stop of Tab, and Left and Right move the focus among its controls, Home
// and End to the first and the last; assistive technology sees a toolbar,
// which Label names. Controls that do not fit go, from the last, into a
// menu at its end, where choosing one clicks it. Spacer pushes those after
// it to the end.
//
//	ui.Toolbar(c, func() {
//		ui.ToggleGroup(c, func() {
//			ui.Toggle(c, &app.bold, "Bold")
//			ui.Toggle(c, &app.italic, "Italic")
//		})
//		ui.Spacer(c)
//		if ui.Button(c, "Share").Clicked() {
//			app.share()
//		}
//	}).Label("Format")
func coreToolbar(c *context, fn func()) *node {
	t := c.theme
	tb := coreRow(c).AlignItems(Center).Gap(t.Space(1)).Padding(t.Space(1)).MinWidth(0)
	tb.widget, tb.role = "Toolbar", RoleToolbar
	tb.FocusGroup(Horizontal)
	items := coreLocal(tb, "overflow", func() []toolbarItem { return nil })
	tb.overflow = items
	saved := c.buttons
	c.buttons = toolbarButtons
	tb.Children(func() {
		fn()
		more := coreButtonBase(c)
		styleButton(c, more, false)
		more.Label(c.texts().More()).Children(func() { chevrons(c) })
		more.Menu(func(m *Menu) {
			for i, it := range *items {
				if it.startsGroup && i > 0 {
					m.Separator()
				}
				mi := m.Item(it.label).Disabled(it.disabled)
				if it.toggle {
					mi.Checked(it.checked)
				}
				if !mi.Chosen() {
					continue
				}
				// Choosing an item does what clicking it does.
				rt := c.rt
				if s := rt.states[it.id]; s != nil {
					if it.menu {
						x, y := more.st.menuCorner()
						rt.askMenu(s, x, y, true)
					} else {
						rt.clickLater = append(rt.clickLater, it.id)
						rt.consumed = true
					}
				}
			}
		})
	})
	c.buttons = saved
	return tb
}

// chevrons draws », of a toolbar's overflow menu, or « right to left.
func chevrons(c *context) {
	t := c.theme
	e := coreBox(c).Size(t.Space(3), t.Space(3)).Shrink(0)
	rtl := e.rtl()
	e.Draw(func(p *Painter, r Rect) {
		for _, x := range []float32{0.2, 0.5} {
			var path Path
			if rtl {
				path.MoveTo(r.X+r.W*(1-x), r.Y+r.H*0.25).LineTo(r.X+r.W*(0.75-x), r.Y+r.H*0.5).LineTo(r.X+r.W*(1-x), r.Y+r.H*0.75)
			} else {
				path.MoveTo(r.X+r.W*x, r.Y+r.H*0.25).LineTo(r.X+r.W*(x+0.25), r.Y+r.H*0.5).LineTo(r.X+r.W*x, r.Y+r.H*0.75)
			}
			p.StrokePath(&path, 1.5, t.Text)
		}
	})
}

// layoutToolbar takes out of the flow the controls of a toolbar cw wide
// that do not fit, from the last, and notes them for its overflow menu,
// whose button, its last child, shows only then.
func (e *node) layoutToolbar(cw float32) {
	var kids []*node
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.collapsed {
			// Laid out again: the room may have grown.
			ch.flags &^= flagAbsolute | flagInvisible
			ch.collapsed = false
		}
		if ch.flags&flagAbsolute == 0 {
			kids = append(kids, ch)
		}
	}
	if len(kids) == 0 {
		return
	}
	more, items := kids[len(kids)-1], kids[:len(kids)-1]
	width := func(ch *node) float32 { return intrinsic(ch, true) + ch.marginX() }
	total := float32(0)
	for i, ch := range items {
		if i > 0 {
			total += e.gapX
		}
		total += width(ch)
	}
	fits := len(items)
	if total > cw+0.5 {
		room := cw - width(more) - e.gapX
		used := float32(0)
		for fits = 0; fits < len(items); fits++ {
			w := width(items[fits])
			if fits > 0 {
				w += e.gapX
			}
			if used+w > room+0.5 {
				break
			}
			used += w
		}
	}
	collapse := func(ch *node) {
		ch.flags |= flagAbsolute | flagInvisible
		ch.collapsed = true
	}
	// The items of a group of controls, as a toggle group's, are set apart
	// by separators.
	list := (*e.overflow)[:0]
	prevGroup := false
	for _, ch := range items[fits:] {
		n := len(list)
		list = appendToolbarItems(list, ch) // while it has its text
		collapse(ch)
		group := len(list)-n > 1
		if len(list) > n && (group || prevGroup) {
			list[n].startsGroup = true
		}
		if len(list) > n {
			prevGroup = group
		}
	}
	*e.overflow = list
	if fits == len(items) {
		collapse(more)
	}
}

// appendToolbarItems appends the controls of e, or e, to list.
func appendToolbarItems(list []toolbarItem, e *node) []toolbarItem {
	if e.flags&(flagClickable|flagMenuButton) != 0 {
		label := e.label
		if label == "" {
			label = e.innerText()
		}
		return append(list, toolbarItem{
			id: e.id, label: label, toggle: e.flags&flagToggle != 0, checked: e.checked == 2,
			menu: e.flags&flagMenuButton != 0, disabled: e.IsDisabled(),
		})
	}
	for ch := e.first; ch != nil; ch = ch.next {
		list = appendToolbarItems(list, ch)
	}
	return list
}
