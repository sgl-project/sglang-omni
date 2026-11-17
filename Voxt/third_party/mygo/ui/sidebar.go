package ui

import (
	"slices"
	"strings"
	"time"
)

// sidebarBuild is a Sidebar being built: the ID of its item chosen, its
// items in order and those of the last frame, for the keys, how deep the
// items being built are, and the element of the item chosen.
type sidebarBuild struct {
	e        *node
	selected *string
	items    []sidebarEntry
	last     []sidebarEntry
	depth    int
	chosen   *node
}

// sidebarEntry is an item of a sidebar: its ID and its label.
type sidebarEntry struct{ id, label string }

// Sidebar creates a sidebar of the sections and the items that fn builds
// (SidebarSection, SidebarItem), as Finder's and Mail's, scrolling when
// they do not fit: a click chooses an item, whose ID *selected holds, and
// while the sidebar has the focus, Up and Down choose among its items, as
// do Home, End and typing the first letters of an item. It is one stop of
// Tab, and Changed reports a new choice. Give it a width:
//
//	ui.Sidebar(c, &app.mailbox, func() {
//		ui.SidebarSection(c, "Mailboxes", &app.mailboxes, func() {
//			ui.SidebarItem(c, "inbox", inboxIcon, "Inbox").Children(func() {
//				ui.Badge(c, "12")
//			})
//			ui.SidebarItem(c, "sent", sentIcon, "Sent")
//		})
//	}).Width(220)
//
// Assistive technology sees a tree, whose sections hold their items.
func coreSidebar(c *context, selected *string, fn func()) *node {
	t := c.theme
	e := coreScroll(c).Padding(t.Space(2.5), t.Space(2.5)).Gap(t.Space(0.5)).Background(t.Surface).Focusable().Shrink(0)
	e.widget, e.role = "Sidebar", RoleTree
	e.flags |= flagTypeSelect | flagOwnRing
	e.choosesItems = true
	last := coreLocal(e, "items", func() []sidebarEntry { return nil })
	sb := &sidebarBuild{e: e, selected: selected, last: *last}
	saved := c.sidebar
	c.sidebar = sb
	e.Children(fn)
	c.sidebar = saved
	*last = sb.items
	e.afterInput(sb.keys)
	// The item chosen has the focus for assistive technology, as a list's
	// row chosen does.
	e.activeDescendant = sb.chosen
	return e
}

// keys chooses among the items with the arrows, Home, End and the first
// letters of an item.
func (sb *sidebarBuild) keys() {
	e, items := sb.e, sb.last
	at := slices.IndexFunc(items, func(it sidebarEntry) bool { return it.id == *sb.selected })
	to := -1
	switch {
	case e.Shortcut(0, KeyDown):
		to = min(at+1, len(items)-1)
	case e.Shortcut(0, KeyUp):
		to = max(at-1, 0)
	case e.Shortcut(0, KeyHome):
		to = 0
	case e.Shortcut(0, KeyEnd):
		to = len(items) - 1
	}
	if s := e.st; s.typing && s.typed != "" {
		// From the item after the one chosen, round to it: the same letter
		// again goes to the next item starting with it.
		start := at
		if len([]rune(s.typed)) > 1 {
			start = at - 1
		}
		for k := 1; k <= len(items); k++ {
			j := (start + k + len(items)) % len(items)
			if strings.HasPrefix(strings.ToLower(items[j].label), strings.ToLower(s.typed)) {
				to = j
				break
			}
		}
	}
	if to >= 0 && to < len(items) && items[to].id != *sb.selected {
		*sb.selected = items[to].id
		e.st.markChanged()
		e.c.rt.consumed = true
	}
}

// SidebarSection creates a section of a Sidebar under title, holding the
// items fn builds while *open is true, as Finder's: a click on the title
// shows and hides them, as does its arrow, which shows as the pointer
// rests on the title. A nil open keeps the section open.
func coreSidebarSection(c *context, title string, open *bool, fn func()) *node {
	t := c.theme
	sb := c.sidebar
	above := c.parent != nil && c.parent.nchild > 0
	sec := coreColumn(c).Gap(t.Space(0.5)).Shrink(0).Role(RoleNone)
	sec.widget = "SidebarSection"
	if above {
		sec.Margin(t.Space(3), 0, 0, 0)
	}
	shown := open == nil || *open
	sec.Children(func() {
		head := coreRow(c).AlignItems(Center).Padding(t.Space(1), t.Space(2)).Radius(t.Radius).Role(RoleTreeItem)
		head.level = 1
		if open != nil {
			head.flags |= flagClickable | flagHover
			head.expandable, head.expanded = true, *open
			head.afterInput(func() {
				toggle := head.Clicked()
				if head.st.expand != 0 {
					// Assistive technology opening or closing it.
					toggle = (head.st.expand > 0) != *open
					head.st.expand = 0
					c.rt.consumed = true
				}
				if toggle {
					*open = !*open
					shown, head.expanded = *open, *open
				}
			})
		}
		head.Children(func() {
			coreText(c, title).FontSize(t.FontSize - 1).FontWeight(600).TextColor(t.TextMuted).Grow(1).SingleLine()
			if open != nil {
				arrow := disclosureArrow(c, 0)
				turn := arrow.Animate("open", 90*b2f(shown), 150*time.Millisecond)
				rtl := arrow.rtl()
				arrow.paintFn = func(p *Painter, r Rect) {
					if head.Hovered() {
						p.StrokePath(disclosurePath(r, turn, rtl), 1.5, t.TextMuted)
					}
				}
			}
		})
		if shown {
			if sb != nil {
				sb.depth++
			}
			fn()
			if sb != nil {
				sb.depth--
			}
		}
	})
	return sec
}

// SidebarItem creates an item of a Sidebar choosing id, showing label
// after icon, nil for none. Add to it with Children, as a Badge.
func coreSidebarItem(c *context, id string, icon *SVG, label string) *node {
	t := c.theme
	sb := c.sidebar
	item := coreRow(c).Key(id).AlignItems(Center).Gap(t.Space(2)).Padding(t.Space(1.5), t.Space(2)).Radius(t.Radius).Shrink(0).Role(RoleTreeItem)
	item.widget = "SidebarItem"
	item.flags |= flagClickable | flagHover | flagChoosable
	chosen := false
	focused := false
	if sb != nil {
		item.level = sb.depth + 1
		item.afterInput(func() {
			if item.Clicked() && *sb.selected != id {
				*sb.selected = id
				sb.e.st.markChanged()
			}

		})
		chosen, focused = *sb.selected == id, sb.e.Focused()
		sb.items = append(sb.items, sidebarEntry{id, label})
		if chosen {
			sb.chosen = item
		}
	}
	iconColor := t.Accent
	switch {
	case chosen && focused:
		item.checked = 2
		item.Background(t.Accent).TextColor(t.AccentText)
		iconColor = t.AccentText
	case chosen:
		// Chosen in a sidebar without the focus, in gray, as AppKit's.
		item.checked = 2
		item.Background(t.SurfacePressed)
	default:
		item.checked = 1
		item.styleFn = func(item *node) {
			if item.Hovered() {
				item.bg = t.SurfaceHover
			}
		}
	}
	if sb != nil && chosen {
		e := sb.e
		item.DrawOver(func(p *Painter, r Rect) {
			if e.FocusVisible() {
				p.FocusRing(r, item.radius)
			}
		})
	}
	item.Children(func() {
		if icon != nil {
			coreIcon(c, icon).FontSize(16).TextColor(iconColor)
		}
		coreText(c, label).Grow(1).SingleLine()
	})
	return item
}

// Badge creates a short text in a pill, as a count of unread messages
// beside a sidebar's item.
func coreBadge(c *context, text string) *node {
	t := c.theme
	b := coreBox(c).Padding(0, t.Space(1.5)).Radius(999).Background(t.Text.Alpha(0.1)).Shrink(0)
	b.widget = "Badge"
	b.Children(func() { coreText(c, text).FontSize(t.FontSize - 2).SingleLine() })
	return b
}
