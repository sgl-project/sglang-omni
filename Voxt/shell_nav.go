package main

// The sidebar's layouts (M7a): the full sidebar, resizable, whose width
// and closed sections the settings remember, and the icon rail
// (Settings → Appearance, or the ‹ button) that keeps only the icons;
// until the user chooses, the rail shows while the window is narrow.

import (
	"slices"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// railWidth is the icon rail's width, DIPs at 100 %.
const railWidth = 52

// navLocalUI is the sidebar's interface-local state: the choice, the
// rail's popover, what the view measures, the sections' open state and
// the drag of the width.
type navLocalUI struct {
	nav         *string // the sidebar's choice (syncNav follows the router)
	noticesOpen *bool   // the rail's popover of notifications
	narrow      *bool   // the view writes it: the window is narrower than autoRailWidth

	sections map[string]*bool // sections: open or closed (seeded from the settings)

	sideW      float32   // the sidebar's width, DIPs at the theme's scale
	sideWScale float32   // the px factor sideW was computed at
	sideWAt    time.Time // when the user last resized it (saved after a pause)
}

// navSectionOpen is the open state of a sidebar section, seeded from the
// settings' closed sections on first use.
func navSectionOpen(local *navLocalUI, state *navViewState, id string) *bool {
	if local.sections == nil {
		local.sections = map[string]*bool{}
	}
	p, ok := local.sections[id]
	if !ok {
		open := !slices.Contains(state.ClosedSections, id)
		p = &open
		local.sections[id] = p
	}
	return p
}

// navIcon is a page's icon in the sidebar and the rail.
func navIcon(path string) icons.Icon {
	for _, g := range navGroups {
		for _, p := range g.pages {
			if p.path == path {
				return p.icon
			}
		}
	}
	return icons.Settings
}

// sidebarFrame lays out the sidebar and the page: the rail, or the full
// sidebar beside a divider that resizes it.
func sidebarFrame(c *ui.Context, state *navViewState, notices []noticeViewState, actions navActions, noticeActs noticeActions, local *navLocalUI, page func()) {
	t := c.Theme()
	k := tok(t)
	headerInset = 0
	w, _ := c.Size()
	*local.narrow = w < px(t, autoRailWidth)
	// The layout: the user's choice, else the rail while the window is
	// narrow (the same rule as railMode, for the frame being laid out).
	rail := *local.narrow
	if state.RailChosen {
		rail = state.Rail
	}
	if rail {
		// The window controls (the traffic lights) reach past the rail:
		// page titles move over to keep clear of them.
		headerInset = max(0, c.TitleBar().Left-px(t, railWidth)-px(t, sp6)+px(t, sp3))
		ui.Row(c).Grow(1).MinHeight(0).AlignItems(ui.Stretch).Children(func() {
			ui.Column(c).Width(px(t, railWidth)).Shrink(0).Background(k.Sidebar).Children(func() {
				navRail(c, state, notices, actions, noticeActs, local)
			})
			ui.Box(c).Width(1).Background(k.Line)
			ui.Column(c).Grow(1).MinWidth(0).Background(k.Bg).Children(page)
		})
		return
	}
	*local.noticesOpen = false // the rail's popover
	scale := px(t, 1)
	if local.sideW == 0 || local.sideWScale != scale {
		w := state.SidebarWidth
		if w == 0 {
			w = sidebarWidth
		}
		local.sideW, local.sideWScale = px(t, float32(w)), scale
	}
	split := ui.Split(c.Key("sidebar-split"), &local.sideW, func() {
		ui.Column(c).Fill().Background(k.Sidebar).Children(func() { sidebar(c, state, notices, actions, noticeActs, local) })
	}, func() {
		ui.Column(c).Grow(1).MinWidth(0).Background(k.Bg).Children(page)
	}).Grow(1).MinHeight(0)
	// Keep the width in its bounds, and save it once the drag rests.
	lo, hi := px(t, float32(state.MinSidebarWidth)), px(t, float32(state.MaxSidebarWidth))
	if local.sideW < lo || local.sideW > hi {
		local.sideW = min(max(local.sideW, lo), hi)
	}
	if split.Changed() {
		local.sideWAt = time.Now()
		c.After(600 * time.Millisecond)
	}
	if !local.sideWAt.IsZero() && time.Since(local.sideWAt) >= 500*time.Millisecond {
		local.sideWAt = time.Time{}
		actions.sidebarWidthChanged(float64(local.sideW / scale))
	}
}

// navRail is the sidebar as a column of icons; their titles show as
// tooltips, a badge as a dot.
func navRail(c *ui.Context, state *navViewState, notices []noticeViewState, actions navActions, noticeActs noticeActions, local *navLocalUI) {
	t := c.Theme()
	k := tok(t)
	if bar := c.TitleBar(); bar.Left > 0 {
		// The traffic lights sit at the top of the rail: the button goes
		// under them, in a row of its own.
		ui.Box(c).Height(titleBarHeight).Shrink(0).DragWindow()
		ui.Row(c).Shrink(0).Justify(ui.Center).Children(func() {
			if iconButton(c.Key("rail-off"), icons.SidebarMinimalistic, i18n.T("Show sidebar")).Clicked() {
				actions.setRail(false)
			}
		})
	} else {
		ui.Row(c).Height(titleBarHeight).Shrink(0).DragWindow().AlignItems(ui.End).Justify(ui.Center).Padding(0, 0, px(t, sp1), 0).Children(func() {
			if iconButton(c.Key("rail-off"), icons.SidebarMinimalistic, i18n.T("Show sidebar")).Clicked() {
				actions.setRail(false)
			}
		})
	}
	// Spaced to hold every icon in the smallest window (680×440); a
	// shorter one scrolls the list, with a fade at its foot as the
	// sidebar's.
	ui.Column(c).AlignItems(ui.Center).Padding(px(t, 2), 0).Shrink(0).Children(func() {
		railButton(c, "rail-search", icons.Magnifer, i18n.T("Command palette"), i18n.T("Command palette")+" (⌘K)", false, "", actions.openPalette)
	})
	ui.Column(c).Grow(1).MinHeight(0).Children(func() {
		ui.Scroll(c).Grow(1).MinHeight(0).Children(func() {
			ui.Column(c).AlignItems(ui.Center).Padding(px(t, 2), 0).Children(func() {
				for gi, g := range state.Groups {
					if gi > 0 {
						ui.Box(c).Height(1).Width(px(t, 20)).Background(k.Line).Margin(px(t, 3), 0)
					}
					for _, p := range g.Pages {
						railItem(c, p, state, notices, actions, noticeActs, local)
					}
				}
			})
		})
		ui.Box(c).Absolute().Left(0).Right(0).Bottom(0).Height(px(t, 16)).Gradient(k.Sidebar.Alpha(0), k.Sidebar, 180).PassThrough()
	})
}

// railItem is a page's icon in the rail. Settings' has a dot while there
// are notifications: then it opens them, with Settings under them
// (shell_notices.go).
func railItem(c *ui.Context, p navPageViewState, state *navViewState, notices []noticeViewState, actions navActions, noticeActs noticeActions, local *navLocalUI) {
	ic := navIcon(p.Path)
	tip := p.Title
	if p.Keys != "" {
		tip += " (" + p.Keys + ")"
	}
	if p.Path != settingsPath {
		railButton(c, "rail"+p.Path, ic, p.Title, tip, p.Selected, p.Badge, func() { actions.navigate(p.Path) })
		return
	}
	tip, dot := tip+" (⌘,)", ""
	if state.NoticesCount > 0 {
		tip = state.NoticesLabel + " · " + tip
		dot = "notices"
	}
	b := railButton(c, "rail"+p.Path, ic, p.Title, tip, state.SettingsSelected || *local.noticesOpen, dot, func() {
		if state.NoticesCount > 0 {
			*local.noticesOpen = !*local.noticesOpen
			return
		}
		actions.navigate(settingsPath)
	})
	if state.NoticesCount == 0 {
		*local.noticesOpen = false
	}
	ui.Popover(c, b, local.noticesOpen, func() { noticePopover(c, notices, noticeActs) })
}

// railButton is a button of the rail: Bold on the pressed face while
// selected, a dot for a badge.
func railButton(c *ui.Context, key string, ic icons.Icon, label, tip string, selected bool, badge string, click func()) ui.Element {
	t := c.Theme()
	k := tok(t)
	b := ui.ButtonBase(c.Key(key)).Size(px(t, 36), px(t, 30)).Radius(px(t, radiusControl)).Center().Tooltip(tip).Label(label)
	switch {
	case selected:
		b.Background(k.Pressed)
	case b.Hovered():
		b.Background(k.Hover)
	}
	b.Children(func() {
		col := k.Text2
		if selected {
			col = k.Text
		}
		ui.Icon(c, ic.Get(selected)).Size(px(t, 18), px(t, 18)).TextColor(col)
		if badge != "" {
			col := k.Accent
			if badge == "notices" {
				col = k.Record // notifications: a red dot
			}
			ui.Box(c).Absolute().Top(px(t, 4)).Right(px(t, 5)).Size(px(t, 7), px(t, 7)).Radius(px(t, 4)).Background(col)
		}
	})
	if b.Clicked() {
		click()
	}
	return b
}

func sidebar(c *ui.Context, state *navViewState, notices []noticeViewState, actions navActions, noticeActs noticeActions, local *navLocalUI) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c).Height(titleBarHeight).Shrink(0).DragWindow().AlignItems(ui.Center).Justify(ui.End).Padding(0, px(t, sp2), 0, 0).Children(func() {
		if iconButton(c.Key("rail-on"), icons.SidebarMinimalistic, i18n.T("Show icons only")).Clicked() {
			actions.setRail(true)
		}
	})
	sidebarSearch(c, actions)
	// The sidebar's own theme: icons in the secondary grey, the chosen
	// page's in the text colour (MyGo draws a sidebar item's icon in the
	// accent, and the chosen one's in the accent's text colour).
	side := *t
	side.Accent, side.AccentText = k.Text2, k.Text
	side.SurfaceHover, side.SurfacePressed = k.G[2], k.G[3]
	chosen := side
	chosen.Accent = k.Text
	// Settings' item is the chosen one while its dialog shows, over the
	// page, and the page's again once it closes.
	if state.SettingsSelected {
		*local.nav = settingsPath
	} else if *local.nav == settingsPath {
		*local.nav = state.Current
	}
	// The list scrolls: a fade at its foot says there is more below.
	ui.Column(c).Grow(1).MinHeight(0).Children(func() {
		ui.Sidebar(c, local.nav, func() {
			for _, g := range state.Groups {
				items := func() {
					for _, p := range g.Pages {
						sel := p.Path == *local.nav
						if sel {
							c.SetTheme(&chosen)
						} else {
							c.SetTheme(&side)
						}
						item := ui.SidebarItem(c, p.Path, navIcon(p.Path).Get(sel), p.Title).Padding(px(t, 4), px(t, sp2))
						c.SetTheme(t)
						if sel {
							item.Background(k.Pressed).FontWeight(600)
						} else {
							item.TextColor(k.G[8])
						}
						if p.Badge != "" {
							item.Children(func() { ui.Badge(c, p.Badge) })
						}
					}
				}
				if g.ID == "" {
					items()
					continue
				}
				ui.SidebarSection(c.Key("nav-"+g.ID), g.Title, navSectionOpen(local, state, g.ID), items).Margin(px(t, 2), 0, 0, 0)
			}
		}).Grow(1).MinHeight(0).Background(ui.Transparent).Padding(0, px(t, 10), px(t, sp2)).Label(i18n.T("Sections")).OnChange(func() {
			actions.navigate(*local.nav)
		})
		ui.Box(c).Absolute().Left(0).Right(0).Bottom(0).Height(px(t, 20)).Gradient(k.Sidebar.Alpha(0), k.Sidebar, 180).PassThrough()
	})
	// The notifications, at the foot (shell_notices.go).
	if len(notices) > 0 {
		ui.Column(c).Shrink(0).Padding(px(t, sp1), px(t, 10), px(t, sp2)).Children(func() {
			noticeStack(c, notices, noticeActs)
		})
	}
}

// sidebarSearch is the sidebar's search field: it opens the palette.
func sidebarSearch(c *ui.Context, actions navActions) {
	t := c.Theme()
	k := tok(t)
	ui.Column(c).Shrink(0).Padding(0, px(t, 10), px(t, sp2)).Children(func() {
		b := ui.ButtonBase(c.Key("nav-search")).Padding(px(t, 6), px(t, sp2)).Gap(px(t, sp2)).Radius(px(t, radiusControl)).
			Background(k.Bg).Border(1, k.Line).Justify(ui.Start).Label(i18n.T("Command palette")).Tooltip(i18n.T("Command palette") + " (⌘K)")
		if b.Hovered() {
			b.Border(1, k.G[4])
		}
		b.Children(func() {
			ui.Icon(c, icons.Magnifer.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text3)
			ui.Text(c, i18n.T("Search")).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine().Grow(1).MinWidth(0)
			ui.Text(c, "⌘K").FontSize(px(t, fsMicro)).TextColor(k.Text3)
		})
		if b.Clicked() {
			actions.openPalette()
		}
	})
}
