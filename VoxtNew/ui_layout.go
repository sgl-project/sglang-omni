package main

// The kit's layout (ui_kit.go): the page header, cards, collapsible
// sections, form rows, search and filter bars, empty states.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
)

// ---- Layout ----

// pageHeader is a page's title bar, which drags the window: its title,
// then on the right what trailing builds (the page's one primary action,
// icon buttons, moreMenu).
// headerInset is the room page titles leave on their left for the window
// controls: set each frame by sidebarFrame, non-zero beside the icon rail.
var headerInset float32

func pageHeader(c *ui.Context, title string, trailing func()) {
	t := c.Theme()
	ui.Row(c).Height(titleBarHeight).Shrink(0).AlignItems(ui.Center).Padding(0, px(t, sp4), 0, px(t, sp6)+headerInset).Gap(px(t, sp2)).DragWindow().Children(func() {
		ui.Text(c, title).FontSize(px(t, fsTitle)).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
		if trailing != nil {
			trailing()
		}
	})
}

// card is a flat card: the page's background inside a hairline border,
// radius 12. Give it children.
func card(c *ui.Context) ui.Element {
	t := c.Theme()
	return ui.Column(c).Padding(px(t, sp4)).Gap(px(t, sp3)).Radius(px(t, radiusCard)).Border(1, tok(t).Line)
}

// fillCard is a flat card on the subtle grey fill, without a border.
func fillCard(c *ui.Context) ui.Element {
	t := c.Theme()
	return ui.Column(c).Padding(px(t, sp4)).Gap(px(t, sp3)).Radius(px(t, radiusCard)).Background(tok(t).Subtle)
}

// collapsible is a section that folds: a header row (chevron, title, and
// what trailing adds on the right, as a summary) that opens and closes
// it, and fn below it while open.
func collapsible(c *ui.Context, title string, open *bool, trailing, fn func()) ui.Element {
	t := c.Theme()
	k := tok(t)
	root := ui.Column(c).Shrink(0)
	root.Children(func() {
		parts := ui.CollapsibleBase(c, open)
		tr := parts.Trigger.Gap(px(t, sp2)).AlignItems(ui.Center).Padding(px(t, 6), px(t, 4)).Radius(px(t, radiusControl)).Label(title)
		if tr.Hovered() {
			tr.Background(k.Hover)
		}
		tr.Children(func() {
			ui.Icon(c, icons.AltArrowRight.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text2).Rotate(90 * parts.Progress())
			ui.Text(c, title).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
			if trailing != nil {
				trailing()
			}
		})
		// Apply a click now, in this build pass: callers often hand in a
		// local copied from a map and store it back after this returns, so a
		// click applied after the frame would land in a dead local and the
		// section would never open or close.
		tr.Changed()
		if p := parts.Panel(fn); p.Valid() {
			p.Gap(px(t, sp3)).Padding(px(t, sp2), px(t, 4), px(t, 4))
		}
	})
	return root
}

// formRow is a setting: its name, a ⓘ with what it does (info, "" for
// none), and its control on the right.
func formRow(c *ui.Context, key, title, info string, control func()) {
	t := c.Theme()
	// The control goes under the name when both do not fit on a line.
	ui.Row(c.Key(key)).Wrap().GapX(px(t, sp4)).GapY(px(t, 6)).AlignItems(ui.Center).MinHeight(px(t, 32)).Children(func() {
		ui.Row(c).Grow(1).Basis(px(t, 160)).MinWidth(px(t, 120)).Gap(px(t, 6)).AlignItems(ui.Center).Children(func() {
			ui.Text(c, title).MinWidth(0)
			if info != "" {
				infoTip(c, info)
			}
		})
		ui.Row(c).Wrap().Gap(px(t, sp2)).AlignItems(ui.Center).Shrink(1).MinWidth(0).Children(control)
	})
}

// fieldInfo is a labelled control with a ⓘ beside its label.
func fieldInfo(c *ui.Context, label, info string, control func()) {
	t := c.Theme()
	ui.Column(c).Gap(px(t, 6)).Children(func() {
		ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Center).Children(func() {
			ui.Text(c, label).MinWidth(0)
			if info != "" {
				infoTip(c, info)
			}
		})
		control()
	})
}

// searchField is a text input with a magnifier before its text. It is
// as wide as its row allows, up to maxW (0: no limit); the input is
// returned for Changed, Label…
func searchField(c *ui.Context, key string, value *string, placeholder string, maxW float32) ui.Element {
	t := c.Theme()
	k := tok(t)
	var in ui.Element
	box := ui.Row(c.Key(key + "-box")).Grow(1).Shrink(1).Basis(0).MinWidth(px(t, 120)).AlignItems(ui.Stretch)
	if maxW > 0 {
		box.MaxWidth(maxW)
	}
	box.Children(func() {
		in = ui.TextInput(c.Key(key), value).Placeholder(placeholder).Label(placeholder).Grow(1).MinWidth(0).
			Padding(t.Space(1.5), t.Space(2.5), t.Space(1.5), px(t, 30))
		ui.Column(c).Absolute().Left(0).Top(0).Bottom(0).Width(px(t, 30)).Center().PassThrough().Children(func() {
			ui.Icon(c, icons.Magnifer.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text3)
		})
	})
	return in
}

// filterBar is the bar of a page's search field and its filters
// (segmented controls, selects, a selection button): on one line when
// the search keeps 240 DIPs beside the filters (filtersW wide, in DIPs at
// 100 %) in room (the bar's width), else the search on a line of its own
// and the filters under it. It returns whether it took two lines.
func filterBar(c *ui.Context, room, filtersW float32, search, filters func()) (twoLines bool) {
	t := c.Theme()
	if room-px(t, filtersW+sp2) >= px(t, 240) {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Shrink(0).Children(func() {
			search()
			filters()
		})
		return false
	}
	ui.Column(c).Gap(px(t, sp2)).Shrink(0).Children(func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(search)
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(filters)
	})
	return true
}

// emptyState is a page's or a list's empty state: a large icon, a title,
// a line of text and actions (one primary at most).
func emptyState(c *ui.Context, ic icons.Icon, title, body string, actions func()) ui.Element {
	t := c.Theme()
	k := tok(t)
	return ui.Column(c).Grow(1).Center().Gap(px(t, sp2)).Padding(px(t, sp6)).Children(func() {
		ui.Icon(c, ic.Linear).Size(px(t, 36), px(t, 36)).TextColor(k.Text3)
		ui.Text(c, title).FontSize(px(t, fsSection)).FontWeight(600).TextAlign(ui.Center)
		if body != "" {
			ui.Text(c, body).TextColor(k.Text2).TextAlign(ui.Center).MaxWidth(px(t, 420))
		}
		if actions != nil {
			ui.Row(c).Gap(px(t, sp2)).Wrap().Justify(ui.Center).Margin(px(t, sp2), 0, 0).Children(actions)
		}
	})
}
