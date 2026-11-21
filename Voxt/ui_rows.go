package main

// The rows and titles every settings section and settings-like page builds
// from (shell_settings.go): a setting's row with help on its title, a row
// with a note under it, and section titles (with a tip, with an add button).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// settingRow is a setting: its name tells what it does as a tooltip
// (desc, "" for none) and its control sits on the right.
func settingRow(c *ui.Context, key, title, desc string, control func()) {
	if !settingsSearchShows(title, desc) {
		return
	}
	formRow(c, key, title, desc, control)
}

// settingRowUnder is settingRow with its control under the name, where
// formRow puts it when the two do not fit on a line.
func settingRowUnder(c *ui.Context, key, title, desc string, control func()) {
	if !settingsSearchShows(title, desc) {
		return
	}
	t := c.Theme()
	ui.Column(c.Key(key)).Gap(px(t, 6)).MinHeight(px(t, 32)).Children(func() {
		name := settingsName(c, title).MinWidth(0)
		if desc != "" {
			name.Focusable().Tooltip(desc)
		}
		ui.Row(c).Wrap().Gap(px(t, sp2)).AlignItems(ui.Center).Children(control)
	})
}

// settingRowNote is a setting whose description is a state to read at
// once (a path, what is missing, a version): it shows under the name.
func settingRowNote(c *ui.Context, key, title, note string, control func()) {
	settingRowNoteTip(c, key, title, "", note, control)
}

// settingRowNoteTip is settingRowNote with help on the name (tip, ""
// for none).
func settingRowNoteTip(c *ui.Context, key, title, tip, note string, control func()) {
	if !settingsSearchShows(title, tip, note) {
		return
	}
	t := c.Theme()
	ui.Row(c.Key(key)).Wrap().GapX(px(t, sp4)).GapY(px(t, 6)).AlignItems(ui.Center).MinHeight(px(t, 32)).Children(func() {
		ui.Column(c).Grow(1).Basis(px(t, 200)).MinWidth(px(t, 140)).Gap(px(t, 2)).Children(func() {
			name := settingsName(c, title).MinWidth(0)
			if tip != "" {
				name.Focusable().Tooltip(tip)
			}
			if note != "" {
				caption(c, note).Selectable()
			}
		})
		ui.Row(c).Wrap().Gap(px(t, sp2)).AlignItems(ui.Center).Shrink(1).MinWidth(0).Children(control)
	})
}

// sectionTitle heads a group of settings.
func sectionTitle(c *ui.Context, title string) {
	if !settingsSearchShows(title) {
		return
	}
	t := c.Theme()
	sectionTitleText(c, title).Margin(px(t, sp2), 0, 0, 0)
}

func sectionTitleText(c *ui.Context, title string) ui.Element {
	t := c.Theme()
	return settingsName(c, title).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(tok(t).Text2)
}

// sectionTitleTip is sectionTitle that tells tip as a tooltip.
func sectionTitleTip(c *ui.Context, title, tip string) {
	t := c.Theme()
	name := sectionTitleText(c, title)
	if tip != "" {
		name.Focusable().Tooltip(tip)
	}
	name.Margin(px(t, sp2), 0, 0, 0)
}

// sectionTitleAdd is sectionTitleTip with a small Add button on the
// right of its row, for a card of items below it (shortcuts, apps), so
// the card holds only its items (tip "" for none). It reports a click.
//
// The row is as wide as the card. As the title starts on the card's left
// edge, the button's label ends on its right edge: the button's right
// padding and hairline hang past it (its hover face shows in the page's
// margin). Its extra height hangs out above and below, so the title does
// not move.
func sectionTitleAdd(c *ui.Context, key, title, tip, add string) bool {
	t := c.Theme()
	clicked := false
	ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Center).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		st := sectionTitleText(c, title).SingleLine().Shrink(0)
		if tip != "" {
			st.Focusable().Tooltip(tip)
		}
		ui.Spacer(c)
		b := quietButton(c.Key(key), icons.Add, add).Margin(px(t, -3.5), -titleAddHang(t), px(t, -3.5), 0)
		clicked = b.Clicked()
	})
	return clicked
}

// titleAddHang is how far sectionTitleAdd's button hangs past the card's
// right edge: quietButton's right padding and its hairline.
func titleAddHang(t *ui.Theme) float32 { return px(t, 10) + 1 }
