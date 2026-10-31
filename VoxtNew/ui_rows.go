package main

// The rows and titles every settings section and settings-like page builds
// from (shell_settings.go): a setting's row with its ⓘ, a row with a note
// under it, and section titles (with a tip, with an add button).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
)

// settingRow is a setting: its name, a ⓘ telling what it does (desc, ""
// for none) and its control on the right (formRow in ui_layout.go).
func settingRow(c *ui.Context, key, title, desc string, control func()) {
	formRow(c, key, title, desc, control)
}

// settingRowNote is a setting whose description is a state to read at
// once (a path, what is missing, a version): it shows under the name.
func settingRowNote(c *ui.Context, key, title, note string, control func()) {
	settingRowNoteTip(c, key, title, "", note, control)
}

// settingRowNoteTip is settingRowNote with a ⓘ beside the name telling
// what the setting does (tip, "" for none).
func settingRowNoteTip(c *ui.Context, key, title, tip, note string, control func()) {
	t := c.Theme()
	ui.Row(c.Key(key)).Wrap().GapX(px(t, sp4)).GapY(px(t, 6)).AlignItems(ui.Center).MinHeight(px(t, 32)).Children(func() {
		ui.Column(c).Grow(1).Basis(px(t, 200)).MinWidth(px(t, 140)).Gap(px(t, 2)).Children(func() {
			if tip == "" {
				ui.Text(c, title)
			} else {
				ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Center).Children(func() {
					ui.Text(c, title).MinWidth(0)
					infoTip(c, tip)
				})
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
	t := c.Theme()
	sectionTitleText(c, title).Margin(px(t, sp2), 0, 0, 0)
}

func sectionTitleText(c *ui.Context, title string) ui.Element {
	t := c.Theme()
	return ui.Text(c, title).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(tok(t).Text2)
}

// sectionTitleTip is sectionTitle with a ⓘ beside it: the margin goes on
// the row, so the title and the ⓘ share one centre line.
func sectionTitleTip(c *ui.Context, title, tip string) {
	t := c.Theme()
	ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Center).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		sectionTitleText(c, title)
		infoTip(c, tip)
	})
}

// sectionTitleAdd is sectionTitleTip with a small Add button on the
// right of its row, for a card of items below it (shortcuts, apps), so
// the card holds only its items. It reports a click.
//
// The row is as wide as the card. As the title starts on the card's left
// edge, the button's label ends on its right edge: the button's right
// padding and hairline hang past it (its hover face shows in the page's
// margin). Its height past the ⓘ's 18 hangs out above and below, so the
// row stands as tall as sectionTitleTip's and the title does not move.
func sectionTitleAdd(c *ui.Context, key, title, tip, add string) bool {
	t := c.Theme()
	clicked := false
	ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Center).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		sectionTitleText(c, title).SingleLine().Shrink(0)
		infoTip(c, tip).Shrink(0)
		ui.Spacer(c)
		b := quietButton(c.Key(key), icons.Add, add).Margin(px(t, -3.5), -titleAddHang(t), px(t, -3.5), 0)
		clicked = b.Clicked()
	})
	return clicked
}

// titleAddHang is how far sectionTitleAdd's button hangs past the card's
// right edge: quietButton's right padding and its hairline.
func titleAddHang(t *ui.Theme) float32 { return px(t, 10) + 1 }
