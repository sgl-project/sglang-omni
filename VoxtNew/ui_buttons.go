package main

// The kit's buttons by rank (ui_kit.go): primary, record, secondary and
// ghost buttons, menus that drop down, icon buttons and toggles, the "⋯"
// menu, and the play buttons.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
)

// ---- Buttons by rank ----

// labelIcon is the size of an icon beside a button's label: no taller than
// the label's line (13 pt is a 15.3 DIP line on the Mac), so a button with
// an icon stands as tall as one without.
func labelIcon(t *ui.Theme) float32 { return t.FontSize * 1.15 }

// primaryButton is the view's one core action (Generate, Export, Start):
// MyGo's primary button in ink, near-black (near-white in the dark).
//
//	if primaryButton(c.Key("generate"), i18n.T("Generate")).Disabled(!ok).Clicked() { … }
func primaryButton(c *ui.Context, label string) ui.Element {
	t := c.Theme()
	c.SetTheme(inkTheme(t))
	b := ui.PrimaryButton(c, label).Border(1, ui.Color{}) // as tall as ui.Button, which has a hairline
	c.SetTheme(t)
	return b
}

// primaryIconButton is primaryButton with an icon before its label.
func primaryIconButton(c *ui.Context, ic icons.Icon, label string) ui.Element {
	t := c.Theme()
	c.SetTheme(inkTheme(t))
	b := ui.PrimaryButton(c, "").Border(1, ui.Color{}) // as tall as ui.Button
	b.Children(func() {
		ui.Icon(c, ic.Linear).Size(labelIcon(t), labelIcon(t))
		ui.Text(c, label).SingleLine()
	})
	c.SetTheme(t)
	return b
}

// recordButton is a record or destructive action (Record, Stop
// recording, Erase…): MyGo's primary button in the red-orange.
func recordButton(c *ui.Context, label string) ui.Element {
	t := c.Theme()
	k := tok(t)
	rec := *t
	rec.Accent, rec.AccentText = k.Record, ui.Hex("#ffffff")
	rec.AccentHover, rec.AccentPressed = k.Record.Mix(ui.Hex("#000000"), 0.1), k.Record.Mix(ui.Hex("#000000"), 0.2)
	c.SetTheme(&rec)
	b := ui.PrimaryButton(c, label).Border(1, ui.Color{}) // as tall as ui.Button
	c.SetTheme(t)
	return b
}

// secondaryIconButton is a secondary button (ui.Button's look) with an
// icon before its label.
func secondaryIconButton(c *ui.Context, ic icons.Icon, label string) ui.Element {
	t := c.Theme()
	b := ui.Button(c, "")
	b.Children(func() {
		ui.Icon(c, ic.Linear).Size(labelIcon(t), labelIcon(t))
		ui.Text(c, label).SingleLine()
	})
	return b
}

// ghostButton is a quiet action: no face until the pointer is on it.
func ghostButton(c *ui.Context, label string) ui.Element {
	t := c.Theme()
	b := ghostBase(c).Padding(t.Space(1.5), t.Space(2.5))
	b.Children(func() { ui.Text(c, label).SingleLine() })
	return b
}

// ghostIconButton is a ghost button with an icon before its label.
func ghostIconButton(c *ui.Context, ic icons.Icon, label string) ui.Element {
	return ghostIconButtonSized(c, ic, label, labelIcon(c.Theme()))
}

// ghostIconButtonSized is ghostIconButton with its icon at size: smaller
// for glyphs that fill their box (Play, Stop).
func ghostIconButtonSized(c *ui.Context, ic icons.Icon, label string, size float32) ui.Element {
	t := c.Theme()
	b := ghostBase(c).Padding(t.Space(1.5), t.Space(2.5)).Gap(t.Space(1.5))
	b.Children(func() {
		ui.Icon(c, ic.Linear).Size(size, size).TextColor(tok(t).Text2)
		ui.Text(c, label).SingleLine()
	})
	return b
}

// labelOrIcon is a ghost button with an icon and its label, or only the
// icon (with the label as its tooltip) when narrow.
func labelOrIcon(c *ui.Context, ic icons.Icon, label string, narrow bool) ui.Element {
	if narrow {
		return iconButton(c, ic, label)
	}
	return ghostIconButton(c, ic, label).Label(label)
}

// dropButton is a ghost button that opens a menu or a popover: its icon,
// its label and a chevron. Its label names it.
func dropButton(c *ui.Context, ic icons.Icon, label string) ui.Element {
	t := c.Theme()
	k := tok(t)
	b := ghostBase(c).Padding(t.Space(1.5), t.Space(2)).Gap(t.Space(1.5)).Label(label).MinWidth(0).Shrink(1)
	b.Children(func() {
		ui.Icon(c, ic.Linear).Size(labelIcon(t), labelIcon(t)).TextColor(k.Text2)
		ui.Text(c, label).SingleLine().Ellipsis("…").MinWidth(0)
		ui.Icon(c, icons.AltArrowDown.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.Text3)
	})
	return b
}

// dropText is a ghost button that opens a menu: its label and a
// chevron, on the grey fill while on (a filter is set).
func dropText(c *ui.Context, label string, on bool) ui.Element {
	t := c.Theme()
	k := tok(t)
	b := ghostBase(c).Padding(px(t, 4), px(t, 6), px(t, 4), px(t, 10)).Gap(px(t, sp1)).AlignItems(ui.Center).Label(label).MinWidth(0).Shrink(1)
	if on {
		b.Background(k.Pressed)
	}
	b.Children(func() {
		col := k.Text2
		if on {
			col = k.Text
		}
		ui.Text(c, label).FontSize(px(t, fsCaption)).TextColor(col).SingleLine().Ellipsis("…").MinWidth(0)
		ui.Icon(c, icons.AltArrowDown.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.Text3)
	})
	return b
}

// quietButton is an action in a row of dropText filters (Reset): the
// same size, type and colour as they are, so the row keeps its height
// when it appears.
func quietButton(c *ui.Context, ic icons.Icon, label string) ui.Element {
	t := c.Theme()
	k := tok(t)
	b := ghostBase(c).Padding(px(t, 4), px(t, 10), px(t, 4), px(t, 8)).Gap(px(t, sp1)).AlignItems(ui.Center).Label(label).Shrink(0)
	b.Children(func() {
		ui.Icon(c, ic.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.Text3)
		ui.Text(c, label).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine()
	})
	return b
}

// ghostBase is the face of ghost and icon buttons.
func ghostBase(c *ui.Context) ui.Element {
	t := c.Theme()
	k := tok(t)
	// A clear hairline: ghost buttons stand as tall as ui.Button (which
	// has a visible one) and line up beside it.
	b := ui.ButtonBase(c).Radius(px(t, radiusControl)).TextColor(k.Text).Border(1, ui.Color{})
	switch {
	case b.Pressed():
		b.Background(k.Pressed)
	case b.Hovered():
		b.Background(k.Hover)
	}
	return b
}

// iconButton is a square ghost button showing only an icon; tip names it
// (for the tooltip, assistive technology and tests).
//
//	if iconButton(c.Key("close"), icons.Close, i18n.T("Close")).Clicked() { … }
func iconButton(c *ui.Context, ic icons.Icon, tip string) ui.Element {
	t := c.Theme()
	b := ghostBase(c).Size(px(t, 28), px(t, 28)).Center().Label(tip).Tooltip(tip)
	b.Children(func() {
		col := tok(t).Text2
		if b.Hovered() {
			col = tok(t).Text
		}
		ui.Icon(c, ic.Linear).Size(px(t, 16), px(t, 16)).TextColor(col)
	})
	return b
}

// iconToggle is an icon button for something on or off (Star, the
// inspector): Bold and in the pressed face while on.
func iconToggle(c *ui.Context, ic icons.Icon, on bool, tip string) ui.Element {
	t := c.Theme()
	k := tok(t)
	b := ghostBase(c).Size(px(t, 28), px(t, 28)).Center().Label(tip).Tooltip(tip).Checked(on)
	if on {
		b.Background(k.Pressed)
	}
	b.Children(func() {
		col := k.Text2
		if on || b.Hovered() {
			col = k.Text
		}
		ui.Icon(c, ic.Get(on)).Size(px(t, 16), px(t, 16)).TextColor(col)
	})
	return b
}

// panelToggle is the header button that shows or hides a side panel
// (an inspector, a page's list): its icon (panel-left or
// panel-right, the side of the panel) on the pressed face while the
// panel shows.
func panelToggle(c *ui.Context, ic icons.Icon, shown bool, tip string) ui.Element {
	b := iconButton(c, ic, tip).Checked(shown)
	if shown {
		b.Background(tok(c.Theme()).Pressed)
	}
	return b
}

// moreMenu is the "⋯" overflow menu of a view, a card or a row: the
// secondary actions that do not need a button of their own.
//
//	moreMenu(c.Key("entry-more"), "", func(m *ui.Menu) {
//		if m.Item(i18n.T("Show in Finder")).Chosen() { … }
//	})
func moreMenu(c *ui.Context, tip string, build func(m *ui.Menu)) ui.Element {
	if tip == "" {
		tip = i18n.T("More actions")
	}
	return iconButton(c, icons.MenuDots, tip).Menu(build)
}

// playButton is the round ink play button of a recording: Pause
// (or Stop, with stop) while it plays. size is in DIPs at 100 %.
func playButton(c *ui.Context, playing, stop bool, size float32) ui.Element {
	t := c.Theme()
	k := tok(t)
	label, ic := i18n.T("Play"), icons.Play
	if playing {
		label, ic = i18n.T("Pause"), icons.Pause
		if stop {
			label, ic = i18n.T("Stop"), icons.Stop
		}
	}
	b := ui.ButtonBase(c).Size(px(t, size), px(t, size)).Radius(px(t, size/2)).Center().Label(label).Tooltip(label).Shrink(0)
	switch {
	case b.Pressed():
		b.Background(k.InkPressed)
	case b.Hovered():
		b.Background(k.InkHover)
	default:
		b.Background(k.Ink)
	}
	b.Children(func() {
		ui.Icon(c, ic.Bold).Size(px(t, size*0.44), px(t, size*0.44)).TextColor(k.OnInk)
	})
	return b
}

// playToggle is a small play / pause icon button: playTip while
// stopped, pauseTip while playing (the label follows).
func playToggle(c *ui.Context, playing bool, playTip, pauseTip string) ui.Element {
	if playing {
		return iconButton(c, icons.Pause, pauseTip)
	}
	return iconButton(c, icons.Play, playTip)
}
