package main

// The kit's text, marks and chips (ui_kit.go), and notes on the page
// (banner, notice).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// ---- Text, marks and chips ----

// infoTip is a ⓘ that tells text as a tooltip, in place of a line of
// help under a control. A click (or Space from the keyboard) keeps the
// tip open in a popover until a click elsewhere or Escape: people click
// an ⓘ rather than rest the pointer on it.
func infoTip(c *ui.Context, text string) ui.Element {
	t := c.Theme()
	k := tok(t)
	b := ui.ButtonBase(c).Size(px(t, 18), px(t, 18)).Center().Radius(px(t, 9)).Label(text)
	open := ui.Local(b, "open", func() bool { return false })
	if b.Clicked() {
		*open = !*open
	}
	if !*open { // the popover says it already
		b.Tooltip(text)
	}
	b.Children(func() {
		col := k.Text3
		if b.Hovered() || *open {
			col = k.Text2
		}
		ui.Icon(c, icons.InfoCircle.Linear).Size(px(t, 14), px(t, 14)).TextColor(col)
	})
	ui.Popover(c, b, open, func() {
		ui.Column(c).MaxWidth(px(t, 320)).Padding(px(t, sp1)).Children(func() {
			ui.Text(c, text).FontSize(px(t, fsCaption))
		})
	})
	return b
}

// tipText is a label that tells tip as a tooltip ("" for none), in place
// of a ⓘ or a line of help beside it.
func tipText(c *ui.Context, text, tip string) ui.Element {
	e := ui.Text(c, text).MinWidth(0)
	if tip != "" {
		e.Tooltip(tip)
	}
	return e
}

// truncText is a line of text that ends in "…" when it does not fit, and
// shows all of itself as a tooltip.
func truncText(c *ui.Context, s string) ui.Element {
	return ui.Text(c, s).SingleLine().Ellipsis("…").MinWidth(0).Tooltip(s)
}

// caption is a secondary line (12, the secondary grey).
func caption(c *ui.Context, s string) ui.Element {
	t := c.Theme()
	return ui.Text(c, s).FontSize(px(t, fsCaption)).TextColor(tok(t).Text2)
}

// sectionHeading is a heading inside a page or a dialog (15, semibold).
func sectionHeading(c *ui.Context, title string) ui.Element {
	t := c.Theme()
	return ui.Text(c, title).FontSize(px(t, fsSection)).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0)
}

// tagChip is a small grey label: a kind, a size, a tag.
func tagChip(c *ui.Context, text string) ui.Element {
	t := c.Theme()
	k := tok(t)
	return ui.Text(c, text).FontSize(px(t, fsMicro)).TextColor(k.Text2).SingleLine().Padding(px(t, 1), px(t, 6)).
		Radius(px(t, 4)).Background(k.Hover).Shrink(0)
}

// keycap is keys ("fn", "⌥ Space") as a key cap in a line of text:
// kbd's face at the text's size, for a shortcut the line tells to press.
func keycap(c *ui.Context, keys string) ui.Element {
	t := c.Theme()
	k := tok(t)
	return ui.Text(c, keys).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text).Padding(px(t, 1), px(t, 7)).
		Radius(px(t, 5)).Background(k.Subtle).Border(1, k.Line).SingleLine().Shrink(0)
}

// statusDot is a 6 px dot of a state's colour.
func statusDot(c *ui.Context, col ui.Color) ui.Element {
	t := c.Theme()
	return ui.Box(c).Size(px(t, 6), px(t, 6)).Radius(px(t, 3)).Background(col).Shrink(0)
}

// statusMark is a status icon (14) in col, then a short text (12) in col:
// "✓ Verified", "⚠ Not downloaded".
func statusMark(c *ui.Context, ic icons.Icon, col ui.Color, text string) ui.Element {
	t := c.Theme()
	return ui.Row(c).Gap(px(t, 4)).AlignItems(ui.Center).MinWidth(0).Children(func() {
		ui.Icon(c, ic.Linear).Size(px(t, 14), px(t, 14)).TextColor(col).Shrink(0)
		ui.Text(c, text).FontSize(px(t, fsCaption)).TextColor(col).SingleLine().Ellipsis("…").MinWidth(0)
	})
}

// iconMark is an icon (14) that says something as its tooltip and label:
// a rule that matched, a term that is off.
func iconMark(c *ui.Context, ic icons.Icon, col ui.Color, tip string) ui.Element {
	t := c.Theme()
	b := ui.Box(c).Size(px(t, 16), px(t, 16)).Center().Label(tip).Tooltip(tip).Shrink(0)
	b.Children(func() {
		ui.Icon(c, ic.Linear).Size(px(t, 14), px(t, 14)).TextColor(col)
	})
	return b
}

// ---- Notes on the page ----

// banner is a line of state the user should read now (a warning, a
// locked voice, a licence) on the subtle fill: an icon in col, the text,
// and trailing actions.
func banner(c *ui.Context, ic icons.Icon, col ui.Color, text string, trailing func()) ui.Element {
	t := c.Theme()
	k := tok(t)
	return ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Padding(px(t, sp2), px(t, sp3)).Radius(px(t, radiusControl)).
		Background(k.Subtle).Wrap().Children(func() {
		ui.Icon(c, ic.Linear).Size(px(t, 16), px(t, 16)).TextColor(col).Shrink(0)
		ui.Text(c, text).FontSize(px(t, fsCaption)).Grow(1).Basis(px(t, 240)).MinWidth(0)
		if trailing != nil {
			trailing()
		}
	})
}

// noticeKind is what a notice is about: it picks the icon and its colour.
type noticeKind int

const (
	noticeInfo noticeKind = iota
	noticeOK
	noticeWarn
	noticeError
)

// notice is a note on the page (a step's result, a warning): a
// banner with an ⓘ telling more ("" for none), the actions (buttons),
// and × when onClose is set.
func notice(c *ui.Context, key string, kind noticeKind, text, more string, actions, onClose func()) {
	k := tok(c.Theme())
	ic, col := icons.InfoCircle, k.Text2
	switch kind {
	case noticeOK:
		ic, col = icons.CheckCircle, k.Success
	case noticeWarn:
		ic, col = icons.DangerTriangle, k.Warning
	case noticeError:
		ic, col = icons.DangerCircle, k.Danger
	}
	banner(c.Key(key), ic, col, text, func() {
		if more != "" {
			infoTip(c.Key("more"), more)
		}
		if actions != nil {
			actions()
		}
		if onClose != nil && iconButton(c.Key("close"), icons.Close, i18n.T("Dismiss")).Clicked() {
			onClose()
		}
	}).Shrink(0)
}
