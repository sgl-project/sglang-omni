package main

// The notifications (the notification centre's, internal/notify), shown
// as a stack of cards at the foot of the sidebar:
// the newest whole, up to two more peeking under it; × dismisses one and
// the next comes up. The icon rail shows a dot on its Settings icon (the
// System section's), which then opens the same stack in a popover. There is no status bar: what it
// showed (the services Transcribe uses, the helper) becomes a card while
// something is wrong (setProblem), with the action that fixes it.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// noticeLayers is how many cards show under the newest, peeking.
const noticeLayers = 2

// noticeStack is the cards, newest on top, the next ones peeking under it,
// with a count and Clear all over them when there are more than one.
func noticeStack(c *ui.Context, notices []noticeViewState, actions noticeActions) {
	t := c.Theme()
	k := tok(t)
	if len(notices) == 0 {
		return
	}
	layers := min(len(notices)-1, noticeLayers)
	step, inset := px(t, 6), px(t, 8)
	ui.Column(c.Key("notices")).Gap(px(t, sp1)).Children(func() {
		if len(notices) > 1 {
			ui.Row(c).AlignItems(ui.Center).Padding(0, px(t, 2)).Children(func() {
				ui.Text(c, i18n.N(len(notices), "%d notification", "%d notifications", len(notices))).
					FontSize(px(t, fsMicro)).TextColor(k.Text3).SingleLine().Grow(1).MinWidth(0)
				b := ui.ButtonBase(c.Key("notices-clear")).Padding(px(t, 1), px(t, 4)).Radius(px(t, 4)).Label(i18n.T("Clear all"))
				if b.Hovered() {
					b.Background(k.Hover)
				}
				b.Children(func() {
					ui.Text(c, i18n.T("Clear all")).FontSize(px(t, fsMicro)).TextColor(k.Text2).SingleLine()
				})
				if b.Clicked() {
					actions.clearNotices()
				}
			})
		}
		ui.Column(c).Children(func() {
			noticeCard(c, notices[0], actions)
			// The cards under it: their foot shows, a step lower and
			// narrower each (the bottom of a rounded card, clipped).
			r := px(t, radiusControl+2)
			for i := 1; i <= layers; i++ {
				f := float32(i)
				ui.Box(c).Height(step).Margin(0, inset*f).Clip().Shrink(0).Children(func() {
					ui.Box(c).Absolute().Left(0).Right(0).Bottom(0).Height(step+2*r).
						Radius(r).Background(k.Bg).Border(1, k.Line).Opacity(1 - 0.15*f)
				})
			}
		})
	})
}

// noticeCard is a notification: its severity's icon, title, one or two
// lines, the action, and × at the top right.
func noticeCard(c *ui.Context, n noticeViewState, actions noticeActions) {
	t := c.Theme()
	k := tok(t)
	ic, col := icons.InfoCircle, k.Text2
	switch n.Severity {
	case noticeLevelError:
		ic, col = icons.DangerCircle, k.Danger
	case noticeLevelWarning:
		ic, col = icons.DangerTriangle, k.Warning
	case noticeLevelSuccess:
		ic, col = icons.CheckCircle, k.Success
	}
	card := ui.Column(c.Key("notice-"+n.ID)).Gap(px(t, 2)).Padding(px(t, 6), px(t, sp1), px(t, sp2), px(t, sp2)).
		Radius(px(t, radiusControl+2)).Background(k.Bg).Border(1, k.Line).Shadow(0, 2, 8, 0, ui.RGBA(0, 0, 0, 0.06)).
		Tooltip(n.TimeHint)
	card.Children(func() {
		ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Start).Children(func() {
			ui.Icon(c, ic.Linear).Size(px(t, 14), px(t, 14)).TextColor(col).Shrink(0).Margin(px(t, 3), 0, 0, 0)
			ui.Text(c, n.Title).FontSize(px(t, fsCaption)).FontWeight(600).MaxLines(2).Ellipsis("…").Grow(1).MinWidth(0).Margin(px(t, 2), 0, 0, 0)
			b := ghostBase(c.Key("notice-close")).Size(px(t, 20), px(t, 20)).Center().Shrink(0).Label(i18n.T("Dismiss")).Tooltip(i18n.T("Dismiss"))
			b.Children(func() {
				ui.Icon(c, icons.Close.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.Text3)
			})
			if b.Clicked() {
				actions.dismissNotice(n.ID)
			}
		})
		// Under the title, from its start.
		ui.Column(c).Gap(px(t, 2)).Padding(0, px(t, sp1), 0, px(t, 20)).Children(func() {
			if n.Body != "" {
				ui.Text(c, n.Body).FontSize(px(t, fsMicro)).TextColor(k.Text2).MaxLines(3).Ellipsis("…")
			}
			if n.ActionLabel != "" {
				b := ui.ButtonBase(c.Key("notice-act")).Padding(px(t, 2), 0).Label(n.ActionLabel).AlignSelf(ui.Start).MinWidth(0)
				b.Children(func() {
					txt := ui.Text(c, n.ActionLabel).FontSize(px(t, fsMicro)).FontWeight(500).TextColor(k.AccentInk).SingleLine().Ellipsis("…").MinWidth(0)
					if b.Hovered() {
						txt.Underline()
					}
				})
				if b.Clicked() {
					actions.noticeCardAction(n.ID)
				}
			}
		})
	})
}

// noticePopover is the rail's: the same stack, then Settings, which the
// rail's icon opens when there is no notification.
func noticePopover(c *ui.Context, notices []noticeViewState, actions noticeActions) {
	t := c.Theme()
	ui.Column(c).Width(px(t, 260)).Gap(px(t, sp2)).Padding(px(t, sp1)).Children(func() {
		noticeStack(c, notices, actions)
		ui.Divider(c)
		b := ghostBase(c.Key("notices-settings")).Padding(px(t, 6), px(t, sp2)).Gap(px(t, sp2)).Justify(ui.Start).Label(i18n.T("Settings"))
		b.Children(func() {
			ui.Icon(c, icons.Settings.Linear).Size(px(t, 16), px(t, 16)).TextColor(tok(t).Text2)
			ui.Text(c, i18n.T("Settings")).SingleLine().Grow(1)
			ui.Text(c, "⌘,").FontSize(px(t, fsMicro)).TextColor(tok(t).Text3)
		})
		if b.Clicked() {
			actions.openMainSettings()
		}
	})
}
