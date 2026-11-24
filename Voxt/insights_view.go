package main

// Home's Insights, above Activity: what Voxt has written for the user,
// the time it saved, their speed and the time they spoke, with Share,
// which opens the cards of them (insights_deck.go).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// homeInsightsSection is the four totals under their title and Share.
func homeInsightsSection(c *ui.Context, state *homeViewState, actions homeActions) {
	ins := state.Insights
	if !state.KeepHistory || ins == nil {
		return
	}
	t := c.Theme()
	k := tok(t)
	ui.Row(c.Key("insights-title")).Gap(px(t, sp2)).AlignItems(ui.Center).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		sectionTitleText(c, i18n.T("Looking back")).Grow(1)
		if insightsShareButton(c.Key("insights-share")).Clicked() {
			actions.openInsightsShare()
		}
	})
	ui.Row(c.Key("home-insights")).Wrap().Gap(px(t, sp3)).Children(func() {
		for i, stat := range ins.Stats {
			fillCard(c.Key(i)).Gap(px(t, 2)).Grow(1).Basis(px(t, 140)).MinWidth(px(t, 120)).Padding(px(t, sp3), px(t, sp4)).Children(func() {
				insightsParts(c, stat.Parts, px(t, fsTitle), px(t, fsCaption), k.Text, k.Text2)
				ui.Text(c, stat.Label).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0)
			})
		}
	})
}

// insightsShareButton is Share. Under the pointer it grows, and the
// logo's two stars come out at its corner.
func insightsShareButton(c *ui.Context) ui.Element {
	t := c.Theme()
	k := tok(t)
	b := ui.ButtonBase(c).Radius(px(t, 14)).Padding(px(t, 4), px(t, 11), px(t, 4), px(t, 9)).Gap(px(t, sp1)).AlignItems(ui.Center).
		Label(i18n.T("Share a look back")).Shrink(0)
	target, ink, mark := float32(0), k.Text2, k.Text3
	switch {
	case b.Pressed():
		target, ink, mark = 1, k.Text, k.Text
		b.Background(k.Pressed)
	case b.Hovered() || b.FocusVisible():
		target, ink, mark = 1, k.Text, k.Text
		b.Background(k.Hover)
	}
	hot := b.AnimateTo("hot", target, ui.Bouncy)
	shown := min(max(hot, 0), 1)
	return b.Scale(1+.16*hot, 1+.16*hot).Children(func() {
		ui.Icon(c, icons.Share.Linear).Size(px(t, 12), px(t, 12)).TextColor(mark)
		ui.Text(c, i18n.T("Share")).FontSize(px(t, fsCaption)).TextColor(ink).SingleLine()
		ui.Icon(c, icons.ShareStar).Absolute().Right(px(t, -3)).Top(px(t, -5)).Size(px(t, 10), px(t, 10)).TextColor(shareOrange).
			Opacity(shown).Scale(hot, hot)
		ui.Icon(c, icons.ShareStar).Absolute().Right(px(t, -9)).Top(px(t, 5)).Size(px(t, 6), px(t, 6)).TextColor(sharePeach).
			Opacity(shown).Scale(hot, hot)
	})
}

// insightsParts is a number and its unit, then the next: "2 hours 34
// minutes", the numbers large.
func insightsParts(c *ui.Context, parts []insightsPart, size, unitSize float32, ink, unitInk ui.Color) {
	ui.Row(c.Key("parts")).Gap(4).AlignItems(ui.End).Children(func() {
		for i, part := range parts {
			ui.Text(c, part.Value).FontSize(size).FontWeight(600).TextColor(ink).FontFeatures("tnum").SingleLine()
			margin := float32(10)
			if i == len(parts)-1 {
				margin = 0
			}
			ui.Text(c, part.Unit).FontSize(unitSize).FontWeight(500).TextColor(unitInk).SingleLine().Margin(0, margin, size*0.14, 0)
		}
	})
}
