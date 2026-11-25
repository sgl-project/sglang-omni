package main

// Home's Insights, above Activity: what Voxt has written for the user,
// the time it saved, their speed and the time they spoke, with Share,
// which opens the cards of them (insights_deck.go).

import (
	"strconv"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// homeInsightsSection is the four totals under their title and Share.
// The title folds the totals away and opens them again
// (settings.Appearance.LookingBackClosed keeps the choice); its arrow
// shows only under the pointer or with the keyboard's focus, as
// Appearance's Color row and the sidebar's sections do. Share stays
// on the title's row either way: the card it opens has the totals, and
// the row does not move as the section folds.
func homeInsightsSection(c *ui.Context, state *homeViewState, actions homeActions) {
	ins := state.Insights
	if !state.KeepHistory || ins == nil {
		return
	}
	t := c.Theme()
	k := tok(t)
	open := !state.InsightsClosed
	var parts ui.CollapsibleParts
	ui.Row(c.Key("insights-title")).Gap(px(t, sp2)).AlignItems(ui.Center).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		title := i18n.T("Looking back")
		parts = ui.CollapsibleBase(c.Key("insights-fold"), &open)
		fold := parts.Trigger.Gap(px(t, sp1)).AlignItems(ui.Center).Radius(px(t, radiusControl)).Label(title)
		fold.Children(func() {
			sectionTitleText(c, title).SingleLine()
			arrow := ui.Icon(c, icons.AltArrowRight.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.Text2).Rotate(90 * parts.Progress())
			if !fold.Hovered() && !fold.FocusVisible() {
				arrow.Opacity(0) // its room stays, so the title does not move
			}
		})
		if fold.Changed() {
			actions.setInsightsClosed(!state.InsightsClosed)
		}
		ui.Spacer(c)
		if shareSparkleButton(c.Key("insights-share"), i18n.T("Share a look back"), i18n.T("Share"), false) {
			actions.openInsightsShare()
		}
	})
	parts.Panel(func() {
		// The totals in pairs: one row of four while their numbers and
		// units fit, two rows of two below that, and one card a row only
		// where a pair does not fit. Cards share a row equally unless one
		// needs more for its number, unit or label, which are never cut.
		gap := px(t, sp3)
		ui.Row(c.Key("home-insights")).Wrap().Gap(gap).Children(func() {
			for first := 0; first < len(ins.Stats); first += 2 {
				pair := ins.Stats[first:min(first+2, len(ins.Stats))]
				need := gap * float32(len(pair)-1)
				for _, stat := range pair {
					label, _ := c.MeasureText(0, ui.Span{Text: stat.Label, Size: px(t, fsCaption)})
					need += max(insightsPartsWidth(c, stat.Parts, px(t, fsTitle), px(t, fsCaption)), label) + 2*px(t, sp4)
				}
				// The pair asks for the room of both cards side by side, so
				// the pairs part before a pair does; it is never wider than
				// the page, where its own row wraps.
				ui.Column(c.Key("pair-" + strconv.Itoa(first))).Grow(1).MaxWidthPercent(100).Children(func() {
					ui.Box(c).Width(need + 1).Height(0)
					ui.Row(c.Key("cards")).Wrap().Gap(gap).Children(func() {
						for i, stat := range pair {
							fillCard(c.Key(i)).Gap(px(t, 2)).Grow(1).Padding(px(t, sp3), px(t, sp4)).Children(func() {
								insightsParts(c, stat.Parts, px(t, fsTitle), px(t, fsCaption), k.Text, k.Text2)
								ui.Text(c, stat.Label).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().NoWrap().AlignSelf(ui.Start)
							})
						}
					})
				})
			}
		})
	})
}

const (
	insightsPartsGap = 4  // between a number, its unit and the next number
	insightsUnitGap  = 10 // more after a unit that another number follows
)

// insightsParts is a number and its unit, then the next: "2 hours 34
// minutes", the numbers large. They are never cut (NoWrap): what holds
// them is at least as wide as they are.
func insightsParts(c *ui.Context, parts []insightsPart, size, unitSize float32, ink, unitInk ui.Color) {
	ui.Row(c.Key("parts")).Gap(insightsPartsGap).AlignItems(ui.End).Children(func() {
		for i, part := range parts {
			ui.Text(c, part.Value).FontSize(size).FontWeight(600).TextColor(ink).FontFeatures("tnum").SingleLine().NoWrap()
			margin := float32(insightsUnitGap)
			if i == len(parts)-1 {
				margin = 0
			}
			ui.Text(c, part.Unit).FontSize(unitSize).FontWeight(500).TextColor(unitInk).SingleLine().NoWrap().Margin(0, margin, size*0.14, 0)
		}
	})
}

// insightsPartsWidth is the width insightsParts takes for parts.
func insightsPartsWidth(c *ui.Context, parts []insightsPart, size, unitSize float32) float32 {
	if len(parts) == 0 {
		return 0
	}
	width := float32(insightsPartsGap*(2*len(parts)-1) + insightsUnitGap*(len(parts)-1))
	for _, part := range parts {
		value, _ := c.MeasureText(0, ui.Span{Text: part.Value, Size: size, Weight: 600, Features: "tnum"})
		unit, _ := c.MeasureText(0, ui.Span{Text: part.Unit, Size: unitSize, Weight: 500})
		width += value + unit
	}
	return width
}
