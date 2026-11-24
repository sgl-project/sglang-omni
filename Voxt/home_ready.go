package main

// Home's readiness list (home_page.go): what is missing before the first
// session (permissions, models, shortcuts), shown until all of it is
// ready.

import (
	"strconv"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// homeReadiness is what is still missing, a numbered line each (the whole
// line fixes it; the next one's number is ink), and below it, quieter, what
// is ready. Something broken has no number. Nothing when all is ready: Home
// is then the day's numbers and the recent results.
func homeReadiness(c *ui.Context, state *homeViewState, actions homeActions) {
	t := c.Theme()
	k := tok(t)
	if len(state.Missing) == 0 {
		return
	}
	sectionTitle(c, i18n.N(len(state.Missing), "%d step to go", "%d steps to go", len(state.Missing)))
	card(c.Key("home-missing")).Padding(px(t, sp1)).Gap(0).Children(func() {
		step := 0
		for _, readiness := range state.Missing {
			number := 0
			if readiness.Kind != "error" {
				step++
				number = step
			}
			row := ui.ButtonBase(c.Key(readiness.Key)).Padding(px(t, sp2), px(t, sp3)).Radius(px(t, radiusControl)).
				Gap(px(t, sp3)).AlignItems(ui.Center).Label(i18n.T("%s: %s", readiness.Title, readiness.Action)).Tooltip(readiness.Action).Disabled(readiness.Fix == "")
			if row.Hovered() && readiness.Fix != "" {
				row.Background(k.Hover)
			}
			row.Children(func() {
				if number == 0 {
					ui.Icon(c, icons.DangerCircle.Linear).Size(px(t, 20), px(t, 20)).TextColor(k.Danger).Shrink(0)
				} else {
					mark, digit := ui.Row(c).Size(px(t, 20), px(t, 20)).Center().Radius(px(t, 10)).Shrink(0), k.Text2
					if number == 1 {
						mark.Background(k.Ink)
						digit = k.OnInk
					} else {
						mark.Border(1, k.G[5])
					}
					mark.Children(func() {
						ui.Text(c, strconv.Itoa(number)).FontSize(px(t, fsMicro)).FontWeight(600).TextColor(digit).SingleLine()
					})
				}
				ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
					ui.Text(c, readiness.Title).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0)
					ui.Text(c, readiness.Text).FontSize(px(t, fsCaption)).TextColor(k.Text2).MinWidth(0)
				})
				if readiness.Fix != "" {
					ui.Row(c).Gap(px(t, 2)).AlignItems(ui.Center).Shrink(0).Children(func() {
						ui.Text(c, readiness.Action).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine()
						ui.Icon(c, icons.AltArrowRight.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text3)
					})
				}
			})
			if row.Clicked() && readiness.Fix != "" {
				actions.fixHomeReadiness(readiness.Fix)
			}
		}
	})
	if len(state.Ready) == 0 {
		return
	}
	// What is ready, quieter: a mark and a name each.
	ui.Row(c.Key("home-ready")).Wrap().GapX(px(t, sp4)).GapY(px(t, sp1)).Padding(0, px(t, sp1)).Children(func() {
		for _, readiness := range state.Ready {
			icon, color := icons.CheckCircle, k.Success
			if readiness.Warn {
				icon, color = icons.DangerTriangle, k.Warning
			}
			ui.Row(c.Key(readiness.Key)).Gap(px(t, 6)).AlignItems(ui.Center).MinWidth(0).Tooltip(readiness.Text).Children(func() {
				ui.Icon(c, icon.Linear).Size(px(t, 14), px(t, 14)).TextColor(color).Shrink(0)
				ui.Text(c, readiness.Title).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine()
				ui.Text(c, readiness.Text).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0).MaxWidth(px(t, 220))
			})
		}
	})
}
