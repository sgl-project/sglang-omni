package main

// Home's Insights, above Activity: what Voxt has written for the user,
// the time it saved, their speed and the time they spoke, with Share,
// which opens a card of them (600×500 DIPs) to copy: those four, the
// active days and streaks beside the last thirteen weeks, and Voxt's
// lockup, on one of the share card's backgrounds.

import (
	"image"
	"strconv"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const (
	insightsCardW, insightsCardH = 600, 524
	// insightsFootH is the lockup's line, insightsFootBottom its distance
	// from the card's bottom: fixed, so it stays put whatever the numbers
	// and the background.
	insightsFootH, insightsFootBottom = 30, 26
)

type insightsCardViewState struct {
	Background int               `json:"background"`
	Dark       bool              `json:"dark"`
	Insights   insightsViewState `json:"insights"`
	Copied     bool              `json:"copied,omitempty"`
}

// renderInsightsCard is the card without its buttons, at 2×: what Copy
// copies.
func renderInsightsCard(state *insightsCardViewState) image.Image {
	return ui.Render(func(c *ui.Context) { insightsCardFace(c, state, nil, false) }, insightsCardW, insightsCardH, 2)
}

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
		if quietButton(c.Key("insights-share"), icons.Share, i18n.T("Share")).Label(i18n.T("Share a look back")).Clicked() {
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

// insightsCardDialog is the insights card over the main window.
func insightsCardDialog(c *ui.Context, state *insightsCardViewState, actions insightsActions) {
	open := true
	ui.Modal(c.Key("insights-dialog"), &open, func() { insightsCardFace(c, state, actions, true) }).
		Padding(0).Background(ui.RGBA(0, 0, 0, 0)).Radius(shareCardRadius)
	if !open {
		actions.closeInsightsShare()
	}
}

// insightsCardFace draws the card; chrome false leaves out its buttons
// (the copied picture).
func insightsCardFace(c *ui.Context, state *insightsCardViewState, actions insightsActions, chrome bool) {
	shareState := &shareCardViewState{Background: state.Background, Dark: state.Dark}
	a, _, _ := shareColors(shareState, false)
	ink, _, muted := shareInk(state.Dark)
	tile := func(key string) ui.Element {
		e := ui.Column(c.Key(key)).Radius(16).Padding(16, 20).Gap(4)
		if state.Dark {
			return e.Background(shareWhite.Alpha(.08)).Border(1, shareWhite.Alpha(.10))
		}
		return e.Background(shareWhite.Alpha(.9))
	}
	radius := float32(shareCardRadius)
	if !chrome {
		radius = 0 // the copied picture: square, as renderShareCard's
	}
	ui.Box(c.Key("insights-card")).Size(insightsCardW, insightsCardH).Radius(radius).Clip().Children(func() {
		shareBackdrop(c, state.Background, insightsCardW, insightsCardH)
		insightsCardBody(c, state, ink, muted, a, tile)
		if chrome {
			insightsCorner(c, state, actions)
		}
	})
}

// insightsCardBody is the card's title, tiles and lockup over its
// background.
func insightsCardBody(c *ui.Context, state *insightsCardViewState, ink, muted, a ui.Color, tile func(string) ui.Element) {
	ins := state.Insights
	ui.Column(c.Key("insights-body")).Absolute().Left(0).Top(0).Size(insightsCardW, insightsCardH).Padding(24).Gap(12).Children(func() {
		ui.Text(c, i18n.T("Me and Voxt, lately")).FontSize(30).FontWeight(700).TextColor(shareWhite).TextAlign(ui.Center).
			AlignSelf(ui.Stretch).Margin(4, 0, 6, 0)
		for row := 0; row < 2; row++ {
			ui.Row(c.Key("insights-row-" + strconv.Itoa(row))).Gap(12).Children(func() {
				for col := 0; col < 2; col++ {
					i := row*2 + col
					if i >= len(ins.Stats) {
						continue
					}
					tile("insights-tile-" + strconv.Itoa(i)).Grow(1).Basis(0).Children(func() {
						insightsParts(c, ins.Stats[i].Parts, 28, 14, ink, ink)
						ui.Text(c, ins.Stats[i].Label).FontSize(13).TextColor(muted).SingleLine()
					})
				}
			})
		}
		tile("insights-days").Children(func() {
			ui.Row(c.Key("insights-days-row")).Gap(20).AlignItems(ui.Center).Children(func() {
				ui.Column(c.Key("insights-days-col")).Gap(10).Shrink(0).Children(func() {
					for i, day := range ins.Days {
						ui.Column(c.Key(i)).Children(func() {
							insightsParts(c, day.Parts, 22, 13, ink, ink)
							ui.Text(c, day.Label).FontSize(12).TextColor(muted).SingleLine()
						})
					}
				})
				ui.Spacer(c)
				insightsWeeksGrid(c, &ins, a.Mix(shareWhite, 0), state.Dark)
			})
		})
	})
	// The lockup on one line: the mascot, the wordmark, "by SGL-omni".
	ui.Row(c.Key("insights-foot")).Absolute().Left(28).Top(insightsCardH - insightsFootBottom - insightsFootH).Height(insightsFootH).
		Gap(8).AlignItems(ui.Center).Children(func() {
		insightsMascotMark(c, 30)
		ui.Image(c, shareWordmark()).Size(24*443/173.0, 24)
		ui.Image(c, shareAttribution()).Size(15*777/118.0, 15).Margin(0, 0, 0, 2)
	})
}

// insightsCorner is the card's buttons, top left: another background,
// Copy.
func insightsCorner(c *ui.Context, state *insightsCardViewState, actions insightsActions) {
	// Side by side, clear of the tiles under them.
	ui.Row(c.Key("insights-corner")).Absolute().Left(14).Top(14).Gap(8).Children(func() {
		if shareCornerButton(c.Key("insights-shuffle"), icons.ShareShuffle, i18n.T("Another background")).Clicked() {
			actions.shuffleInsightsShare()
		}
		if shareCopyButton(c.Key("insights-copy"), state.Copied).Clicked() {
			actions.copyInsightsShare()
		}
	})

}

// insightsWeeksGrid is the last weeks as squares, a column a week,
// shaded in the card's first color by each day's results.
func insightsWeeksGrid(c *ui.Context, ins *insightsViewState, color ui.Color, dark bool) {
	const cell, gap = 12, 4
	empty := ui.RGBA(30, 27, 46, .06)
	if dark {
		empty = shareWhite.Alpha(.08)
		color = color.Mix(shareWhite, .45)
	}
	ui.Column(c.Key("insights-weeks")).Gap(6).Shrink(0).Children(func() {
		ui.Row(c.Key("insights-grid")).Gap(gap).Children(func() {
			for w, week := range ins.Weeks {
				ui.Column(c.Key(w)).Gap(gap).Children(func() {
					for d, level := range week {
						sq := ui.Box(c.Key(d)).Size(cell, cell).Radius(3)
						switch {
						case level < 0:
							sq.Background(ui.RGBA(0, 0, 0, 0))
						case level == 0:
							sq.Background(empty)
						default:
							sq.Background(color.Alpha(.25 + .75*float32(level)/4))
						}
					}
				})
			}
		})
		ui.Row(c.Key("insights-months")).Gap(gap).Children(func() {
			for w, month := range ins.Months {
				ui.Text(c.Key(w), month).FontSize(10).TextColor(ui.RGBA(138, 135, 154, 1)).Width(cell).NoWrap()
			}
		})
	})
}

// insightsMascotMark is the mascot as a mark h DIPs tall: the white
// microphone with its dark eyes and the logo's two stars, without a hat.
func insightsMascotMark(c *ui.Context, h float32) {
	u := h / 600
	ui.Box(c.Key("insights-mascot")).Size(650*u, h).Shrink(0).Children(func() {
		shareMicrophone(c, false, u)
		ui.Icon(c, icons.ShareStar).Absolute().Left(480*u).Top(6*u).Size(130*u, 130*u).TextColor(shareOrange)
		ui.Icon(c, icons.ShareStar).Absolute().Left(570*u).Top(136*u).Size(80*u, 80*u).TextColor(sharePeach)
	})
}
