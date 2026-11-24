package main

// The insights cards' window, over the main one: the card shown between
// two arrows, a dot a card under it, then Another color, Copy picture,
// Save picture and Share to LinkedIn. The arrow keys turn the cards too. A card comes in
// from the side it was asked from and then plays its entrance
// (insights_cards.go).

import (
	"strconv"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const (
	insightsDeckArrow  = 40
	insightsDeckGap    = 18
	insightsDeckPad    = 26
	insightsDeckW      = insightsCardW + 2*(insightsDeckArrow+insightsDeckGap) + 2*insightsDeckPad
	insightsDeckButton = 38
	insightsDeckH      = insightsDeckPad + insightsCardH + 18 + 8 + 20 + insightsDeckButton + insightsDeckPad
	insightsDeckR      = 26
	// insightsSlide is how long a card takes to come in.
	insightsSlide = 280 * time.Millisecond
)

// insightsDeckDialog is the cards' window.
func insightsDeckDialog(c *ui.Context, state *insightsDeckViewState, actions insightsActions) {
	open := true
	t := c.Theme()
	k := tok(t)
	s := shareFit(c, insightsDeckW, insightsDeckH)
	if state.Elapsed < insightsEntrance {
		c.After(time.Second / 60) // the entrance plays
	}
	ui.Modal(c.Key("insights-dialog"), &open, func() {
		ui.Box(c.Key("insights-fit")).Size(insightsDeckW*s, insightsDeckH*s).Children(func() {
			ui.Column(c.Key("insights-deck")).Absolute().Left(0).Top(0).Size(insightsDeckW, insightsDeckH).TransformOrigin(0, 0).Scale(s, s).
				Padding(insightsDeckPad).AlignItems(ui.Center).Children(func() {
				ui.Row(c.Key("stage")).Gap(insightsDeckGap).AlignItems(ui.Center).Children(func() {
					if insightsDeckArrowButton(c.Key("previous"), icons.AltArrowLeft, i18n.T("Previous card")).Clicked() || c.Shortcut(0, ui.KeyLeft) {
						actions.turnInsightsShare(-1)
					}
					insightsDeckStage(c, state)
					if insightsDeckArrowButton(c.Key("next"), icons.AltArrowRight, i18n.T("Next card")).Clicked() || c.Shortcut(0, ui.KeyRight) {
						actions.turnInsightsShare(1)
					}
				})
				if state.Posted != "" {
					insightsDeckPasteHint(c, state.Posted)
				} else {
					insightsDeckDots(c, state, actions)
				}
				insightsDeckButtons(c, state, actions)
			})
		})
	}).Padding(0).Background(k.Bg).Radius(insightsDeckR * s)
	if !open {
		actions.closeInsightsShare()
	}
}

// insightsDeckStage is the card where it stands: sliding in from its
// side, or growing a little as the window opens.
func insightsDeckStage(c *ui.Context, state *insightsDeckViewState) {
	in := insightsProgress(state.Elapsed, 0, insightsSlide)
	ui.Box(c.Key("insights-stage")).Size(insightsCardW, insightsCardH).Children(func() {
		// Keyed by the card, so the one before leaves at once.
		card := ui.Box(c.Key("card-"+strconv.Itoa(state.Card))).Absolute().Left(0).Top(0).Size(insightsCardW, insightsCardH).
			Radius(shareCardRadius).Shadow(0, 14, 36, -8, ui.RGBA(20, 16, 40, .28)).Opacity(in)
		if state.Slide == 0 {
			card.Scale(.96+.04*in, .96+.04*in)
		} else {
			card.Translate(float32(state.Slide)*36*(1-in), 0)
		}
		card.Children(func() { insightsCardFace(c, &state.Face, state.Elapsed, true) })
	})
}

// insightsDeckArrowButton is Previous or Next: round, growing a little
// under the pointer.
func insightsDeckArrowButton(c *ui.Context, ic icons.Icon, label string) ui.Element {
	t := c.Theme()
	k := tok(t)
	b := ui.ButtonBase(c).Size(insightsDeckArrow, insightsDeckArrow).Radius(insightsDeckArrow / 2).Center().Label(label).Tooltip(label)
	hot := float32(0)
	if b.Hovered() || b.FocusVisible() {
		hot = 1
		b.Background(k.Hover)
	}
	grow := 1 + .12*b.AnimateTo("hot", hot, ui.Snappy)
	return b.Scale(grow, grow).Children(func() { ui.Icon(c, ic.Linear).Size(20, 20).TextColor(k.Text) })
}

// insightsDeckDots are a dot a card, the shown one long; a click goes to
// its card.
func insightsDeckDots(c *ui.Context, state *insightsDeckViewState, actions insightsActions) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c.Key("dots")).Height(8).Gap(4).AlignItems(ui.Center).Margin(18, 0, 20, 0).Children(func() {
		for i := range state.Count {
			label := i18n.T("Card %d of %d", i+1, state.Count)
			b := ui.ButtonBase(c.Key(i)).Height(20).Padding(0, 3).Center().Label(label)
			width, color := float32(8), k.Text3.Alpha(.45)
			switch {
			case i == state.Card:
				width, color = 22, k.Text
			case b.Hovered():
				color = k.Text3
			}
			width = b.AnimateTo("w", width, ui.Snappy)
			if b.Children(func() { ui.Box(c).Size(width, 8).Radius(4).Background(color) }).Clicked() {
				actions.showInsightsShare(i)
			}
		}
	})
}

// insightsDeckButtons are Another color, then what to do with the card:
// Copy picture, Save picture, Share to LinkedIn and Share to X, an icon
// each.
func insightsDeckButtons(c *ui.Context, state *insightsDeckViewState, actions insightsActions) {
	t := c.Theme()
	ui.Row(c.Key("buttons")).Height(insightsDeckButton).Gap(10).AlignItems(ui.Center).Children(func() {
		if insightsDeckRoundButton(c.Key("shuffle"), icons.ShareShuffle, i18n.T("Another color"), false).Clicked() {
			actions.shuffleInsightsShare()
		}
		ui.Box(c.Key("rule")).Size(1, 18).Background(tok(t).Line).Margin(0, 4)
		ic, label := icons.ShareCopy, i18n.T("Copy picture")
		if state.Copied {
			c.After(400 * time.Millisecond) // the tick goes back to Copy
			ic, label = icons.ShareCopied, i18n.T("Copied")
		}
		if insightsDeckRoundButton(c.Key("copy"), ic, label, true).Clicked() {
			actions.copyInsightsShare()
		}
		if insightsDeckRoundButton(c.Key("save"), icons.DownloadMinimalistic, i18n.T("Save picture"), false).Clicked() {
			actions.saveInsightsShare()
		}
		if insightsDeckRoundButton(c.Key("linkedin"), icons.ShareLinkedIn, i18n.T("Share to LinkedIn"), false).Clicked() {
			actions.postInsightsShare("linkedin")
		}
		if insightsDeckRoundButton(c.Key("x"), icons.ShareX, i18n.T("Share to X"), false).Clicked() {
			actions.postInsightsShare("x")
		}
	})
}

// insightsDeckRoundButton is a round icon button under the card, growing
// a little under the pointer; the main one is filled with the ink.
func insightsDeckRoundButton(c *ui.Context, ic icons.Icon, label string, main bool) ui.Element {
	t := c.Theme()
	k := tok(t)
	b := ui.ButtonBase(c).Size(insightsDeckButton, insightsDeckButton).Radius(insightsDeckButton / 2).Center().Label(label).Tooltip(label)
	hot, ink := float32(0), k.Text
	if b.Hovered() || b.FocusVisible() {
		hot = 1
	}
	switch {
	case main:
		ink = k.Bg
		b.Background(k.Ink)
	case b.Pressed():
		b.Background(k.Pressed).Border(1, k.Line)
	case hot > 0:
		b.Background(k.Hover).Border(1, k.Line)
	default:
		b.Border(1, k.Line)
	}
	grow := 1 + .1*b.AnimateTo("hot", hot, ui.Snappy)
	return b.Scale(grow, grow).Children(func() { ui.Icon(c, ic.Linear).Size(18, 18).TextColor(ink) })
}

// insightsDeckPasteHint says, where the dots were, that the card is on
// the clipboard for site's post: neither site takes a picture from a
// link, so the user pastes it there.
func insightsDeckPasteHint(c *ui.Context, site string) {
	t := c.Theme()
	k := tok(t)
	c.After(500 * time.Millisecond) // the dots come back
	text := i18n.T("Picture copied. Press ⌘V in your LinkedIn post to paste it.")
	if site == "x" {
		text = i18n.T("Picture copied. Press ⌘V in your post on X to paste it.")
	}
	ui.Row(c.Key("paste-hint")).Height(26).Gap(6).AlignItems(ui.Center).Margin(9, 0, 11, 0).Padding(0, 12).Radius(13).
		Background(k.Hover).Children(func() {
		ui.Icon(c, icons.ShareCopied.Linear).Size(14, 14).TextColor(k.Text2)
		ui.Text(c, text).FontSize(12).FontWeight(500).TextColor(k.Text).SingleLine()
	})
}
