package main

// The session language/thinking menu's widget tree. Native window placement
// and provider/menu projection live in session_overlay_menu.go and its state file.

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

const (
	ovLangW      = 200
	ovLangRowH   = 30
	ovLangPad    = 6
	ovLangMaxH   = 336
	ovLangGap    = 6
	ovLangRadius = 12
)

func sessionLanguageMenuHeight(rows int) int {
	return min(rows*ovLangRowH+2*ovLangPad, ovLangMaxH)
}

func sessionLanguageMenuFrame(card mygo.Rectangle, top bool, rows int) mygo.Rectangle {
	faceX, faceY := card.X+layout.Pad, card.Y+layout.Pad
	// Insert, Copy, the size and Close right of the capsule.
	capsuleRight := faceX + card.Width - 2*layout.Pad - layout.CardPadX - layout.CardTools(4) - layout.CardHeadGap
	height := sessionLanguageMenuHeight(rows)
	x := capsuleRight - ovLangW
	y := faceY - ovLangGap - height
	if top {
		y = faceY + layout.CardPadTop + layout.CardHeadH + ovLangGap
	}
	return mygo.Rectangle{X: x - layout.Pad, Y: y - layout.Pad, Width: ovLangW + 2*layout.Pad, Height: height + 2*layout.Pad}
}

func thinkMenuFrame(language mygo.Rectangle, top bool, rows int) mygo.Rectangle {
	height := min(rows*ovLangRowH+2*ovLangPad, ovLangMaxH) + 2*layout.Pad
	if !top {
		language.Y += language.Height - height
	}
	language.Height = height
	return language
}

func pillMenuFrame(pill mygo.Rectangle, top bool, rows int) mygo.Rectangle {
	height := min(rows*ovLangRowH+2*ovLangPad, ovLangMaxH)
	x := pill.X + pill.Width - layout.Pad - ovLangW
	y := pill.Y + layout.Pad - ovLangGap - height
	if top {
		y = pill.Y + pill.Height - layout.Pad + ovLangGap
	}
	return mygo.Rectangle{X: x - layout.Pad, Y: y - layout.Pad, Width: ovLangW + 2*layout.Pad, Height: height + 2*layout.Pad}
}

func sessionLanguageMenuContent(state func() sessionMenuViewState, palette func() string, actions sessionOverlayActions, local *sessionOverlayLocalUI) func(*ui.Context) {
	return func(context *ui.Context) {
		sessionLanguageMenuView(context, state(), palette(), actions, local)
	}
}

func sessionLanguageMenuView(context *ui.Context, state sessionMenuViewState, palette string, actions sessionOverlayActions, local *sessionOverlayLocalUI) {
	context.SetTheme(overlayTheme(palette))
	defer context.Root().Background(ui.Transparent)
	k := tok(context.Theme())
	ui.Column(context).Fill().Padding(layout.Pad).Children(func() {
		ui.Column(context.Key("ov-lang-face")).Fill().Radius(ovLangRadius).Background(k.OverlayFace.Mix(ui.RGB(255, 255, 255), 0.035).Alpha(0.98)).
			Border(1, k.OverlayLine).Shadow(0, layout.ShadowY, layout.ShadowBlur, 0, ui.RGBA(0, 0, 0, layout.ShadowA)).
			Label(state.Label).Children(func() {
			ui.Scroll(context.Key("ov-lang-scroll")).Fill().TrackScroll(&local.langScroll).Children(func() {
				ui.Column(context).Padding(ovLangPad - layout.Border).Children(func() {
					if state.Kind != sessionMenuLanguage {
						thinkMenuView(context, state.Options, actions)
						return
					}
					for _, option := range state.Options {
						if menuRow(context.Key("ov-lang-"+option.Language), option.Label, option.Selected) {
							actions.chooseLanguage(option.Language)
						}
					}
				})
			})
		})
	})
}

func menuRow(context *ui.Context, name string, chosen bool) bool {
	button := ui.ButtonBase(context).Height(ovLangRowH).Padding(0, 10).Radius(8).Label(name).Checked(chosen)
	if button.Hovered() {
		button.Background(ui.RGBA(255, 255, 255, 0.08))
	}
	return button.Children(func() {
		ui.Row(context).Fill().Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Text(context, name).FontSize(13).TextColor(ui.RGBA(255, 255, 255, 0.92)).SingleLine().Ellipsis("…").Grow(1).MinWidth(0)
			if chosen {
				ui.Icon(context, icons.Check.Linear).Size(14, 14).TextColor(tok(context.Theme()).Accent).Shrink(0)
			}
		})
	}).Clicked()
}
