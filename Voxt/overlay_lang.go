package main

// The session language/thinking menu's widget tree. Native window placement
// and provider/menu projection live in session_overlay_menu.go and its state file.

import (
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

func sessionLanguageMenuContent(state func() sessionMenuViewState, actions sessionOverlayActions, local *sessionOverlayLocalUI) func(*ui.Context) {
	return func(context *ui.Context) {
		sessionLanguageMenuView(context, state(), actions, local)
	}
}

func sessionLanguageMenuView(context *ui.Context, state sessionMenuViewState, actions sessionOverlayActions, local *sessionOverlayLocalUI) {
	context.SetTheme(overlayTheme())
	defer context.Root().Background(ui.Transparent)
	ui.Column(context).Fill().Padding(layout.Pad).Children(func() {
		ui.Column(context.Key("ov-lang-face")).Fill().Radius(ovLangRadius).Background(ui.RGBA(30, 30, 34, 0.98)).
			Border(1, ui.RGBA(255, 255, 255, 0.12)).Shadow(0, layout.ShadowY, layout.ShadowBlur, 0, ui.RGBA(0, 0, 0, layout.ShadowA)).
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
