package main

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
)

type (
	uiColor   = ui.Color
	uiElement = ui.Element
)

func transcriptWebView(context *ui.Context, view *mygo.WebView) ui.Element {
	return ui.WebView(context, view)
}

func transcriptResult(context *ui.Context, render transcriptRenderFunc, kind, key, source string, streaming bool, native ...func()) {
	theme := context.Theme()
	input := transcriptPreviewInput{
		key: key, source: source, size: theme.FontSize, rightPadding: px(theme, sp3-2),
		text: theme.Text, link: theme.Accent, background: sessionTextBackground(theme, kind), dark: theme.Dark, streaming: streaming,
	}
	if render != nil {
		if view := render(context, input, px(theme, 110), px(theme, 510)); view.Valid() {
			view.Margin(0, -input.rightPadding, 0, 0)
			return
		}
	}
	if len(native) > 0 {
		native[0]()
		return
	}
	ui.Text(context, source).Selectable()
}

func sessionTextCard(context *ui.Context, kind string) ui.Element {
	return fillCard(context).Background(sessionTextBackground(context.Theme(), kind))
}

// The native card and its embedded formatted page use the same fill.
func sessionTextBackground(theme *ui.Theme, kind string) ui.Color {
	if kind == "transcribe" {
		return tok(theme).TranscriptFace
	}
	return tok(theme).Subtle
}
