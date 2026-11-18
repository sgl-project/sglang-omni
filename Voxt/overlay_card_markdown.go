package main

// Markdown in the answer card (#4119): the language model answers in
// Markdown, and a Rewrite's answer shows as it reads — headings, lists,
// code, math's TeX — as the chat apps show replies, not as its source.
// A Translate's or Transcribe's text stays plain: it goes in verbatim.
// The view reports each rendered text's height as the measurement of its
// text's key, and the card's next sizing takes it; until then the plain
// text's measure stands in, as for every text of the face.

import (
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// mdReveal shows a streaming answer as it comes, without pacing or
// fades: the card paces it itself (overlay_stream.go).
var mdReveal = markdown.Reveal{Unit: markdown.AtOnce, Fade: -1}

// cardMarkdown renders a card block's Markdown: Stream while the model
// writes it (live: its half-written syntax heals), View once it is done.
func cardMarkdown(context *ui.Context, local *sessionOverlayLocalUI, text ovSized, color ui.Color, live bool, onMeasured func([]sessionOverlayMeasurement)) ui.Element {
	options := &local.mdCard
	if text.size == layout.BubbleTextSize {
		options = &local.mdBubble
	}
	options.Theme = &markdown.Theme{
		FontSize:            text.size,
		Text:                color,
		Muted:               ui.RGBA(255, 255, 255, 0.6),
		Link:                tok(context.Theme()).Accent,
		QuoteBar:            ui.RGBA(255, 255, 255, 0.24),
		QuoteText:           ui.RGBA(255, 255, 255, 0.6),
		Rule:                ovCardRule,
		CodeBackground:      ui.RGBA(255, 255, 255, 0.1),
		CodeBlockBackground: ui.RGBA(255, 255, 255, 0.06),
		CodeBlockBorder:     ui.RGBA(255, 255, 255, 0.1),
		TableBorder:         ui.RGBA(255, 255, 255, 0.12),
		TableHeader:         ui.RGBA(255, 255, 255, 0.06),
	}
	options.Reveal = &mdReveal
	var element ui.Element
	if live {
		element = markdown.Stream(context, text.text, options)
	} else {
		element = markdown.View(context, text.text, options)
	}
	if onMeasured == nil {
		return element
	}
	return element.Draw(func(_ *ui.Painter, rect ui.Rect) {
		onMeasured([]sessionOverlayMeasurement{{Key: text, Width: rect.W, Height: rect.H}})
	})
}
