package main

// Markdown in the answer card (#4119): the language model answers in
// Markdown, and a Rewrite's answer shows as it reads — headings, lists,
// code, math's TeX — as the chat apps show replies, not as its source.
// A Translate's or Transcribe's text stays plain: it goes in verbatim.
// The view reports each rendered text's height (noteMarkdownHeight) and
// the card's next sizing takes it (blockHeight); until then the plain
// text's measure stands in, as for every text of the face.

import (
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// mdKey keys a rendered Markdown text's height by what sets it: the
// text, its wrap and its size.
type mdKey struct {
	text string
	wrap float32
	size float32
}

// mdReveal shows a streaming answer as it comes, without pacing or
// fades: the card paces it itself (overlay_stream.go).
var mdReveal = markdown.Reveal{Unit: markdown.AtOnce, Fade: -1}

// cardMarkdown renders a card block's Markdown: Stream while the model
// writes it (its half-written syntax heals), View once it is done.
func (a *app) cardMarkdown(c *ui.Context, b cardBlock) ui.Element {
	o := &a.ov.mdCard
	if b.text.size == layout.BubbleTextSize {
		o = &a.ov.mdBubble
	}
	color := b.color
	if color == (ui.Color{}) {
		color = ui.RGBA(255, 255, 255, 0.92) // a bubble's answer text
	}
	o.Theme = &markdown.Theme{
		FontSize:            b.text.size,
		Text:                color,
		Muted:               ui.RGBA(255, 255, 255, 0.6),
		Link:                tok(c.Theme()).Accent,
		QuoteBar:            ui.RGBA(255, 255, 255, 0.24),
		QuoteText:           ui.RGBA(255, 255, 255, 0.6),
		Rule:                ovCardRule,
		CodeBackground:      ui.RGBA(255, 255, 255, 0.1),
		CodeBlockBackground: ui.RGBA(255, 255, 255, 0.06),
		CodeBlockBorder:     ui.RGBA(255, 255, 255, 0.1),
		TableBorder:         ui.RGBA(255, 255, 255, 0.12),
		TableHeader:         ui.RGBA(255, 255, 255, 0.06),
	}
	o.Reveal = &mdReveal
	var el ui.Element
	if b.stream {
		el = markdown.Stream(c, b.text.text, o)
	} else {
		el = markdown.View(c, b.text.text, o)
	}
	if b.live {
		el.Opacity(0.91) // a bubble's live text: 0.84 of 0.92
	}
	key := mdKey{b.text.text, b.text.wrap, b.text.size}
	return el.Draw(func(_ *ui.Painter, r ui.Rect) {
		h := r.H
		a.post(func() { a.noteMarkdownHeight(key, h) })
	})
}

// noteMarkdownHeight keeps a Markdown text's laid-out height: the next
// sizing of the face takes it (blockHeight), as overlayMeasure's text
// measures.
func (a *app) noteMarkdownHeight(k mdKey, h float32) {
	if a.ov.mdH == nil || len(a.ov.mdH) > 256 {
		a.ov.mdH = map[mdKey]float32{}
	}
	if a.ov.mdH[k] == h {
		return
	}
	a.ov.mdH[k] = h
	a.overlayRestyle()
}
