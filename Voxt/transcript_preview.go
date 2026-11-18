package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown/parser"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/transcriptpreview"
)

type transcriptPreviewInput struct {
	key, source            string
	size, rightPadding     float32
	text, link, background ui.Color
	dark                   bool
	streaming              bool
}

// Preview state belongs to its host window's main thread.
type transcriptPreview struct {
	input                          transcriptPreviewInput
	view                           *mygo.WebView
	version                        uint64
	height                         int
	ready, loaded, running, closed bool
}

var transcriptUpdate = mygo.NewEvent[transcriptpreview.Update]("transcript-preview")

func (p *transcriptPreview) place(c *ui.Context, win *mygo.Window, in transcriptPreviewInput, fallback, limit float32) ui.Element {
	if win == nil || in.source == "" {
		return ui.Element{}
	}
	if p.input != in || p.view != nil && p.view.IsDestroyed() {
		old := p.input
		p.input = in
		p.version++
		if old.key != in.key || old.size != in.size || old.rightPadding != in.rightPadding || old.text != in.text || old.link != in.link || old.dark != in.dark || old.background != in.background || p.view != nil && p.view.IsDestroyed() {
			p.height, p.ready, p.loaded = 0, false, false
		}
		win.Update(func() { p.update(win) })
	}
	if !p.ready || p.view == nil || p.view.IsDestroyed() {
		return ui.Element{}
	}
	height := fallback
	if p.height > 0 {
		height = float32(p.height)
	}
	return ui.WebView(c, p.view).Height(min(height, limit)).MinWidth(0).Shrink(0)
}

// Only one parse runs per preview. Updates arriving meanwhile replace the
// pending input, so a fast stream cannot queue an unbounded number of jobs.
func (p *transcriptPreview) update(win *mygo.Window) {
	if p.running || p.closed || win.IsDestroyed() {
		return
	}
	in, version := p.input, p.version
	p.running = true
	go func() {
		markup := transcriptMarkup(in.source, in.streaming)
		win.Update(func() {
			p.running = false
			if p.closed || win.IsDestroyed() {
				return
			}
			if p.version != version {
				p.update(win)
				return
			}
			if markup == "" {
				p.ready = false
				return
			}
			if p.view == nil || p.view.IsDestroyed() {
				view, err := win.NewWebView(mygo.WebViewOptions{BackgroundColor: cssColor(in.background)})
				if err != nil {
					return
				}
				p.view, p.loaded = view, false
			}
			data := transcriptpreview.Update{HTML: markup, Version: version, Streaming: in.streaming}
			if !p.loaded {
				p.view.Page().LoadHTML(transcriptpreview.Page(data, in.size, in.rightPadding, cssColor(in.text), cssColor(in.link), cssColor(in.background), in.dark), "")
				p.loaded = true
			} else {
				_ = transcriptUpdate.EmitPage(p.view.Page(), data)
			}
			p.ready = true
		})
	}()
}

func cssColor(c ui.Color) string { return fmt.Sprintf("#%02x%02x%02x%02x", c.R, c.G, c.B, c.A) }

// Plain text that needs no formatting keeps its dictionary highlights.
func (a *app) transcriptResult(c *ui.Context, kind store.Kind, key, source string, highlights []dict.Span, streaming bool) {
	t := c.Theme()
	in := transcriptPreviewInput{key: key, source: source, size: t.FontSize, rightPadding: px(t, sp3-2), text: t.Text, link: t.Accent, background: sessionTextBackground(t, kind), dark: t.Dark, streaming: streaming}
	if view := a.mainTranscript.place(c, a.win, in, px(t, 110), px(t, 510)); view.Valid() {
		view.Margin(0, -in.rightPadding, 0, 0)
	} else {
		dictionaryText(c, source, highlights).Selectable()
	}
}

type transcriptPreviewBridge struct{ app *app }

func (b transcriptPreviewBridge) Resize(ctx context.Context, version uint64, height int) {
	page := mygo.CallerPage(ctx)
	if page == nil || page.WebView() == nil || height < 1 {
		return
	}
	host := page.Window()
	host.Update(func() {
		a := b.app
		previews := append([]*transcriptPreview{&a.mainTranscript, &a.overlayTranscript}, a.bubbleTranscripts...)
		for _, p := range previews {
			if p == nil || !p.ready || p.view != page.WebView() || p.version != version || p.height == height {
				continue
			}
			p.height = height
			if p != &a.mainTranscript {
				a.overlayRestyle()
			}
			break
		}
	})
}

// transcriptMarkdown returns the markup for a formatted transcript. An empty result
// keeps ordinary dictation in the native selectable text view.
func transcriptMarkdown(source string) string {
	return transcriptMarkup(source, false)
}

func transcriptMarkup(source string, streaming bool) string {
	doc := parser.New(parser.Default | parser.ExtDollarMath)
	doc.SetStreaming(streaming)
	doc.Set(source)
	source = inlineMathTail(doc, streaming)
	doc.Set(source)
	for i := doc.Len() - 1; i > 0; i-- {
		block := doc.Block(i)
		if streaming || block.Kind != parser.MathBlock || !block.Open {
			continue
		}
		// A missing closing delimiter must not consume the rest of a transcript.
		start := block.Start
		if strings.HasPrefix(source[start:], "$$") {
			source = source[:start] + `\$\$` + source[start+2:]
		} else if strings.HasPrefix(source[start:], `\[`) {
			source = source[:start] + `\\[` + source[start+2:]
		}
	}
	if source != doc.Source() {
		doc.Set(source)
	}
	if strings.Contains(source, "$") {
		doc.Set(numericInlineMath(doc))
	}
	if streaming {
		return highlightTranscriptCode(doc.HTML())
	}
	for i := 1; i < doc.Len(); i++ {
		switch doc.Block(i).Kind {
		case parser.Paragraph:
			for _, inline := range doc.Inlines(i) {
				switch inline.Kind {
				case parser.Text, parser.SoftBreak, parser.HTML:
				default:
					return highlightTranscriptCode(doc.HTML())
				}
			}
		case parser.HTMLBlock, parser.FootnoteDefinition:
		default:
			return highlightTranscriptCode(doc.HTML())
		}
	}
	return ""
}

// Complete a streaming inline formula, or preserve its opening delimiter
// when the answer ends before it closes. Code and existing math are excluded.
func inlineMathTail(doc *parser.Document, streaming bool) string {
	source := doc.Source()
	last := doc.Len() - 1
	if last < 1 || doc.Block(last).Kind != parser.Paragraph {
		return source
	}
	for _, node := range doc.Inlines(last) {
		if node.Kind != parser.Text {
			continue
		}
		text := source[min(node.Start, len(source)):min(node.End, len(source))]
		for i := 0; i+1 < len(text); i++ {
			if text[i] != '\\' {
				continue
			}
			if text[i+1] == '(' && !strings.Contains(source[node.Start+i+2:], `\)`) {
				if streaming {
					return source + `\)`
				}
				at := node.Start + i
				return source[:at] + `\` + source[at:]
			}
			i++ // a doubled backslash is literal
		}
	}
	return source
}

// The Markdown parser treats a leading digit after a dollar as currency.
// Accept a complete pair in ordinary text, outside code and existing math.
func numericInlineMath(doc *parser.Document) string {
	source := doc.Source()
	var normalized strings.Builder
	written := 0
	for i := 1; i < doc.Len(); i++ {
		for _, node := range doc.Inlines(i) {
			if node.Kind != parser.Text || node.Start < written {
				continue
			}
			fragment := source[node.Start:node.End]
			for open := 0; open+1 < len(fragment); open++ {
				if fragment[open] == '\\' {
					open++
					continue
				}
				if fragment[open] != '$' || fragment[open+1] < '0' || fragment[open+1] > '9' {
					continue
				}
				next := strings.IndexAny(fragment[open+1:], "$\r\n")
				if next < 0 {
					break
				}
				close := open + 1 + next
				tex := fragment[open+1 : close]
				escaped := (len(tex)-len(strings.TrimRight(tex, `\`)))%2 != 0
				if fragment[close] != '$' || escaped || strings.TrimRight(tex, " \t") != tex {
					continue
				}
				end := node.Start + close + 1
				if end < len(source) && (source[end] >= '0' && source[end] <= '9' || source[end] == '$') {
					continue
				}
				normalized.WriteString(source[written : node.Start+open])
				normalized.WriteString(`\(` + tex + `\)`)
				written = end
				open = close
			}
		}
	}
	if written == 0 {
		return source
	}
	normalized.WriteString(source[written:])
	return normalized.String()
}
