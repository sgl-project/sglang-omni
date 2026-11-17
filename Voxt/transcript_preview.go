package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown/parser"
	"github.com/sgl-project/sglang-omni/Voxt/internal/transcriptpreview"
)

type transcriptPreviewInput struct {
	key, source string
	size        float32
	text, link  ui.Color
	dark        bool
}

// One preview per window, on the main thread. The page reports its
// content height again when a resized window reflows it.
type transcriptPreview struct {
	input   transcriptPreviewInput
	view    *mygo.WebView
	version uint64
	height  int
	ready   bool
}

func (p *transcriptPreview) place(c *ui.Context, win *mygo.Window, in transcriptPreviewInput, fallback, limit float32) ui.Element {
	if win == nil || in.source == "" {
		return ui.Element{}
	}
	if p.input != in {
		p.input = in
		p.version++
		p.height, p.ready = 0, false
		version := p.version
		win.Update(func() {
			if p.version != version {
				return
			}
			markup := transcriptMarkdown(in.source)
			if markup == "" {
				return
			}
			if p.view == nil || p.view.IsDestroyed() {
				view, err := win.NewWebView(mygo.WebViewOptions{Transparent: true})
				if err != nil {
					return
				}
				p.view = view
			}
			p.ready = true
			p.view.Page().LoadHTML(transcriptpreview.Page(markup, version, in.size, cssColor(in.text), cssColor(in.link), in.dark), "")
		})
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

func cssColor(c ui.Color) string { return fmt.Sprintf("#%02x%02x%02x%02x", c.R, c.G, c.B, c.A) }

func (a *app) transcriptResult(c *ui.Context, key, source string) {
	t := c.Theme()
	in := transcriptPreviewInput{key: key, source: source, size: t.FontSize, text: t.Text, link: t.Accent, dark: t.Dark}
	if !a.mainTranscript.place(c, a.win, in, px(t, 110), px(t, 510)).Valid() {
		ui.Text(c, source).Selectable()
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
		for _, p := range []*transcriptPreview{&a.mainTranscript, &a.overlayTranscript} {
			if !p.ready || p.view != page.WebView() || p.version != version || p.height == height {
				continue
			}
			p.height = height
			if p == &a.overlayTranscript {
				a.overlayRestyle()
			}
			break
		}
	})
}

// transcriptMarkdown returns the markup for a formatted transcript. An empty result
// keeps ordinary dictation in the native selectable text view.
func transcriptMarkdown(source string) string {
	doc := parser.New(parser.Default | parser.ExtDollarMath)
	doc.Set(source)
	for i := doc.Len() - 1; i > 0; i-- {
		block := doc.Block(i)
		if block.Kind != parser.MathBlock || !block.Open {
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
	for i := 1; i < doc.Len(); i++ {
		switch doc.Block(i).Kind {
		case parser.Paragraph:
			for _, inline := range doc.Inlines(i) {
				switch inline.Kind {
				case parser.Text, parser.SoftBreak, parser.HTML:
				default:
					return doc.HTML()
				}
			}
		case parser.HTMLBlock, parser.FootnoteDefinition:
		default:
			return doc.HTML()
		}
	}
	return ""
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
