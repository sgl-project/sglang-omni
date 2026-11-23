package main

// The transcript preview's view half: the result's WebView once its markup
// is ready (transcript_preview_controller.go builds it), and the markup.

import (
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown/parser"
)

// uiColor names ui.Color for the preview's logic file, which does not
// import MyGo's UI (shell.go's aliases do the same).
type uiColor = ui.Color

func (p *transcriptPreview) place(c *ui.Context, win *mygo.Window, in transcriptPreviewInput, fallback, limit float32) ui.Element {
	if win == nil || in.source == "" {
		return ui.Element{}
	}
	p.request(win, in)
	if !p.ready || p.view == nil || p.view.IsDestroyed() {
		return ui.Element{}
	}
	height := fallback
	if p.height > 0 {
		height = float32(p.height)
	}
	return ui.WebView(c, p.view).Height(min(height, limit)).MinWidth(0).Shrink(0)
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
