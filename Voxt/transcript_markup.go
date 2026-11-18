package main

import (
	"strings"

	"github.com/egoist/mygo/ui/markdown/parser"
)

// transcriptMarkdown returns markup only when formatted rendering adds
// something to ordinary text. An empty result keeps the native fallback.
func transcriptMarkdown(source string) string {
	return transcriptMarkup(source, false)
}

func transcriptMarkup(source string, streaming bool) string {
	doc := parser.New(parser.Default | parser.ExtDollarMath)
	doc.SetStreaming(streaming)
	doc.Set(source)
	source = inlineMathTail(doc, streaming)
	doc.Set(source)
	for index := doc.Len() - 1; index > 0; index-- {
		block := doc.Block(index)
		if streaming || block.Kind != parser.MathBlock || !block.Open {
			continue
		}
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
	for index := 1; index < doc.Len(); index++ {
		switch doc.Block(index).Kind {
		case parser.Paragraph:
			for _, inline := range doc.Inlines(index) {
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
		for index := 0; index+1 < len(text); index++ {
			if text[index] != '\\' {
				continue
			}
			if text[index+1] == '(' && !strings.Contains(source[node.Start+index+2:], `\)`) {
				if streaming {
					return source + `\)`
				}
				at := node.Start + index
				return source[:at] + `\` + source[at:]
			}
			index++
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
	for index := 1; index < doc.Len(); index++ {
		for _, node := range doc.Inlines(index) {
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
