package main

import (
	"html"
	"regexp"
	"strings"

	"github.com/egoist/mygo/ui/editor"
)

var transcriptCode = regexp.MustCompile(`(?s)(<pre><code class="language-([^"]+)">)(.*?)(</code></pre>)`)

// Use the same language registry and line tokenizers as MyGo's Markdown view.
// Unknown languages remain literal code. Escape each token before adding spans.
func highlightTranscriptCode(markup string) string {
	return transcriptCode.ReplaceAllStringFunc(markup, func(block string) string {
		parts := transcriptCode.FindStringSubmatch(block)
		language := editor.LanguageNamed(html.UnescapeString(parts[2]))
		if language == nil || language.Tokenizer == nil {
			return block
		}
		var out strings.Builder
		out.WriteString(parts[1])
		var state uint64
		var tokens []editor.Token
		for line := range strings.SplitAfterSeq(html.UnescapeString(parts[3]), "\n") {
			text := strings.TrimSuffix(line, "\n")
			tokens, state = language.Tokenizer.Tokenize(text, state, tokens[:0])
			start := 0
			for _, token := range tokens {
				end := min(max(token.End, start), len(text))
				value := html.EscapeString(text[start:end])
				if class := transcriptTokenClass(token.Kind); class != "" {
					out.WriteString(`<span class="token-` + class + `">` + value + `</span>`)
				} else {
					out.WriteString(value)
				}
				start = end
			}
			out.WriteString(html.EscapeString(text[start:]))
			if strings.HasSuffix(line, "\n") {
				out.WriteByte('\n')
			}
		}
		out.WriteString(parts[4])
		return out.String()
	})
}

func transcriptTokenClass(kind editor.TokenKind) string {
	switch kind {
	case editor.TokenKeyword, editor.TokenOperator, editor.TokenTag:
		return "keyword"
	case editor.TokenString, editor.TokenRegexp, editor.TokenInserted:
		return "string"
	case editor.TokenComment:
		return "comment"
	case editor.TokenNumber, editor.TokenConstant, editor.TokenEscape, editor.TokenAttribute:
		return "number"
	case editor.TokenFunction, editor.TokenType, editor.TokenMeta:
		return "function"
	case editor.TokenDeleted, editor.TokenInvalid:
		return "invalid"
	}
	return ""
}
