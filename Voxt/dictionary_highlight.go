package main

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
)

func dictionaryText(c *ui.Context, text string, highlights []dict.Span) ui.Element {
	if len(highlights) == 0 {
		return ui.Text(c, text)
	}
	k := tok(c.Theme())
	spans := make([]ui.Span, 0, 2*len(highlights)+1)
	plain := func(from, to int) {
		if to > from {
			spans = append(spans, ui.Span{Text: text[from:to]})
		}
	}
	at := 0
	for _, highlight := range highlights {
		// A span belongs to the text it was made for; one from an older,
		// longer text must not reach outside this one.
		start := min(max(highlight.Start, at), len(text))
		end := min(max(highlight.End, start), len(text))
		plain(at, start)
		if end > start {
			spans = append(spans, ui.Span{
				Text: text[start:end], Background: k.AccentSoft, Radius: 3,
			})
		}
		at = end
	}
	plain(at, len(text))
	return ui.RichText(c, spans...).Label(text)
}
