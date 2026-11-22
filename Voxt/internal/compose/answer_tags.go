package compose

import "strings"

// CleanRewrite is the clean for a Rewrite answer: it also drops echoed
// input tags, and any unclosed pair or tag fragment a stopped stream left.
func CleanRewrite(out string, inputs []string, partial bool, quoteSource ...string) string {
	return strings.TrimSpace(stripPartialInputTags(clean(out, inputs, partial, quoteSource...), inputs...))
}

// stripInputTags drops an <instruction>, <selected_text> or <typed_text>
// pair a model echoed from RewriteMessage, keeping the text between: a
// pair that wraps the whole answer, or one around exactly a text the user sent (whole
// lines of input). Other pairs are part of the answer (a prompt template the
// user asked for) and stay, as do the tags the user's own text has.
func stripInputTags(s string, input ...string) string {
	sent := "\n" + strings.Join(input, "\n") + "\n"
	for _, name := range []string{"instruction", "selected_text", "typed_text"} {
		open, close := "<"+name+">", "</"+name+">"
		if strings.Contains(sent, open) || strings.Contains(sent, close) {
			continue
		}
		for from := 0; ; {
			i := strings.Index(s[from:], open)
			if i < 0 {
				break
			}
			i += from
			j := strings.Index(s[i+len(open):], close)
			if j < 0 {
				break
			}
			j += i + len(open)
			inner := strings.TrimSpace(s[i+len(open) : j])
			whole := strings.TrimSpace(s[:i]) == "" && strings.TrimSpace(s[j+len(close):]) == ""
			if whole || inner != "" && strings.Contains(sent, "\n"+inner+"\n") {
				s = s[:i] + inner + s[j+len(close):]
				from = i + len(inner)
			} else {
				from = j + len(close)
			}
		}
	}
	return s
}

func stripPartialInputTags(s string, input ...string) string {
	s = stripInputTags(s, input...)
	sent := "\n" + strings.Join(input, "\n") + "\n"
	for _, name := range []string{"instruction", "selected_text", "typed_text"} {
		open, close := "<"+name+">", "</"+name+">"
		if strings.Contains(sent, open) || strings.Contains(sent, close) {
			continue
		}
		i := strings.LastIndex(s, open)
		if i >= 0 {
			rest := s[i+len(open):]
			literalMention := strings.HasPrefix(rest, " tag") || strings.HasPrefix(rest, " 标记") || strings.HasPrefix(rest, " literally")
			if i > strings.LastIndex(s, close) && !literalMention {
				s = s[:i] + rest
			}
		}
		// A lone "<" is the answer's own text, as in stripPartialControlMarkup.
		if n := partialTagSuffix(s, open); n > 1 {
			s = s[:len(s)-n]
		}
		if n := partialTagSuffix(s, close); n > 1 {
			s = s[:len(s)-n]
		}
	}
	return s
}
