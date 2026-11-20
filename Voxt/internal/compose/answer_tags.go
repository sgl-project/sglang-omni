package compose

import (
	"strings"
	"unicode"
)

// CleanRewrite is the clean for a Rewrite answer: it also drops echoed
// input tags, and any unclosed pair or tag fragment a stopped stream left.
func CleanRewrite(out string, inputs []string, partial bool, quoteSource ...string) string {
	return strings.TrimSpace(stripPartialInputTags(clean(out, inputs, partial, quoteSource...), true, inputs...))
}

// stripInputTags drops an <instruction>, <selected_text> or <typed_text>
// pair a model echoed from RewriteMessage, keeping the text between: a pair
// that wraps the whole answer when the user did not request a template, or
// one around exactly a text the user sent (whole lines of input). Requested
// templates and tags in the user's own text stay. content is false for titles.
func stripInputTags(s string, content bool, input ...string) string {
	sent := "\n" + strings.Join(input, "\n") + "\n"
	requestedTemplate := content && requestsInputTemplate(input)
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
			if whole && !requestedTemplate || inner != "" && strings.Contains(sent, "\n"+inner+"\n") {
				s = s[:i] + inner + s[j+len(close):]
				from = i + len(inner)
			} else {
				from = j + len(close)
			}
		}
	}
	return s
}

func requestsInputTemplate(input []string) bool {
	if len(input) == 0 {
		return false
	}
	// A session repeats the current typed and spoken request at input[1], before earlier turns.
	request := input[0]
	if len(input) > 1 {
		request = input[1]
	}
	for _, action := range []string{"写", "生成", "创建", "输出", "返回", "给", "用", "改", "按", "做"} {
		actionAt := strings.Index(request, action)
		if actionAt < 0 {
			continue
		}
		rest := request[actionAt+len(action):]
		for _, keyword := range []string{"标签", "模板", "提示词"} {
			if strings.Contains(rest, keyword) {
				return true
			}
		}
	}
	words := strings.FieldsFunc(strings.ToLower(request), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
	})
	requested := false
	for _, word := range words {
		switch word {
		case "write", "create", "generate", "make", "produce", "use", "turn", "give", "show", "output", "return", "provide", "convert", "rewrite", "wrap", "need", "want":
			requested = true
		case "tag", "tags", "template", "templates", "prompt", "prompts":
			if requested {
				return true
			}
		}
	}
	return false
}

func stripPartialInputTags(s string, content bool, input ...string) string {
	s = stripInputTags(s, content, input...)
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
