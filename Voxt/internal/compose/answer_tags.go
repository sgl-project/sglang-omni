package compose

import "strings"

// CleanRewrite is the clean for a Rewrite answer: it also drops echoed
// input tags, and any unclosed pair or tag fragment a stopped stream left.
func CleanRewrite(out string, inputs []string, partial bool, quoteSource ...string) string {
	return strings.TrimSpace(stripPartialInputTags(clean(out, inputs, partial, quoteSource...), true, inputs...))
}

// stripInputTags drops an <instruction>, <selected_text> or <typed_text>
// pair a model echoed from RewriteMessage, keeping the text between: a pair
// that wraps the whole answer when the current request does not name a tag
// and ask for a template, or one around exactly a text the user sent (whole
// lines of input). Requested templates and tags in the user's own text stay.
// content is false for titles.
func stripInputTags(s string, content bool, input ...string) string {
	sent := "\n" + strings.Join(input, "\n") + "\n"
	for _, name := range []string{"instruction", "selected_text", "typed_text"} {
		requestedTemplate := content && requestsInputTemplate(input, name)
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

func requestsInputTemplate(input []string, name string) bool {
	if len(input) == 0 {
		return false
	}
	// A session repeats the current typed and spoken request at input[1], before earlier turns.
	request := input[0]
	if len(input) > 1 {
		request = input[1]
	}
	words := strings.FieldsFunc(strings.ToLower(request), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_')
	})
	named := false
	for _, word := range words {
		if word == name {
			named = true
			break
		}
	}
	if !named {
		return false
	}
	for _, action := range []string{"写", "生成", "创建", "输出", "返回", "给", "用", "改", "按", "做"} {
		actionAt := strings.Index(request, action)
		if actionAt < 0 {
			continue
		}
		rest := request[actionAt+len(action):]
		if strings.Contains(rest, "标签") || strings.Contains(rest, "模板") {
			return true
		}
		if strings.Contains(rest, "提示词") {
			promptAt := strings.Index(rest, "提示词")
			if action == "改" && strings.Contains(rest[:promptAt], "成") {
				return true
			}
			if action != "用" && action != "改" && action != "按" {
				return true
			}
		}
	}
	hasWord := func(parts []string, targets ...string) bool {
		for _, part := range parts {
			for _, target := range targets {
				if part == target {
					return true
				}
			}
		}
		return false
	}
	for _, word := range words {
		switch word {
		case "write", "create", "generate", "make", "produce", "use", "turn", "give", "show", "output", "return", "provide", "convert", "rewrite", "wrap", "need", "want":
			if hasWord(words, "tag", "tags", "template", "templates") {
				return true
			}
		}
	}
	// Bare prompt wording counts only for an indefinite request or an explicit transformation.
	for i, word := range words {
		if word != name || i == 0 || words[i-1] != "a" && words[i-1] != "an" {
			continue
		}
		for j := 0; j < i; j++ {
			if words[j] != "into" && words[j] != "as" && words[j] != "to" {
				continue
			}
			if hasWord(words[:j], "turn", "convert", "rewrite") {
				return true
			}
		}
		if hasWord(words[:i], "turn", "convert", "rewrite") {
			continue
		}
		if hasWord(words[:i], "write", "create", "generate", "make", "produce", "give", "show", "output", "return", "provide") {
			return true
		}
		if hasWord(words, "need", "want") && (!hasWord(words[i+1:], "to") || hasWord(words, "write", "create", "generate", "produce", "give", "show", "output", "return", "provide")) {
			return true
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
