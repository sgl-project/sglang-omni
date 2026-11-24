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
	// Source text and earlier turns follow input[0] and cannot authorize a template.
	request := input[0]
	splitWords := func(text string) []string {
		return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
			return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_')
		})
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
	requestWords := splitWords(request)
	requestNamesFormat := hasWord(requestWords, "tag", "tags", "template", "templates") || strings.Contains(request, "标签") || strings.Contains(request, "模板")
	question := strings.ContainsAny(request, "?？")
	clauseRequest := strings.NewReplacer(" and ", ", ", " but ", ", ", " so ", ", ").Replace(strings.ToLower(request))
	clauses := strings.FieldsFunc(clauseRequest, func(r rune) bool {
		return r == ',' || r == '，' || r == '.' || r == '。' || r == ';' || r == '；' || r == '!' || r == '！' || r == '?' || r == '？' || r == '\n'
	})
	for _, clause := range clauses {
		clauseWords := splitWords(clause)
		nameIndex := -1
		for i, word := range clauseWords {
			if word == name {
				nameIndex = i
				break
			}
		}
		if nameIndex < 0 {
			continue
		}
		namesFormat := hasWord(clauseWords, "tag", "tags", "template", "templates") || strings.Contains(clause, "标签") || strings.Contains(clause, "模板")
		namesFormat = namesFormat || len(clauseWords) == 1 && requestNamesFormat
		namesIndefinitePrompt := nameIndex > 0 && (clauseWords[nameIndex-1] == "a" || clauseWords[nameIndex-1] == "an") && hasWord(clauseWords, "prompt", "prompts")
		nameAt := strings.Index(clause, name)
		namesTransformedPrompt := nameAt >= 0 && strings.Contains(clause[:nameAt], "改成") && strings.Contains(clause, "提示词")
		if !namesFormat && !namesIndefinitePrompt && !namesTransformedPrompt {
			continue
		}
		if nameAt < 0 {
			continue
		}
		prefix := clause[:nameAt]
		politeNegativeQuestion := question && (strings.Contains(prefix, "can't you") || strings.Contains(prefix, "couldn't you") || strings.Contains(prefix, "cannot you") || strings.Contains(prefix, "不") && strings.Contains(clause, "吗") && (strings.Contains(clause, "你") || strings.Contains(clause, "您")))
		negated := false
		if !politeNegativeQuestion {
			englishPrefix := strings.NewReplacer("not forget to", "", "n't forget to", "", "not just", "", "n't just", "").Replace(prefix)
			negated = hasWord(splitWords(englishPrefix), "not", "never", "no") || strings.Contains(englishPrefix, "n't")
			chinesePrefix := strings.NewReplacer("分别", "", "不错", "", "告别", "").Replace(prefix)
			for _, denial := range []string{"不要", "不用", "不需要", "不必", "无需", "无须", "没必要", "勿"} {
				negated = negated || strings.Contains(chinesePrefix, denial)
			}
			negated = negated || strings.Contains(chinesePrefix, "不") || strings.Contains(chinesePrefix, "别")
		}
		if negated || question && !politeNegativeQuestion && (hasWord(clauseWords, "use", "need", "want") || strings.Contains(clause, "用") || strings.Contains(clause, "需要") || strings.Contains(clause, "想要")) {
			continue
		}
		return true
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
