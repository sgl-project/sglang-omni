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
		positiveIntent, negatedIntent := inputTemplateIntent(input, name)
		requestedTemplate := content && positiveIntent
		open, close := "<"+name+">", "</"+name+">"
		currentHasLiteral := len(input) > 0 && (strings.Contains(input[0], open) || strings.Contains(input[0], close))
		requestHasLiteral := len(input) > 1 && (strings.Contains(input[1], open) || strings.Contains(input[1], close))
		if currentHasLiteral && (!requestHasLiteral || !negatedIntent) {
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

func inputTemplateIntent(input []string, name string) (bool, bool) {
	if len(input) == 0 {
		return false, false
	}
	// A session puts the full current input first and its request second. The
	// suffix check keeps an earlier turn at input[1] from becoming a request.
	request := input[0]
	if len(input) > 1 && strings.HasSuffix(input[0], input[1]) {
		request = input[1]
	}
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
	clauseRequest := strings.NewReplacer(" and ", ", ", " but ", ", ", " so ", ", ", "?", "?\n", "？", "？\n").Replace(strings.ToLower(request))
	clauses := strings.FieldsFunc(clauseRequest, func(r rune) bool {
		return r == ',' || r == '，' || r == '.' || r == '。' || r == ';' || r == '；' || r == '!' || r == '！' || r == '\n'
	})
	negatedMention := false
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
		namesFormat := hasWord(clauseWords, "tag", "tags", "template", "templates") || strings.Contains(clause, "标签") || strings.Contains(clause, "模板") || strings.Contains(clause, "<"+name+">") || strings.Contains(clause, "</"+name+">")
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
		suffix := clause[nameAt+len(name):]
		question := strings.ContainsAny(clause, "?？")
		positiveQuestion := question && (strings.Contains(prefix, "can't you") || strings.Contains(prefix, "couldn't you") || strings.Contains(prefix, "cannot you") || strings.Contains(clause, "能不能") || strings.Contains(prefix, "不") && strings.Contains(clause, "吗") && (strings.Contains(clause, "你") || strings.Contains(clause, "您")))
		chineseDenials := []string{"不要", "不用", "不需要", "不必", "无需", "无须", "没必要", "勿"}
		negated := strings.Contains(suffix, " is not ") || strings.Contains(suffix, " isn't ") || strings.Contains(suffix, " is no ")
		for _, denial := range chineseDenials {
			negated = negated || strings.Contains(suffix, "模板"+denial)
		}
		if !positiveQuestion {
			englishPrefix := strings.NewReplacer("not forget to", "", "n't forget to", "", "not hesitate to", "", "n't hesitate to", "", "not just", "", "n't just", "").Replace(prefix)
			negated = negated || hasWord(splitWords(englishPrefix), "not", "never", "no") || strings.Contains(englishPrefix, "n't")
			chinesePrefix := strings.NewReplacer("分别", "", "不错", "", "告别", "", "特别", "").Replace(prefix)
			for _, denial := range chineseDenials {
				negated = negated || strings.Contains(chinesePrefix, denial)
			}
			negated = negated || strings.Contains(chinesePrefix, "不") || strings.Contains(chinesePrefix, "别")
		}
		secondPerson := hasWord(clauseWords, "you") || strings.Contains(clause, "你") || strings.Contains(clause, "您")
		questionIsNotARequest := question && !positiveQuestion && (!secondPerson || hasWord(clauseWords, "use", "need", "want") || strings.Contains(clause, "用") || strings.Contains(clause, "需要") || strings.Contains(clause, "想要"))
		if negated || questionIsNotARequest {
			negatedMention = true
			continue
		}
		return true, false
	}
	return false, negatedMention
}

func stripPartialInputTags(s string, content bool, input ...string) string {
	s = stripInputTags(s, content, input...)
	for _, name := range []string{"instruction", "selected_text", "typed_text"} {
		open, close := "<"+name+">", "</"+name+">"
		positiveIntent, negatedIntent := inputTemplateIntent(input, name)
		requestedTemplate := content && positiveIntent
		currentHasLiteral := len(input) > 0 && (strings.Contains(input[0], open) || strings.Contains(input[0], close))
		requestHasLiteral := len(input) > 1 && (strings.Contains(input[1], open) || strings.Contains(input[1], close))
		if requestedTemplate || currentHasLiteral && (!requestHasLiteral || !negatedIntent) {
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
