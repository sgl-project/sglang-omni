package compose

import (
	"strings"
	"unicode"
)

// CleanRewrite is the clean for a Rewrite answer: it also drops echoed
// input tags, and any unclosed pair or tag fragment a stopped stream left.
func CleanRewrite(out string, inputs []string, partial bool, quoteSource ...string) string {
	return CleanRewriteWithRequest(out, inputs, requestFromInput(inputs), partial, quoteSource...)
}

// CleanRewriteWithRequest is CleanRewrite with the current request kept
// separate from the current input and earlier turns.
func CleanRewriteWithRequest(out string, inputs []string, request string, partial bool, quoteSource ...string) string {
	return strings.TrimSpace(stripPartialInputTags(clean(out, inputs, partial, quoteSource...), true, request, inputs...))
}

// stripInputTags drops an <instruction>, <selected_text> or <typed_text>
// pair a model echoed from RewriteMessage, keeping the text between: a pair
// that wraps the whole answer when the current request does not name a tag
// and ask for a template, or one around exactly a text the user sent (whole
// lines of input). Requested templates and tags in the user's own text stay.
// content is false for titles.
func stripInputTags(s string, content bool, request string, input ...string) string {
	sent := "\n" + strings.Join(input, "\n") + "\n"
	for _, name := range []string{"instruction", "selected_text", "typed_text"} {
		open, close := "<"+name+">", "</"+name+">"
		currentHasLiteral := len(input) > 0 && (strings.Contains(input[0], open) || strings.Contains(input[0], close))
		requestHasLiteral := strings.Contains(request, open) || strings.Contains(request, close)
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
			positiveIntent, negatedIntent := inputTemplateIntent(request, name)
			requestedTemplate := content && positiveIntent
			keepLiteral := content && currentHasLiteral && (!requestHasLiteral || !negatedIntent)
			if (whole && !requestedTemplate || inner != "" && strings.Contains(sent, "\n"+inner+"\n")) && !keepLiteral {
				s = s[:i] + inner + s[j+len(close):]
				from = i + len(inner)
			} else {
				from = j + len(close)
			}
		}
	}
	return s
}

func requestFromInput(input []string) string {
	if len(input) == 0 {
		return ""
	}
	request := input[0]
	if len(input) > 1 && strings.HasSuffix(input[0], input[1]) {
		request = input[1]
	}
	return request
}

// inputTemplateIntent uses tag shape and direct clause scope rather than
// intent inferred from the answer or an expanding list of explanation verbs.
func inputTemplateIntent(request, name string) (bool, bool) {
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
	clauseRequest := strings.NewReplacer(" and ", ", ", " but ", ", ", " so ", ", ", "?", "?\n", "？", "？\n").Replace(strings.ToLower(request))
	clauses := strings.FieldsFunc(clauseRequest, func(r rune) bool {
		return r == ',' || r == '，' || r == '.' || r == '。' || r == ';' || r == '；' || r == '!' || r == '！' || r == '\n'
	})
	positiveMention, negatedMention := false, false
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
		nameAt := strings.Index(clause, name)
		literalTag := strings.Contains(clause, "<"+name+">") || strings.Contains(clause, "</"+name+">")
		adjacentFormat := nameIndex+1 < len(clauseWords) && hasWord(clauseWords[nameIndex+1:nameIndex+2], "tag", "tags", "template", "templates")
		indefinitePrompt := nameIndex > 0 && (clauseWords[nameIndex-1] == "a" || clauseWords[nameIndex-1] == "an") && hasWord(clauseWords, "prompt", "prompts")
		suffix := strings.TrimSpace(clause[nameAt+len(name):])
		adjacentFormat = adjacentFormat || strings.HasPrefix(suffix, "模板") || strings.HasPrefix(suffix, "标签")
		transformedPrompt := strings.Contains(clause[:nameAt], "改成") && strings.HasPrefix(suffix, "提示词")
		namedTags := 0
		for _, tagName := range []string{"instruction", "selected_text", "typed_text"} {
			if strings.Contains(clause, tagName) {
				namedTags++
			}
		}
		sharedFormat := namedTags > 1 && (hasWord(clauseWords, "tag", "tags", "template", "templates", "prompt", "prompts") || strings.Contains(clause, "模板") || strings.Contains(clause, "标签") || strings.Contains(clause, "提示词"))
		if !literalTag && !adjacentFormat && !indefinitePrompt && !sharedFormat && !transformedPrompt {
			continue
		}
		prefix := clause[:nameAt]
		englishPrefix := strings.NewReplacer("not forget to", "", "n't forget to", "", "not hesitate to", "", "n't hesitate to", "", "not just", "", "n't just", "").Replace(prefix)
		negated := hasWord(splitWords(englishPrefix), "not", "never", "no") || strings.Contains(englishPrefix, "n't")
		for _, denial := range []string{"不要", "不用", "不需要", "不必", "无需", "无须", "没必要", "勿"} {
			negated = negated || strings.Contains(prefix, denial)
		}
		previous := rune(0)
		for _, r := range prefix {
			if (r == '不' || r == '别') && (previous == 0 || unicode.IsSpace(previous) || strings.ContainsRune("我你您他她它，。！？；,.!?;", previous)) {
				negated = true
			}
			previous = r
		}
		predicate := suffix
		for _, formatWord := range []string{"template", "tag", "prompt", "模板", "标签", "提示词"} {
			predicate = strings.TrimSpace(strings.TrimPrefix(predicate, formatWord))
		}
		negated = negated || strings.HasPrefix(predicate, "is not ") || strings.HasPrefix(predicate, "isn't ") || strings.HasPrefix(predicate, "is no ") || strings.HasPrefix(predicate, "不需要")
		question := strings.ContainsAny(clause, "?？")
		positiveQuestion := question && (strings.Contains(prefix, "can't you") || strings.Contains(prefix, "couldn't you") || strings.Contains(prefix, "cannot you") || strings.Contains(prefix, "不") && strings.Contains(clause, "吗") && (strings.Contains(clause, "你") || strings.Contains(clause, "您")))
		secondPerson := hasWord(clauseWords, "you") || strings.Contains(clause, "你") || strings.Contains(clause, "您") || strings.Contains(clause, "能不能")
		actionRequest := false
		for _, word := range clauseWords {
			switch word {
			case "write", "use", "draft", "create", "generate", "make", "turn", "convert", "rewrite", "want", "need":
				actionRequest = true
			}
		}
		for _, action := range []string{"写", "用", "起草", "改成", "生成", "制作", "创建", "想要", "需要"} {
			actionRequest = actionRequest || strings.Contains(clause, action)
		}
		switch {
		case positiveQuestion:
			positiveMention, negatedMention = true, false
		case negated:
			positiveMention, negatedMention = false, true
		case question && secondPerson && actionRequest:
			positiveMention, negatedMention = true, false
		case question:
			positiveMention, negatedMention = false, true
		default:
			positiveMention, negatedMention = true, false
		}
	}
	return positiveMention, negatedMention
}

func stripPartialInputTags(s string, content bool, request string, input ...string) string {
	s = stripInputTags(s, content, request, input...)
	for _, name := range []string{"instruction", "selected_text", "typed_text"} {
		open, close := "<"+name+">", "</"+name+">"
		i := strings.LastIndex(s, open)
		rest := ""
		if i >= 0 {
			rest = s[i+len(open):]
		}
		positiveIntent, negatedIntent := inputTemplateIntent(request, name)
		requestedTemplate := content && positiveIntent
		currentHasLiteral := len(input) > 0 && (strings.Contains(input[0], open) || strings.Contains(input[0], close))
		requestHasLiteral := strings.Contains(request, open) || strings.Contains(request, close)
		if requestedTemplate || content && currentHasLiteral && (!requestHasLiteral || !negatedIntent) {
			continue
		}
		if i >= 0 {
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
