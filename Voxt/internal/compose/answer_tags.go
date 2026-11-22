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
	englishActions := []string{"write", "create", "generate", "make", "produce", "use", "turn", "give", "show", "output", "return", "provide", "convert", "rewrite", "wrap", "need", "want"}
	chineseActions := []string{"写", "生成", "创建", "输出", "返回", "给", "用", "改", "按", "做"}
	chineseDenials := []string{"不要", "不用", "不需要", "不必", "无需", "无须", "勿"}
	positiveMention, negatedMention := false, false
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
		if nameIndex < 0 || !hasWord(clauseWords, "tag", "tags", "template", "templates", "prompt", "prompts") && !strings.Contains(clause, "标签") && !strings.Contains(clause, "模板") && !strings.Contains(clause, "提示词") {
			continue
		}
		negated := false
		nameAt := strings.Index(clause, name)
		prefix := clause[:nameAt]
		for _, denial := range chineseDenials {
			negated = negated || strings.Contains(prefix, denial)
		}
		for _, action := range chineseActions {
			for _, denial := range []string{"不", "别", "勿"} {
				if strings.Contains(prefix, denial+action) && (denial != "别" || !strings.Contains(prefix, "分别"+action)) {
					negated = true
				}
			}
		}
		for i := 1; i < nameIndex; i++ {
			if !hasWord(clauseWords[i:i+1], englishActions...) {
				continue
			}
			previous := clauseWords[i-1]
			directNegation := previous == "not" || previous == "never" || previous == "t" && i > 1 && strings.HasSuffix(clauseWords[i-2], "n")
			contractionWithAdverb := i > 2 && clauseWords[i-2] == "t" && strings.HasSuffix(clauseWords[i-3], "n") && (previous == "really" || previous == "actually")
			if directNegation || contractionWithAdverb || i > 1 && (clauseWords[i-2] == "not" || clauseWords[i-2] == "never") {
				negated = true
			}
		}
		if negated {
			negatedMention = true
		} else {
			positiveMention = true
		}
	}
	if negatedMention && !positiveMention {
		return false
	}
	words := splitWords(request)
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
	question := strings.ContainsAny(request, "?？")
	for _, action := range chineseActions {
		actionAt := strings.Index(request, action)
		if actionAt < 0 {
			continue
		}
		prefix := strings.TrimSuffix(request[:actionAt], "使")
		negated := strings.HasSuffix(prefix, "不") || strings.HasSuffix(prefix, "别") && !strings.HasSuffix(prefix, "分别")
		for _, denial := range chineseDenials {
			negated = negated || strings.HasSuffix(prefix, denial)
		}
		if negated || question && action == "用" {
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
	hasAction := func(parts []string, targets ...string) bool {
		for i, part := range parts {
			if !hasWord(parts[i:i+1], targets...) {
				continue
			}
			negated := i > 0 && (parts[i-1] == "not" || parts[i-1] == "never") || i > 1 && parts[i-1] == "t" && strings.HasSuffix(parts[i-2], "n")
			if negated || question && (part == "use" || part == "need" || part == "want") {
				continue
			}
			return true
		}
		return false
	}
	if hasAction(words, englishActions...) && hasWord(words, "tag", "tags", "template", "templates") {
		return true
	}
	if !hasWord(words, "prompt", "prompts") {
		return false
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
			if hasAction(words[:j], "turn", "convert", "rewrite") {
				return true
			}
		}
		if hasAction(words[:i], "turn", "convert", "rewrite") {
			continue
		}
		if hasAction(words[:i], "write", "create", "generate", "make", "produce", "give", "show", "output", "return", "provide") {
			return true
		}
		if hasAction(words, "need", "want") && (!hasWord(words[i+1:], "to") || hasAction(words, "write", "create", "generate", "produce", "give", "show", "output", "return", "provide")) {
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
