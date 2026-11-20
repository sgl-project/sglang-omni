package compose

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

func previousControlToken(token string, inputs []string) bool {
	for _, input := range inputs {
		if preservedControlToken(token, input) {
			return true
		}
	}
	return false
}

func inlineLiteral(s string, start int, token string) bool {
	if start == 0 || !strings.ContainsRune(" \t\r\n", rune(s[start-1])) {
		return false
	}
	next, _ := utf8.DecodeRuneInString(s[start+len(token):])
	return start+len(token) < len(s) && (strings.ContainsRune(" \t", next) || unicode.IsPunct(next))
}

func markLiteral(s string, marker *string, index int, token string) {
	if *marker == "" {
		*marker = s[index : index+len(token)]
	}
}

func fieldStart(s string, start int) bool {
	return strings.TrimSpace(s[:start]) == ""
}

func inlineLiteralAtStart(s, token string) bool {
	s = strings.TrimLeft(s, " \t\r\n")
	next, _ := utf8.DecodeRuneInString(s[len(token):])
	return len(s) > len(token) && (strings.ContainsRune(" \t\r\n", next) || unicode.IsPunct(next))
}

func proseContinuation(rest string) bool {
	if rest == "" {
		return false
	}
	first, _ := utf8.DecodeRuneInString(rest)
	if !unicode.IsSpace(first) && !unicode.IsPunct(first) {
		return false
	}
	trimmed := strings.TrimLeftFunc(rest, func(r rune) bool {
		if r == '<' || r == '>' {
			return false
		}
		return unicode.IsSpace(r) || unicode.IsPunct(r)
	})
	if trimmed == "" {
		return false
	}
	next, _ := utf8.DecodeRuneInString(trimmed)
	return unicode.IsLower(next) || unicode.Is(unicode.Han, next)
}

func lowerOrHan(rest string) bool {
	trimmed := strings.TrimLeft(rest, " \t")
	if trimmed == "" {
		return false
	}
	next, _ := utf8.DecodeRuneInString(trimmed)
	return unicode.IsLower(next) || unicode.Is(unicode.Han, next)
}

func literalThinkingContext(s string, start int, token, input string, previous []string) bool {
	return quotedControlToken(s, start, token) || preservedControlToken(token, input) || previousControlToken(token, previous)
}

func structuralControlSuffix(s string) bool {
	for _, token := range controlTokens {
		index := strings.Index(s, token)
		if index < 0 {
			continue
		}
		after := s[index+len(token):]
		if !structuralControlToken(token, after) {
			continue
		}
		if !strings.HasPrefix(token, "<|tool") || bracketedToolPayload(after) {
			return true
		}
		if fields := strings.Fields(after); len(fields) > 0 && strings.Contains(fields[0], "_") {
			return true
		}
	}
	return false
}

func proseTokenMention(s string, start int, token string, inputs ...string) bool {
	if quotedControlToken(s, start, token) {
		return true
	}
	input := ""
	var previous []string
	if len(inputs) > 0 {
		input, previous = inputs[0], inputs[1:]
	}
	rest := s[start+len(token):]
	for _, tag := range thinkingTags {
		if token == tag[1] {
			if fieldStart(s, start) && previousControlToken(token, previous) && inlineLiteralAtStart(s, token) && proseContinuation(rest) {
				return true
			}
			return proseContinuation(rest)
		}
		if token == tag[0] {
			close := strings.Index(rest, tag[1])
			if close < 0 || fieldStart(s, start) {
				return false
			}
			after := rest[close+len(tag[1]):]
			return strings.TrimSpace(after) != "" && lowerOrHan(rest) && proseContinuation(after) && literalThinkingContext(s, start, tag[0], input, previous)
		}
	}
	if structuralControlSuffix(rest) {
		return false
	}
	if structuralControlToken(token, rest) {
		if !strings.HasPrefix(token, "<|tool") {
			return false
		}
		if bracketedToolPayload(rest) {
			return false
		}
		if fields := strings.Fields(rest); len(fields) > 0 && strings.Contains(fields[0], "_") {
			return false
		}
	}
	if !trailingControlToken(token) || turnEndLeak(rest) {
		return false
	}
	if inlineLiteral(s, start, token) {
		return true
	}
	return false
}

// turnEndLeak says the text after a turn-end marker is a fresh
// sentence: the model kept writing into the next turn, so the marker
// gets no prose exemption. A fragment or a continuation without
// sentence shape ("x", "放在句尾") stays literal.
func turnEndLeak(after string) bool {
	rest := strings.TrimSpace(after)
	if rest == "" || !strings.ContainsRune(".!?。！？", rune(rest[len(rest)-1])) {
		return false
	}
	first, _ := utf8.DecodeRuneInString(rest)
	return unicode.IsUpper(first) || unicode.Is(unicode.Han, first)
}

func bracketedToolPayload(rest string) bool {
	index := strings.IndexAny(rest, "{[")
	if index < 0 {
		return false
	}
	before := strings.TrimSpace(rest[:index])
	if before == "" {
		return true
	}
	last, _ := utf8.DecodeLastRuneInString(before)
	return !strings.ContainsRune(".!?。！？", last)
}

func exactThinkingPair(pair string, input string, previous []string) bool {
	for _, context := range append([]string{input}, previous...) {
		if strings.Contains(context, pair) {
			return true
		}
	}
	return false
}

func literalThinkingPair(pair string, tag [2]string, input string, previous []string) bool {
	for _, context := range append([]string{input}, previous...) {
		if strings.Contains(context, pair) {
			return true
		}
		if strings.Contains(context, "literally") && (strings.Contains(context, tag[0]) || strings.Contains(context, tag[1])) {
			return true
		}
	}
	return false
}

func quotedControlToken(s string, start int, token string) bool {
	if start == 0 || start+len(token) >= len(s) {
		return false
	}
	before, _ := utf8.DecodeLastRuneInString(s[:start])
	after, _ := utf8.DecodeRuneInString(s[start+len(token):])
	for _, quotes := range [][2]rune{{'`', '`'}, {'"', '"'}, {'“', '”'}, {'「', '」'}, {'‘', '’'}} {
		if before == quotes[0] && after == quotes[1] {
			return true
		}
	}
	return false
}

func structuralControlToken(token, after string) bool {
	if trailingControlToken(token) && strings.TrimSpace(after) == "" {
		return true
	}
	rest := strings.TrimLeft(after, " \t\r\n")
	switch token {
	case "<|tool_call|>", "<|/tool_call|>", "<|tool_call_begin|>", "<|tool_call_end|>", "<|tool_calls_section_begin|>", "<|tool_calls_section_end|>":
		return true
	case "<|im_start|>", "<|start|>":
		for _, role := range []string{"assistant", "user", "system"} {
			if strings.HasPrefix(rest, role) {
				return true
			}
		}
	}
	return false
}

// controlTokenStart is the start of the first model control token in s
// that did not occur in the user's input; -1 when s has only answer text.
func controlTokenStart(s string, inputs ...string) int {
	return controlTokenStartSkipping(s, "", inputs...)
}

func controlTokenStartSkipping(s, literalOpen string, inputs ...string) int {
	input := ""
	var previous []string
	if len(inputs) > 0 {
		input, previous = inputs[0], inputs[1:]
	}
	start := -1
	for _, token := range controlTokens {
		preserved := preservedControlToken(token, input)
		previousPreserved := !preserved && previousControlToken(token, previous)
		for offset := 0; offset < len(s); {
			i := strings.Index(s[offset:], token)
			if i < 0 {
				break
			}
			i += offset
			if literalOpen != "" && strings.HasPrefix(s[i:], literalOpen) && strings.Index(s, literalOpen) == i {
				offset = i + len(token)
				continue
			}
			after := s[i+len(token):]
			proseBlock := false
			for _, tag := range thinkingTags {
				if token == tag[0] {
					if close := strings.Index(after, tag[1]); close >= 0 {
						close += i + len(token)
						if strings.TrimSpace(s[close+len(tag[1]):]) != "" && proseTokenMention(s, i, token, inputs...) {
							offset, proseBlock = close+len(tag[1]), true
						}
					}
				}
			}
			if proseBlock {
				continue
			}
			if proseTokenMention(s, i, token, inputs...) {
				offset = i + len(token)
				continue
			}
			if preserved && trailingControlToken(token) && strings.TrimSpace(s) == token {
				offset = i + len(token)
				continue
			}
			literalPair := false
			for _, tag := range thinkingTags {
				open := strings.LastIndex(s[:i], tag[0])
				if token == tag[1] && strings.TrimSpace(after) == "" && open > strings.LastIndex(s[:i], tag[1]) &&
					literalThinkingPair(s[open:i+len(token)], tag, input, previous) && !proseTokenMention(s, open, tag[0], inputs...) {
					literalPair = true
				}
			}
			if literalPair {
				offset = i + len(token)
				continue
			}
			if preserved && !structuralControlToken(token, after) {
				offset = i + len(token)
				continue
			}
			if previousPreserved && trailingControlToken(token) && strings.TrimSpace(s) == token {
				offset = i + len(token)
				continue
			}
			if token != "</think>" && trailingControlToken(token) && !turnEndLeak(after) && !onlyControlMarkup(after, input, previous...) {
				offset = i + len(token)
				continue
			}
			if start < 0 || i < start {
				start = i
			}
			break
		}
	}
	return start
}
