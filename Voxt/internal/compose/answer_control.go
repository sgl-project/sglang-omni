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
	if start == 0 || !strings.ContainsRune(" \t", rune(s[start-1])) {
		return false
	}
	next, _ := utf8.DecodeRuneInString(s[start+len(token):])
	return start+len(token) < len(s) && (strings.ContainsRune(" \t", next) || unicode.IsPunct(next))
}

func proseTokenMention(s string, start int, token string) bool {
	if quotedControlToken(s, start, token) {
		return true
	}
	rest := s[start+len(token):]
	trimmed := strings.TrimLeft(rest, " \t")
	if trimmed == rest {
		return false
	}
	if structuralControlToken(token, rest) && bracketedToolPayload(trimmed) {
		return false
	}
	for _, word := range []string{"tag", "marker", "token", "tokens", "literally", "标记"} {
		if strings.HasPrefix(trimmed, word) {
			return true
		}
	}
	if structuralControlToken(token, rest) && structuredToolPayload(rest) {
		return false
	}
	if start == 0 {
		return strings.HasPrefix(trimmed, "is ")
	}
	for _, phrase := range []string{"and ", "at ", "in ", "into ", "is ", "to ", "for ", "on ", "starts ", "before the ", "before its ", "before your ", "before a ", "before every ", "就", "可以", "能", "让", "模型就"} {
		if strings.HasPrefix(trimmed, phrase) {
			return true
		}
	}
	return strings.HasPrefix(trimmed, "before it ") && modelOutputContext(s[:start])
}

func modelOutputContext(prefix string) bool {
	if index := strings.LastIndexAny(prefix, ".!?;,:\n，。！？；："); index >= 0 {
		prefix = prefix[index+1:]
	}
	prefix = strings.ToLower(prefix)
	return strings.Contains(prefix, "chatml") || strings.Contains(prefix, "emit")
}

func bracketedToolPayload(rest string) bool {
	if rest == "" {
		return false
	}
	if strings.ContainsRune("{[\"", rune(rest[0])) {
		return true
	}
	fields := strings.Fields(rest)
	return len(fields) > 1 && strings.ContainsRune("{[\"", rune(fields[1][0]))
}

func structuredToolPayload(after string) bool {
	rest := strings.TrimLeft(after, " \t\r\n")
	if rest == "" {
		return false
	}
	if bracketedToolPayload(rest) {
		return true
	}
	fields := strings.Fields(rest)
	return len(fields) == 1 || strings.Contains(fields[0], "_")
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
			after := s[i+len(token):]
			proseBlock := false
			for _, tag := range thinkingTags {
				if token == tag[0] {
					if close := strings.Index(after, tag[1]); close >= 0 {
						close += i + len(token)
						if proseTokenMention(s, close, tag[1]) {
							offset, proseBlock = close+len(tag[1]), true
						}
					}
				}
			}
			if proseBlock {
				continue
			}
			if proseTokenMention(s, i, token) {
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
					literalThinkingPair(s[open:i+len(token)], tag, input, previous) && !proseTokenMention(s, open, tag[0]) {
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
			if token != "</think>" && trailingControlToken(token) && !onlyControlMarkup(after, input, previous...) {
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
