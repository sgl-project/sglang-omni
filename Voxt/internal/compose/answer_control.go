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

func proseTagMention(rest string) bool {
	if strings.HasPrefix(rest, " 标记") || strings.HasPrefix(rest, " literally") {
		return true
	}
	if !strings.HasPrefix(rest, " tag") && !strings.HasPrefix(rest, "\ttag") {
		return false
	}
	if len(rest) == len(" tag") {
		return true
	}
	next, _ := utf8.DecodeRuneInString(rest[len(" tag"):])
	return unicode.IsSpace(next) || unicode.IsPunct(next)
}

func proseTokenMention(s string, start int, token string) bool {
	if quotedControlToken(s, start, token) {
		return true
	}
	if start > 0 && !strings.ContainsRune(" \t\r\n", rune(s[start-1])) {
		return false
	}
	rest := s[start+len(token):]
	if proseTagMention(rest) {
		return true
	}
	switch token {
	case "<|im_start|>", "<|im_end|>":
		return strings.HasPrefix(rest, " to ") && explanatoryControlContext(s[:start])
	case "<think>":
		return strings.HasPrefix(rest, " before it ") && explanatoryControlContext(s[:start])
	default:
		return false
	}
}

func explanatoryControlContext(prefix string) bool {
	if index := strings.LastIndexAny(prefix, ".!?;,:\n，。！？；："); index >= 0 {
		prefix = prefix[index+1:]
	}
	prefix = strings.ToLower(prefix)
	return strings.Contains(prefix, "chatml") || strings.Contains(prefix, "emit")
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
				if preserved && token == tag[1] && strings.TrimSpace(after) == "" &&
					strings.LastIndex(s[:i], tag[0]) > strings.LastIndex(s[:i], tag[1]) {
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
