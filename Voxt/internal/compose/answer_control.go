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
			if preserved {
				offset = i + len(token)
				continue
			}
			if previousPreserved && trailingControlToken(token) && (strings.TrimSpace(s) == token || strings.TrimSpace(after) != "" && inlineLiteral(s, i, token)) {
				offset = i + len(token)
				continue
			}
			if trailingControlToken(token) && !onlyControlMarkup(after, input) {
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
