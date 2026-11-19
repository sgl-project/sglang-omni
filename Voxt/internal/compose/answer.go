package compose

import (
	"encoding/json"
	"strings"
	"unicode/utf8"
)

// Answer is Rewrite's structured answer: a title for the answer card and
// the text to insert.
type Answer struct {
	Title   string `json:"title"`
	Content string `json:"content"`
}

// reply is the answer as an earlier assistant message: the JSON object
// when the request asks for one (the model keeps the form it sees), else
// the content.
func (a Answer) reply(asJSON bool) string {
	if !asJSON {
		return a.Content
	}
	b, _ := json.Marshal(a)
	return string(b)
}

// outputOpen and outputClose are the tags the clean-up's output contract
// asks the final text to sit between, so the model's commentary about
// its edits stays outside what is shown and inserted (#4086).
const (
	outputOpen  = "<output>"
	outputClose = "</output>"
)

// taggedOutput is the content of the first outputOpen…outputClose pair
// in s (of outputOpen alone when the answer was cut short). ok is false
// when s has no outputOpen; the caller then takes the text as it is.
func taggedOutput(s string) (string, bool) {
	i := strings.Index(s, outputOpen)
	if i < 0 {
		return "", false
	}
	rest := s[i+len(outputOpen):]
	if j := strings.Index(rest, outputClose); j >= 0 {
		rest = rest[:j]
	}
	return rest, true
}

// unwrappedOutput is s without one outermost outputOpen…outputClose
// pair, when that pair wraps the whole of s: a contract-abiding model's
// wrapper around an answer that has the user's own tags (#4086). A pair
// that closes early is the user's own text (say two pairs in a row)
// and stays.
func unwrappedOutput(s string) string {
	if !strings.HasPrefix(s, outputOpen) || !strings.HasSuffix(s, outputClose) {
		return s
	}
	mid := s[len(outputOpen) : len(s)-len(outputClose)]
	depth := 1
	for rest := mid; ; {
		o, c := strings.Index(rest, outputOpen), strings.Index(rest, outputClose)
		switch {
		case c >= 0 && (o < 0 || c < o):
			if depth--; depth == 0 {
				return s
			}
			rest = rest[c+len(outputClose):]
		case o >= 0:
			depth++
			rest = rest[o+len(outputOpen):]
		default:
			return mid
		}
	}
}

var thinkingTags = [][2]string{{"<think>", "</think>"}, {"<|channel>", "<channel|>"}}

const assistantAnswerPrefix = "<|im_start|>assistant"

var controlTokens = []string{
	"<|tool_call|>", "<|/tool_call|>", "<|tool_call_begin|>", "<|tool_call_end|>",
	"<|tool_calls_section_begin|>", "<|tool_calls_section_end|>",
	"<|im_end|>", "<|im_start|>", "<|start|>", "<|end|>", "<|message|>",
	"<|channel>", "<|channel|>", "<channel|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>", "</think>",
}

func preservedControlToken(token, input string) bool {
	for _, tag := range thinkingTags {
		if (token == tag[0] || token == tag[1]) && (strings.Contains(input, tag[0]) || strings.Contains(input, tag[1])) {
			return true
		}
	}
	return strings.Contains(input, token)
}

func partialControlToken(s, input string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	for _, token := range controlTokens {
		if len(s) < len(token) && strings.HasPrefix(token, s) && !preservedControlToken(token, input) {
			return true
		}
	}
	return false
}

// controlTokenStart is the start of the first model control token in s
// that did not occur in the user's input; -1 when s has only answer text.
func controlTokenStart(s, input string) int {
	start := -1
	for _, token := range controlTokens {
		if preservedControlToken(token, input) {
			continue
		}
		for offset := 0; offset < len(s); {
			i := strings.Index(s[offset:], token)
			if i < 0 {
				break
			}
			i += offset
			if trailingControlToken(token) && !onlyControlMarkup(s[i+len(token):], input) {
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

func trailingControlToken(token string) bool {
	switch token {
	case "<|/tool_call|>", "<|tool_call_end|>", "<|tool_calls_section_end|>", "<|im_end|>", "<|end|>",
		"<|channel|>", "<channel|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>":
		return true
	default:
		return false
	}
}

// onlyControlMarkup reports whether s is a control-only ending. A block
// token makes the rest its payload, including braces and plain text.
func onlyControlMarkup(s, input string) bool {
	for {
		s = strings.TrimLeft(s, " \t\r\n")
		if s == "" {
			return true
		}
		if partialControlToken(s, input) {
			return true
		}
		matched := false
		for _, tag := range thinkingTags {
			if strings.Contains(input, tag[0]) || strings.Contains(input, tag[1]) {
				continue
			}
			if strings.HasPrefix(s, tag[0]) {
				rest := s[len(tag[0]):]
				j := strings.Index(rest, tag[1])
				if j < 0 {
					return true
				}
				s = rest[j+len(tag[1]):]
				matched = true
				break
			}
			if strings.HasPrefix(s, tag[1]) {
				s = s[len(tag[1]):]
				matched = true
				break
			}
		}
		if matched {
			continue
		}
		for _, token := range controlTokens {
			if preservedControlToken(token, input) || !strings.HasPrefix(s, token) {
				continue
			}
			if !trailingControlToken(token) {
				return true
			}
			s = strings.TrimPrefix(s, token)
			matched = true
			break
		}
		if !matched {
			return false
		}
	}
}

// stripControlMarkup removes model control blocks, including an opening
// marker whose block has not closed. Ordinary angle-bracket text and
// control markup that occurred in the user's input stay.
func stripControlMarkup(s string, inputs ...string) string {
	input := strings.Join(inputs, "\n")
	trimmed := strings.TrimLeft(s, " \t\r\n")
	if !strings.Contains(input, "<|im_start|>") && strings.HasPrefix(trimmed, assistantAnswerPrefix) {
		s = strings.TrimLeft(trimmed[len(assistantAnswerPrefix):], " \t\r\n")
	}
	for _, tag := range thinkingTags {
		if strings.Contains(input, tag[0]) || strings.Contains(input, tag[1]) {
			continue
		}
		closeSearchFrom := 0
		for {
			i := strings.Index(s, tag[0])
			j := strings.Index(s[closeSearchFrom:], tag[1])
			if j >= 0 {
				j += closeSearchFrom
			}
			if j >= 0 && (i < 0 || j < i) {
				if tag[1] == "<channel|>" && j > 0 && j+len(tag[1]) < len(s) &&
					strings.ContainsRune(" \t\r\n", rune(s[j-1])) && strings.ContainsRune(" \t\r\n", rune(s[j+len(tag[1])])) {
					closeSearchFrom = j + len(tag[1])
					continue
				}
				after := j + len(tag[1])
				if onlyControlMarkup(s[after:], input) {
					s = s[:j]
				} else {
					s = s[after:]
				}
				closeSearchFrom = 0
				continue
			}
			if i < 0 {
				break
			}
			rest := s[i+len(tag[0]):]
			j = strings.Index(rest, tag[1])
			if j < 0 {
				s = s[:i]
				break
			}
			s = s[:i] + rest[j+len(tag[1]):]
			closeSearchFrom = 0
		}
	}
	if i := controlTokenStart(s, input); i >= 0 {
		s = s[:i]
	}
	return s
}

// stripPartialControlMarkup also holds back a suffix that may still
// become a control marker, so no marker fragment shows while streaming.
func stripPartialControlMarkup(s string, inputs ...string) string {
	s = stripControlMarkup(s, inputs...)
	input := strings.Join(inputs, "\n")
	start := -1
	for _, token := range controlTokens {
		if preservedControlToken(token, input) {
			continue
		}
		if n := partialTagSuffix(s, token); n > 1 {
			if i := len(s) - n; start < 0 || i < start {
				start = i
			}
		}
	}
	for _, tag := range thinkingTags {
		if strings.Contains(input, tag[0]) || strings.Contains(input, tag[1]) {
			continue
		}
		if n := partialTagSuffix(s, tag[0]); n > 0 {
			if i := len(s) - n; start < 0 || i < start {
				start = i
			}
		}
	}
	if start >= 0 {
		s = s[:start]
	}
	return s
}

// PartialOutput is a streaming clean-up answer's text so far, for the
// card: without the contract's <output> tags, whole or as far as they
// came, and nothing of the commentary around them. A closing tag only
// truncates after an opening tag came; without one the text is the
// user's own and shows as it is, as CleanOutput keeps it. When the
// input itself has the tags, one outermost wrapper pair goes, as in
// CleanOutput.
func PartialOutput(s, input string) string {
	t := strings.TrimLeft(stripPartialControlMarkup(s, input), " \t\r\n")
	if strings.Contains(input, outputOpen) {
		return unwrappedOutput(t)
	}
	if t == "" || strings.HasPrefix(outputOpen, t) {
		return "" // nothing yet, or the opening tag is still coming
	}
	i := strings.Index(t, outputOpen)
	if i < 0 {
		// No opening tag came: the text shows as it is, but a stream
		// ending in the opening tag's start holds back — the tag may
		// complete, and the commentary before it then goes.
		if partialTagSuffix(t, outputOpen) > 0 {
			return ""
		}
		return t
	}
	// The commentary before the opening tag goes, as CleanOutput drops
	// it.
	t = strings.TrimLeft(t[i+len(outputOpen):], " \t\r\n")
	if i := strings.Index(t, outputClose); i >= 0 {
		return strings.TrimRight(t[:i], " \t\r\n")
	}
	if i := partialTagSuffix(t, outputClose); i > 0 {
		return strings.TrimRight(t[:len(t)-i], " \t\r\n")
	}
	return t
}

// CleanOutput tidies a Transcribe clean-up's answer to input: the
// content of the <output> tags the clean-up's contract asks for when
// they are there and the input did not have them (never the model's
// commentary outside them, #4086); when the input itself has the tags,
// one outermost pair the model wrapped its whole answer in goes and the
// text keeps its own tags. Then Clean's tidy. Only the Transcribe
// clean-up asks for the tags: Rewrite's and Translate's answers go to
// Clean, so tags their text itself has (HTML they were asked for) stay.
func CleanOutput(out, input string) string {
	s := strings.TrimSpace(out)
	if strings.Contains(input, outputOpen) {
		s = unwrappedOutput(s)
	} else if content, ok := taggedOutput(s); ok {
		s = strings.TrimSpace(content)
	}
	return Clean(s, input)
}

// Clean tidies a model's answer to input: without the <input> tags a
// model echoed, or the quotes it wrapped the whole answer in (unless
// the input was quoted so itself).
func Clean(out, input string) string {
	return clean(out, input, false)
}

// PartialClean is Clean for a streaming answer: possible control-marker
// fragments stay hidden until more text arrives.
func PartialClean(out, input string) string {
	return clean(out, input, true)
}

func clean(out, input string, partial bool) string {
	s := strings.TrimSpace(out)
	s = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(s, "<input>"), "</input>"))
	in := strings.TrimSpace(input)
	for _, q := range [][2]string{{`"`, `"`}, {"“", "”"}, {"「", "」"}} {
		wrapped := func(t string) bool {
			return len(t) > len(q[0])+len(q[1]) && strings.HasPrefix(t, q[0]) && strings.HasSuffix(t, q[1]) &&
				!strings.Contains(t[len(q[0]):len(t)-len(q[1])], q[0])
		}
		if wrapped(s) && !wrapped(in) {
			s = strings.TrimSpace(s[len(q[0]) : len(s)-len(q[1])])
		}
	}
	if partial {
		return strings.TrimSpace(stripPartialControlMarkup(s, input))
	}
	return strings.TrimSpace(stripControlMarkup(s, input))
}

// partialTagSuffix is the length of s's suffix that is a proper prefix
// of tag (the tag's first characters, the rest still coming); 0 for
// none.
func partialTagSuffix(s, tag string) int {
	for i := len(tag) - 1; i > 0; i-- {
		if strings.HasSuffix(s, tag[:i]) {
			return i
		}
	}
	return 0
}

// ParseAnswer reads a model's JSON answer {title, content}: the object
// may sit in a code fence or among other text. ok is false when there is
// no such object; the caller then takes the text as it is (Clean),
// without a title. An object whose content is empty is an empty answer.
// Input is the user's source text and instruction, whose own markers stay.
func ParseAnswer(text string, input ...string) (Answer, bool) {
	raw := extractObject(text)
	if raw == "" {
		return Answer{}, false
	}
	var fields map[string]json.RawMessage
	var a Answer
	if json.Unmarshal([]byte(raw), &fields) != nil || json.Unmarshal([]byte(raw), &a) != nil {
		return Answer{}, false
	}
	if _, ok := fields["content"]; !ok {
		return Answer{}, false
	}
	a.Title = strings.TrimSpace(strings.Trim(strings.TrimSpace(stripControlMarkup(a.Title, input...)), `"“”#*`))
	a.Content = strings.TrimSpace(stripControlMarkup(a.Content, input...))
	return a, true
}

// extractObject is the first balanced JSON object in text ("" for none).
func extractObject(text string) string {
	for i := strings.IndexByte(text, '{'); i >= 0; {
		depth, inStr, esc := 0, false, false
		for j := i; j < len(text); j++ {
			c := text[j]
			switch {
			case esc:
				esc = false
			case inStr && c == '\\':
				esc = true
			case c == '"':
				inStr = !inStr
			case inStr:
			case c == '{':
				depth++
			case c == '}':
				depth--
				if depth == 0 {
					if json.Valid([]byte(text[i : j+1])) {
						return text[i : j+1]
					}
					j = len(text)
				}
			}
		}
		next := strings.IndexByte(text[i+1:], '{')
		if next < 0 {
			break
		}
		i += 1 + next
	}
	return ""
}

// PartialAnswer reads what a streaming answer holds so far: the title and
// the content of a JSON answer as far as they came, or, when the model
// writes plain text instead, the text itself as content. Input is the
// user's source text and instruction, whose own markers stay.
func PartialAnswer(text string, input ...string) Answer {
	t := strings.TrimSpace(text)
	if t == "" {
		return Answer{}
	}
	if strings.HasPrefix(t, "```") {
		if nl := strings.IndexByte(t, '\n'); nl >= 0 {
			t = strings.TrimSpace(t[nl+1:])
		} else {
			return Answer{} // the fence's first line, still coming
		}
	}
	if !strings.HasPrefix(t, "{") {
		t = strings.TrimSpace(stripPartialControlMarkup(t, input...))
		if strings.HasPrefix(t, "```") {
			if nl := strings.IndexByte(t, '\n'); nl >= 0 {
				t = strings.TrimSpace(t[nl+1:])
			} else {
				return Answer{} // the fence's first line, still coming
			}
		}
		if !strings.HasPrefix(t, "{") {
			return Answer{Content: strings.TrimSpace(strings.TrimSuffix(t, "```"))}
		}
	}
	title := strings.TrimSpace(stripPartialControlMarkup(partialField(t, "title"), input...))
	content := strings.TrimLeft(stripPartialControlMarkup(partialField(t, "content"), input...), " \n")
	return Answer{Title: title, Content: content}
}

// partialField is the string value of key in a JSON object that may be
// cut short, decoded as far as it goes ("" when the key has not come).
func partialField(obj, key string) string {
	k := `"` + key + `"`
	i := strings.Index(obj, k)
	for i >= 0 {
		rest := strings.TrimLeft(obj[i+len(k):], " \t\r\n")
		if strings.HasPrefix(rest, ":") {
			rest = strings.TrimLeft(rest[1:], " \t\r\n")
			if !strings.HasPrefix(rest, `"`) {
				return ""
			}
			return decodePartialString(rest[1:])
		}
		next := strings.Index(obj[i+1:], k)
		if next < 0 {
			return ""
		}
		i += 1 + next
	}
	return ""
}

// decodePartialString decodes a JSON string's body up to its closing
// quote or the end of s, dropping an escape cut in half.
func decodePartialString(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '"':
			return b.String()
		case c != '\\':
			r, n := utf8.DecodeRuneInString(s[i:])
			if r == utf8.RuneError && n == 1 && !utf8.FullRuneInString(s[i:]) {
				return b.String()
			}
			b.WriteString(s[i : i+n])
			i += n
			continue
		}
		if i+1 >= len(s) {
			return b.String()
		}
		switch e := s[i+1]; e {
		case 'n':
			b.WriteByte('\n')
		case 't':
			b.WriteByte('\t')
		case 'r':
		case 'b', 'f':
		case 'u':
			// \uXXXX, or a surrogate pair of two.
			var out string
			adv := 0
			if i+12 <= len(s) && s[i+6] == '\\' && s[i+7] == 'u' &&
				json.Unmarshal([]byte(`"`+s[i:i+12]+`"`), &out) == nil && !strings.ContainsRune(out, utf8.RuneError) {
				adv = 12
			} else {
				if i+6 > len(s) {
					return b.String()
				}
				out = ""
				if json.Unmarshal([]byte(`"`+s[i:i+6]+`"`), &out) == nil && strings.ContainsRune(out, utf8.RuneError) && i+12 > len(s) {
					return b.String() // a pair's second half is still coming
				}
				adv = 6
			}
			b.WriteString(out)
			i += adv
			continue
		default:
			b.WriteByte(e) // \" \\ \/
		}
		i += 2
	}
	return b.String()
}
