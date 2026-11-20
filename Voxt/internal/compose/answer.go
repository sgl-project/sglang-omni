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

// PartialOutput is a streaming clean-up answer's text so far, for the
// card: without the contract's <output> tags, whole or as far as they
// came, and nothing of the commentary around them. A closing tag only
// truncates after an opening tag came; without one the text is the
// user's own and shows as it is, as CleanOutput keeps it. When the
// input itself has the tags, one outermost wrapper pair goes, as in
// CleanOutput.
func PartialOutput(s, input string) string {
	t := strings.TrimLeft(s, " \t\r\n")
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

// stripInputTags drops an <instruction>, <selected_text> or <typed_text>
// pair a model echoed from RewriteMessage, keeping the text between: a
// pair that wraps the whole answer, or one around exactly a text the user sent (whole
// lines of input). Other pairs are part of the answer (a prompt template the
// user asked for) and stay, as do the tags the user's own text has.
func stripInputTags(s string, input ...string) string {
	sent := "\n" + strings.Join(input, "\n") + "\n"
	for _, name := range []string{"instruction", "selected_text", "typed_text"} {
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
			if whole || inner != "" && strings.Contains(sent, "\n"+inner+"\n") {
				s = s[:i] + inner + s[j+len(close):]
				from = i + len(inner)
			} else {
				from = j + len(close)
			}
		}
	}
	return s
}

func stripPartialInputTags(s string, input ...string) string {
	s = stripInputTags(s, input...)
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
		if n := partialTagSuffix(s, open); n > 0 {
			s = s[:len(s)-n]
		}
		if n := partialTagSuffix(s, close); n > 0 {
			s = s[:len(s)-n]
		}
	}
	return s
}

// ParseAnswer reads a model's JSON answer {title, content}: the object
// may sit in a code fence or among other text. ok is false when there is
// no such object; the caller then takes the text as it is (Clean),
// without a title. An object whose content is empty is an empty answer.
// Input is what the user sent (the selection, typed text, the instruction).
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
	a.Title = strings.TrimSpace(strings.Trim(strings.TrimSpace(stripPartialInputTags(a.Title, input...)), `"“”#*`))
	a.Content = strings.TrimSpace(stripPartialInputTags(a.Content, input...))
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
// writes plain text instead, the text itself as content. Input is as
// ParseAnswer's.
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
		return Answer{Content: strings.TrimSpace(strings.TrimSuffix(stripPartialInputTags(t, input...), "```"))}
	}
	return Answer{Title: strings.TrimSpace(stripPartialInputTags(partialField(t, "title"), input...)), Content: strings.TrimLeft(stripPartialInputTags(partialField(t, "content"), input...), " \n")}
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
