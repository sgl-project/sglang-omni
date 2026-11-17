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

// ParseAnswer reads a model's JSON answer {title, content}: the object
// may sit in a code fence or among other text. ok is false when there is
// no such object; the caller then takes the text as it is (Clean),
// without a title. An object whose content is empty is an empty answer.
func ParseAnswer(text string) (Answer, bool) {
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
	a.Title = strings.TrimSpace(strings.Trim(strings.TrimSpace(StripControl(a.Title)), `"“”#*`))
	a.Content = strings.TrimSpace(StripControl(a.Content))
	return a, true
}

// StripControl removes a model's internal control blocks (<|...|>:
// tool-call and thinking scaffolding some backends emit as plain text,
// #4085). They are markup, never answer text: a <|..._begin|> block goes
// with what it wraps, through its <|..._end|>, and a block cut off by
// the end of the stream goes too.
func StripControl(s string) string {
	for at := 0; ; {
		k := strings.Index(s[at:], "<|")
		if k < 0 {
			return s
		}
		i := at + k
		j := strings.Index(s[i+2:], "|>")
		if j < 0 {
			// Only a cut-off tag's tail goes with the stream's end; text
			// after a real <| (F#'s, a user's) stays.
			if controlTag(s[i+2:]) {
				return strings.TrimRight(s[:i], " \t\r\n")
			}
			return s
		}
		name := s[i+2 : i+2+j]
		rest := s[i+2+j+2:]
		// A wrapper goes with what it wraps: <|..._begin|> through its
		// <|..._end|>, a thought <|channel|> through <channel|> (Gemma 4's
		// channel, llm's thinkTags). Anything else is text.
		close := ""
		if controlTag(name) {
			switch {
			case strings.HasSuffix(name, "_begin"):
				close = "<|" + strings.TrimSuffix(name, "_begin") + "_end|>"
			case name == "channel":
				close = "<channel|>"
			}
		}
		if close == "" {
			at = i + 2
			continue
		}
		e := strings.Index(rest, close)
		if e < 0 {
			return strings.TrimRight(s[:i], " \t\r\n")
		}
		s = s[:i] + rest[e+len(close):]
		at = i
	}
}

// controlTag reports whether tail is a control tag's name, whole or
// cut off by the stream's end: channel (Gemma 4's thought channel), or
// an identifier of _-joined parts whose last part starts begin or end,
// as tool_call_be is tool_call_begin cut short. A plain word is none of
// these: F#'s <|x and the tags a user writes stay text.
func controlTag(tail string) bool {
	if strings.HasPrefix("channel", tail) {
		return tail != ""
	}
	i := strings.LastIndexByte(tail, '_')
	if i < 0 || !tagName(tail) {
		return false
	}
	last := tail[i+1:]
	return strings.HasPrefix("begin", last) || strings.HasPrefix("end", last)
}

// tagName reports whether tail has a tag name's shape: a letter, then
// letters, digits and _.
func tagName(tail string) bool {
	for i, r := range tail {
		switch {
		case 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z':
		case i > 0 && (r == '_' || '0' <= r && r <= '9'):
		default:
			return false
		}
	}
	return tail != ""
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
// writes plain text instead, the text itself as content.
func PartialAnswer(text string) Answer {
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
		return Answer{Content: strings.TrimSpace(StripControl(strings.TrimSuffix(t, "```")))}
	}
	return Answer{Title: strings.TrimSpace(StripControl(partialField(t, "title"))), Content: strings.TrimLeft(StripControl(partialField(t, "content")), " \n")}
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
