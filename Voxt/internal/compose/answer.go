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
	a.Title = strings.TrimSpace(strings.Trim(strings.TrimSpace(a.Title), `"“”#*`))
	a.Content = strings.TrimSpace(a.Content)
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
		return Answer{Content: strings.TrimSpace(strings.TrimSuffix(t, "```"))}
	}
	return Answer{Title: strings.TrimSpace(partialField(t, "title")), Content: strings.TrimLeft(partialField(t, "content"), " \n")}
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
