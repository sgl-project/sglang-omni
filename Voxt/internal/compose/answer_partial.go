package compose

import (
	"encoding/json"
	"strings"
	"unicode/utf8"
)

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
