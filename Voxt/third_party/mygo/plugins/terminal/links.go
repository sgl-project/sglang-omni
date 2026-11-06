package terminal

import (
	"regexp"
	"strings"
)

// urlPattern finds the URLs programs print as text, which a Command+click
// opens as it does hyperlinks.
var urlPattern = regexp.MustCompile(`(?:https?|ftp|file)://[^\s"'<>` + "`" + `\x00-\x1f]+|mailto:[^\s"'<>` + "`" + `]+`)

// pathPattern finds the file paths programs print as text: absolute, from
// ~, ./ or ../, with a slash, or a file name with a :line; each may end in
// :line or :line:col.
var pathPattern = regexp.MustCompile(`(?:(?:~|\.{1,2})?/[^\s"'<>` + "`" + `\x00-\x1f:]+|[\w.@+-]+(?:/[\w.@+-]+)+/?)(?::\d+(?::\d+)?)?|[\w@+-][\w.@+-]*\.\w+:\d+(?::\d+)?`)

// urlAt returns the URL of text, a row whose runes are in columns cols, at
// column col, or "".
func urlAt(text []rune, cols []int, col int) string {
	return matchAt(urlPattern, text, cols, col)
}

// pathAt returns the file path of text at column col, as urlAt, or "".
func pathAt(text []rune, cols []int, col int) string {
	return matchAt(pathPattern, text, cols, col)
}

func matchAt(re *regexp.Regexp, text []rune, cols []int, col int) string {
	s := string(text)
	for _, m := range re.FindAllStringIndex(s, -1) {
		u := trimURL(s[m[0]:m[1]])
		if u == "" {
			continue
		}
		start := len([]rune(s[:m[0]]))
		end := start + len([]rune(u))
		if start < len(cols) && cols[start] <= col && col <= cols[end-1] {
			return u
		}
	}
	return ""
}

// trimURL trims what likely ends the sentence around a URL rather than the
// URL: punctuation, and closing brackets that the URL did not open.
func trimURL(u string) string {
	for len(u) > 0 {
		switch last := u[len(u)-1]; last {
		case '.', ',', ';', ':', '!', '?':
			u = u[:len(u)-1]
			continue
		case ')', ']', '}':
			open := map[byte]string{')': "(", ']': "[", '}': "{"}[last]
			if strings.Count(u, open) < strings.Count(u, string(last)) {
				u = u[:len(u)-1]
				continue
			}
		}
		return u
	}
	return u
}
