package parser

import (
	"slices"
	"strings"
)

// Healing completes what the end of a streaming document leaves open in
// the inline content of its last block, as Vercel's remend does to the
// text, but from the parser's own delimiters, and only for that block:
// the source never changes. It returns the text to parse and the length
// of what came from the source, past which the text was added.

// heal returns s completed, and whether the tree the parser holds is
// already that of the text returned.
func (ip *inlineParser) heal(s string) (string, int, bool) {
	s = trimTrailingMarkers(s)
	from := len(s)
	// A link whose destination is being written: [text](http…
	s += pendingLink(s)
	// A code span not closed.
	if ticks := openCodeSpan(s); ticks > 0 {
		run := 0
		for run < len(s) && s[len(s)-1-run] == '`' {
			run++
		}
		if run >= ticks || run == 0 {
			run = 0
		}
		s += strings.Repeat("`", ticks-run)
	}
	// The closing tildes of strikethrough half written: ~~struck~.
	if strings.HasSuffix(s, "~") && !strings.HasSuffix(s, "~~") && strings.Contains(s, "~~") {
		s += "~"
	}
	// Emphasis, strikethrough and math not closed: the openers the parser
	// leaves, innermost first; and the text of a link being written shows
	// without its bracket.
	ip.run(s)
	type opener struct {
		at     int
		closer string
	}
	var open []opener
	for _, d := range ip.leftover {
		if d.c == '*' && wordInside(s, d.at, d.orig) || d.c == '~' && d.orig == 1 {
			continue // 2*3, or a tilde for "about", as ~200
		}
		open = append(open, opener{d.at, strings.Repeat(string(d.c), d.n)})
	}
	if ip.ext&ExtMath != 0 && ip.openMath >= 0 {
		open = append(open, opener{ip.openMath, "$$"})
	}
	if ip.lastB >= 0 {
		ip.hide = ip.nodes[ip.br[ip.lastB].node].start
	}
	if len(open) == 0 && ip.hide < 0 {
		return s, from, true
	}
	slices.SortStableFunc(open, func(a, b opener) int { return b.at - a.at })
	var b strings.Builder
	b.WriteString(s)
	for _, o := range open {
		b.WriteString(o.closer)
	}
	return b.String(), from, false
}

// wordInside reports whether a run of n delimiters at i has letters or
// digits of ASCII on both sides, as 2*3, which is not emphasis being
// written.
func wordInside(s string, i, n int) bool {
	if i == 0 || i+n >= len(s) {
		return false
	}
	isWord := func(c byte) bool { return isASCIILetter(c) || isASCIIDigit(c) }
	return isWord(s[i-1]) && isWord(s[i+n])
}

// trimTrailingMarkers drops what the end of s may be starting, with
// nothing after it yet: a run of delimiters opening emphasis, a code span
// or math, a bracket, or a backslash.
func trimTrailingMarkers(s string) string {
	for {
		t := strings.TrimRight(s, " \t")
		if t == "" {
			return t
		}
		c := t[len(t)-1]
		k := len(t)
		for k > 0 && t[k-1] == c {
			k--
		}
		run := len(t) - k
		opening := k == 0 || t[k-1] == ' ' || t[k-1] == '\t' || t[k-1] == '\n'
		switch {
		case strings.HasSuffix(t, "!["):
			t = t[:len(t)-2]
		case c == '[':
			t = t[:k]
		case (c == '*' || c == '_' || c == '~' || c == '$') && opening:
			t = t[:k]
		case c == '`' && !closesSpan(t[:k], run):
			t = t[:k]
		case c == '\\' && run%2 == 1:
			t = t[:len(t)-1]
		default:
			return t
		}
		s = t
	}
}

// closesSpan reports whether a run of n backticks after s closes a code
// span s opens.
func closesSpan(s string, n int) bool { return openCodeSpan(s) == n }

// pendingLink returns what closes the destination of a link being
// written at the end of s, after ](, or nothing: the parentheses it
// opened, and the link's, or > and ) for one in angle brackets.
func pendingLink(s string) string {
	k := strings.LastIndex(s, "](")
	if k < 0 || !strings.Contains(s[:k], "[") {
		return ""
	}
	// Not inside a code span.
	if strings.Count(s[:k], "`")%2 == 1 {
		return ""
	}
	rest := s[k+2:]
	if strings.HasPrefix(rest, "<") {
		if strings.ContainsAny(rest, ">\n") {
			return ""
		}
		return ">)"
	}
	depth := 0
	for i := 0; i < len(rest); i++ {
		switch rest[i] {
		case '\\':
			i++
		case '(':
			depth++
		case ')':
			if depth == 0 {
				return "" // the link is closed
			}
			depth--
		case ' ', '\t', '\n':
			return ""
		}
	}
	return strings.Repeat(")", depth+1)
}

// openCodeSpan returns the length of the run of backticks opening a code
// span that s leaves open, or 0.
func openCodeSpan(s string) int {
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '\\':
			i++
		case '`':
			j := i
			for j < len(s) && s[j] == '`' {
				j++
			}
			n := j - i
			// Its closer: the next run of as many.
			k := j
			found := false
			for k < len(s) {
				m := strings.IndexByte(s[k:], '`')
				if m < 0 {
					break
				}
				k += m
				e := k
				for e < len(s) && s[e] == '`' {
					e++
				}
				if e-k == n {
					found = true
					i = e - 1
					break
				}
				k = e
			}
			if !found {
				if j == len(s) {
					return 0 // a run at the very end opens nothing yet
				}
				return n
			}
		}
	}
	return 0
}
