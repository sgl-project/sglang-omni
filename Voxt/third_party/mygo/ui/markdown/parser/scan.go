package parser

import (
	"html"
	"strings"
	"unicode/utf8"
)

// scanATXStart returns the length of an ATX heading's opening sequence
// at i with the spaces after it, or 0.
func scanATXStart(ln string, i int) int {
	j := i
	for j < len(ln) && ln[j] == '#' && j-i < 7 {
		j++
	}
	n := j - i
	if n == 0 || n > 6 {
		return 0
	}
	if j == len(ln) {
		return n
	}
	if !isSpaceOrTab(ln[j]) {
		return 0
	}
	for j < len(ln) && isSpaceOrTab(ln[j]) {
		j++
	}
	return j - i
}

// scanOpenFence returns the length of an opening code fence at i, or 0.
func scanOpenFence(ln string, i int) int {
	c := ln[i]
	j := i
	for j < len(ln) && ln[j] == c {
		j++
	}
	n := j - i
	if n < 3 {
		return 0
	}
	if c == '`' && strings.IndexByte(ln[j:], '`') >= 0 {
		return 0
	}
	return n
}

// scanCloseFence returns the length of a closing code fence at i, or 0.
func scanCloseFence(ln string, i int) int {
	c := ln[i]
	if c != '`' && c != '~' {
		return 0
	}
	j := i
	for j < len(ln) && ln[j] == c {
		j++
	}
	n := j - i
	if n < 3 {
		return 0
	}
	for ; j < len(ln); j++ {
		if !isSpaceOrTab(ln[j]) {
			return 0
		}
	}
	return n
}

// scanSetextLine returns 1 for a line of =, 2 for one of -, or 0.
func scanSetextLine(ln string, i int) int {
	c := ln[i]
	j := i
	for j < len(ln) && ln[j] == c {
		j++
	}
	for ; j < len(ln); j++ {
		if !isSpaceOrTab(ln[j]) {
			return 0
		}
	}
	if c == '=' {
		return 1
	}
	return 2
}

// scanFootnoteDef returns the length of a footnote definition's start at
// i, [^label]: and the spaces after it, or 0.
func scanFootnoteDef(ln string, i int) int {
	if !strings.HasPrefix(ln[i:], "[^") {
		return 0
	}
	j := i + 2
	for j < len(ln) && ln[j] != ']' {
		switch ln[j] {
		case ' ', '\t', '\r', '\n', 0:
			return 0
		}
		j++
	}
	if j == i+2 || j+1 >= len(ln) || ln[j+1] != ':' {
		return 0
	}
	j += 2
	for j < len(ln) && isSpaceOrTab(ln[j]) {
		j++
	}
	return j - i
}

var htmlBlockTags = map[string]bool{
	"address": true, "article": true, "aside": true, "base": true, "basefont": true, "blockquote": true, "body": true,
	"caption": true, "center": true, "col": true, "colgroup": true, "dd": true, "details": true, "dialog": true, "dir": true,
	"div": true, "dl": true, "dt": true, "fieldset": true, "figcaption": true, "figure": true, "footer": true, "form": true,
	"frame": true, "frameset": true, "h1": true, "h2": true, "h3": true, "h4": true, "h5": true, "h6": true, "head": true,
	"header": true, "hr": true, "html": true, "iframe": true, "legend": true, "li": true, "link": true, "main": true,
	"menu": true, "menuitem": true, "nav": true, "noframes": true, "ol": true, "optgroup": true, "option": true, "p": true,
	"param": true, "search": true, "section": true, "summary": true, "table": true, "tbody": true, "td": true, "tfoot": true,
	"th": true, "thead": true, "title": true, "tr": true, "track": true, "ul": true,
}

// hasPrefixFold reports whether s starts with prefix, ignoring ASCII case.
func hasPrefixFold(s, prefix string) bool {
	return len(s) >= len(prefix) && strings.EqualFold(s[:len(prefix)], prefix)
}

// scanHTMLBlockStart returns the type (1 to 6) of an HTML block starting
// at i, or 0.
func scanHTMLBlockStart(ln string, i int) int {
	s := ln[i:]
	if len(s) < 2 || s[0] != '<' {
		return 0
	}
	for _, tag := range [...]string{"<script", "<pre", "<style", "<textarea"} {
		if hasPrefixFold(s, tag) {
			if len(s) == len(tag) || s[len(tag)] == ' ' || s[len(tag)] == '\t' || s[len(tag)] == '>' {
				return 1
			}
		}
	}
	switch {
	case strings.HasPrefix(s, "<!--"):
		return 2
	case strings.HasPrefix(s, "<?"):
		return 3
	case strings.HasPrefix(s, "<![CDATA["):
		return 5
	case len(s) > 2 && s[1] == '!' && isASCIILetter(s[2]):
		return 4
	}
	j := 1
	if s[j] == '/' {
		j++
	}
	k := j
	for k < len(s) && (isASCIILetter(s[k]) || isASCIIDigit(s[k])) {
		k++
	}
	if k == j || !htmlBlockTags[strings.ToLower(s[j:k])] {
		return 0
	}
	if k == len(s) || s[k] == ' ' || s[k] == '\t' || s[k] == '>' || strings.HasPrefix(s[k:], "/>") {
		return 6
	}
	return 0
}

// scanHTMLBlockStart7 returns 7 for a line that is a complete open or
// closing tag, other than script, style, pre and textarea, followed only
// by spaces, or 0.
func scanHTMLBlockStart7(ln string, i int) int {
	s := ln[i:]
	var n int
	if strings.HasPrefix(s, "</") {
		n = scanClosingTag(s, 0)
	} else {
		n = scanOpenTag(s, 0)
		if n > 0 {
			name := s[1:]
			k := 0
			for k < len(name) && (isASCIILetter(name[k]) || isASCIIDigit(name[k]) || name[k] == '-') {
				k++
			}
			switch strings.ToLower(name[:k]) {
			case "script", "style", "pre", "textarea":
				return 0
			}
		}
	}
	if n == 0 || strings.Trim(s[n:], " \t") != "" {
		return 0
	}
	return 7
}

// htmlBlockEnds reports whether a line ends an HTML block of type t.
func htmlBlockEnds(t int, s string) bool {
	switch t {
	case 1:
		l := strings.ToLower(s)
		return strings.Contains(l, "</script>") || strings.Contains(l, "</pre>") || strings.Contains(l, "</style>") || strings.Contains(l, "</textarea>")
	case 2:
		return strings.Contains(s, "-->")
	case 3:
		return strings.Contains(s, "?>")
	case 4:
		return strings.Contains(s, ">")
	case 5:
		return strings.Contains(s, "]]>")
	}
	return false
}

// The scanners of inline HTML (CommonMark's raw HTML) return the length
// of what they find at i, or 0.

func scanTagName(s string, i int) int {
	if i >= len(s) || !isASCIILetter(s[i]) {
		return 0
	}
	j := i + 1
	for j < len(s) && (isASCIILetter(s[j]) || isASCIIDigit(s[j]) || s[j] == '-') {
		j++
	}
	return j - i
}

// skipTagSpace skips whitespace within a tag, at most one line ending.
func skipTagSpace(s string, i int) int {
	for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') {
		i++
	}
	return i
}

func scanOpenTag(s string, i int) int {
	if i >= len(s) || s[i] != '<' {
		return 0
	}
	n := scanTagName(s, i+1)
	if n == 0 {
		return 0
	}
	j := i + 1 + n
	for {
		k := skipTagSpace(s, j)
		if k < len(s) && s[k] == '>' {
			return k + 1 - i
		}
		if k+1 < len(s) && s[k] == '/' && s[k+1] == '>' {
			return k + 2 - i
		}
		if k == j {
			return 0 // attributes need whitespace before them
		}
		// An attribute name.
		if k >= len(s) || !(isASCIILetter(s[k]) || s[k] == '_' || s[k] == ':') {
			return 0
		}
		k++
		for k < len(s) && (isASCIILetter(s[k]) || isASCIIDigit(s[k]) || strings.IndexByte("_.:-", s[k]) >= 0) {
			k++
		}
		// An optional value.
		v := skipTagSpace(s, k)
		if v < len(s) && s[v] == '=' {
			v = skipTagSpace(s, v+1)
			if v >= len(s) {
				return 0
			}
			switch s[v] {
			case '"', '\'':
				e := strings.IndexByte(s[v+1:], s[v])
				if e < 0 {
					return 0
				}
				k = v + 1 + e + 1
			default:
				e := v
				for e < len(s) && strings.IndexByte(" \t\n\r\"'=<>`", s[e]) < 0 {
					e++
				}
				if e == v {
					return 0
				}
				k = e
			}
		}
		j = k
	}
}

func scanClosingTag(s string, i int) int {
	if !strings.HasPrefix(s[i:], "</") {
		return 0
	}
	n := scanTagName(s, i+2)
	if n == 0 {
		return 0
	}
	k := skipTagSpace(s, i+2+n)
	if k < len(s) && s[k] == '>' {
		return k + 1 - i
	}
	return 0
}

// scanHTMLTag returns the length of the raw HTML at i: a tag, a comment,
// a processing instruction, a declaration or a CDATA section, or 0. The
// scanners noted in no remember that a kind found no end after a point,
// so that unclosed comments repeated do not take quadratic time.
func scanHTMLTag(s string, i int, no *htmlMemo) int {
	r := s[i:]
	if len(r) < 2 {
		return 0
	}
	switch {
	case r[1] == '/':
		return scanClosingTag(s, i)
	case r[1] == '?':
		return no.until(s, i, 2, "?>", &no.pi)
	case strings.HasPrefix(r, "<!--"):
		if strings.HasPrefix(r, "<!-->") {
			return 5
		}
		if strings.HasPrefix(r, "<!--->") {
			return 6
		}
		return no.until(s, i, 4, "-->", &no.comment)
	case strings.HasPrefix(r, "<![CDATA["):
		return no.until(s, i, 9, "]]>", &no.cdata)
	case r[1] == '!':
		if len(r) > 2 && isASCIILetter(r[2]) {
			return no.until(s, i, 3, ">", &no.decl)
		}
		return 0
	}
	return scanOpenTag(s, i)
}

type htmlMemo struct {
	pi, comment, cdata, decl int
	steps                    int // the bytes searched, as Document.steps
}

func (m *htmlMemo) reset() { *m = htmlMemo{-1, -1, -1, -1, 0} }

// until returns the length up to and including end, searched from
// i+skip, or 0; *none notes the offset after which none is.
func (m *htmlMemo) until(s string, i, skip int, end string, none *int) int {
	if *none >= 0 && i+skip >= *none {
		return 0
	}
	k := strings.Index(s[i+skip:], end)
	if k < 0 {
		m.steps += len(s) - i - skip
		*none = i + skip
		return 0
	}
	m.steps += k
	return skip + k + len(end)
}

// scanDelimiterRow reads a table's delimiter row: cells of dashes, with
// a colon at either end for alignment, between pipes.
func scanDelimiterRow(line string) ([]Align, bool) {
	line = strings.TrimRight(line, " \t")
	if line == "" {
		return nil, false
	}
	pipes := strings.Count(line, "|")
	s := line
	if s[0] == '|' {
		s = s[1:]
	}
	if s != "" && s[len(s)-1] == '|' && pipes > 0 && (len(line) > 1 || line[0] != '|') {
		if len(s) > 0 {
			s = s[:len(s)-1]
		}
	}
	var aligns []Align
	for part := range strings.SplitSeq(s, "|") {
		part = strings.Trim(part, " \t")
		left := strings.HasPrefix(part, ":")
		right := strings.HasSuffix(part, ":") && len(part) > 1
		if left {
			part = part[1:]
		}
		if right {
			part = part[:len(part)-1]
		}
		if part == "" || strings.Trim(part, "-") != "" {
			return nil, false
		}
		a := AlignNone
		switch {
		case left && right:
			a = AlignCenter
		case left:
			a = AlignLeft
		case right:
			a = AlignRight
		}
		aligns = append(aligns, a)
	}
	if len(aligns) == 0 {
		return nil, false
	}
	return aligns, true
}

// splitRow splits a table row into its cells' byte ranges, trimmed: an
// optional pipe, then cells between pipes, whose \| do not split.
func splitRow(line string) [][2]int {
	var cells [][2]int
	i := 0
	if i < len(line) && line[i] == '|' {
		i++
		for i < len(line) && isSpaceOrTab(line[i]) {
			i++
		}
	}
	for i < len(line) {
		j := i
		for j < len(line) && line[j] != '|' {
			if line[j] == '\\' && j+1 < len(line) && isEscapable(line[j+1]) {
				j++
			}
			j++
		}
		j = min(j, len(line))
		pipe := j < len(line)
		a, b := i, j
		for a < b && isSpaceOrTab(line[a]) {
			a++
		}
		for b > a && isSpaceOrTab(line[b-1]) {
			b--
		}
		if j > i || pipe {
			cells = append(cells, [2]int{a, b})
		}
		if !pipe {
			break
		}
		i = j + 1
		for i < len(line) && isSpaceOrTab(line[i]) {
			i++
		}
	}
	return cells
}

// rowHasCells reports whether a line makes a table row of cells.
func rowHasCells(line string) bool {
	return len(splitRow(line)) > 0
}

// isEscapable reports whether a backslash escapes c: ASCII punctuation.
func isEscapable(c byte) bool {
	return c >= '!' && c <= '/' || c >= ':' && c <= '@' || c >= '[' && c <= '`' || c >= '{' && c <= '~'
}

// unescapeString resolves backslash escapes and entities.
func unescapeString(s string) string {
	if strings.IndexByte(s, '\\') < 0 && strings.IndexByte(s, '&') < 0 {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '\\' && i+1 < len(s) && isEscapable(s[i+1]):
			b.WriteByte(s[i+1])
			i++
		case c == '&':
			if n, text := scanEntity(s, i); n > 0 {
				b.WriteString(text)
				i += n - 1
			} else {
				b.WriteByte(c)
			}
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

// Entity reads the entity or numeric character reference s starts with,
// as &amp; or &#35;, returning its length and the text it stands for, or
// 0 when s starts with none that Markdown reads.
func Entity(s string) (n int, text string) { return scanEntity(s, 0) }

// scanEntity reads an entity or a numeric character reference at i,
// returning its length and its text, or 0.
func scanEntity(s string, i int) (int, string) {
	j := i + 1
	if j < len(s) && s[j] == '#' {
		j++
		hex := j < len(s) && (s[j] == 'x' || s[j] == 'X')
		if hex {
			j++
		}
		start, v := j, 0
		for j < len(s) {
			c := s[j]
			var d int
			switch {
			case isASCIIDigit(c):
				d = int(c - '0')
			case hex && c|0x20 >= 'a' && c|0x20 <= 'f':
				d = int(c|0x20-'a') + 10
			default:
				goto end
			}
			if v < 0x110000 {
				if hex {
					v = v*16 + d
				} else {
					v = v*10 + d
				}
			}
			j++
		}
	end:
		n := j - start
		if n == 0 || hex && n > 6 || !hex && n > 7 || j >= len(s) || s[j] != ';' {
			return 0, ""
		}
		r := rune(v)
		if v == 0 || v > utf8.MaxRune || v >= 0xD800 && v <= 0xDFFF {
			r = utf8.RuneError
		}
		return j + 1 - i, string(r)
	}
	start := j
	for j < len(s) && j-start <= 32 && (isASCIILetter(s[j]) || isASCIIDigit(s[j])) {
		j++
	}
	if j == start || j >= len(s) || s[j] != ';' || !isASCIILetter(s[start]) {
		return 0, ""
	}
	ent := s[i : j+1]
	text := html.UnescapeString(ent)
	// A legacy name's prefix decodes, leaving the rest: &notit; is ¬it;.
	if text == ent || strings.HasSuffix(text, ";") && ent != "&semi;" {
		return 0, ""
	}
	return j + 1 - i, text
}

// literalToSource returns the offset in source of byte j of text, the
// literal it reads as: escapes and entities decoded, the rest as it is.
func literalToSource(source, text string, j int) int {
	if source == text {
		return min(j, len(source))
	}
	i, k := 0, 0
	for i < len(source) && k < j {
		switch c := source[i]; {
		case c == '\\' && i+1 < len(source) && isEscapable(source[i+1]) && k < len(text) && text[k] == source[i+1]:
			i, k = i+2, k+1
		case c == '&':
			if n, lit := scanEntity(source, i); n > 0 && strings.HasPrefix(text[k:], lit) {
				i, k = i+n, k+len(lit)
				continue
			}
			i, k = i+1, k+1
		default:
			i, k = i+1, k+1
		}
	}
	return i
}
