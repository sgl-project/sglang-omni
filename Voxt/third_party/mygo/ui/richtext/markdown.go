package richtext

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Markdown returns t as CommonMark with GitHub's extensions: headings,
// lists and task lists, quotations, fenced blocks of code, **bold**,
// *italic*, ~~strikethrough~~, `code`, links and images; underlines are
// HTML's <u>, which Markdown keeps. Body paragraphs are apart by a blank
// line, so that they stay paragraphs; alignment, fonts and colors have no
// Markdown and are left out.
func (t Text) Markdown() string {
	var b strings.Builder
	var paras []Paragraph
	for _, p := range t.Paragraphs() {
		paras = append(paras, p)
	}
	// numbers counts the items of the numbered lists open, by level.
	var numbers []int
	prevList := false
	for i := 0; i < len(paras); i++ {
		p := paras[i]
		st := p.Style
		if i > 0 && !(st.List != NoList && prevList) {
			b.WriteString("\n\n")
		} else if i > 0 {
			b.WriteByte('\n')
		}
		prevList = st.List != NoList
		if st.List == NoList {
			numbers = numbers[:0]
		}
		quote := ""
		if st.Quote {
			quote = "> "
		}
		switch {
		case st.CodeBlock:
			fence := "```"
			for k := i; k < len(paras) && paras[k].Style.CodeBlock; k++ {
				for strings.Contains(plainOf(paras[k]), fence) {
					fence += "`"
				}
			}
			b.WriteString(quote + fence + "\n")
			for ; i < len(paras) && paras[i].Style.CodeBlock; i++ {
				b.WriteString(quote + plainOf(paras[i]) + "\n")
			}
			i--
			b.WriteString(quote + fence)
			continue
		case st.Heading > 0:
			b.WriteString(quote + strings.Repeat("#", min(st.Heading, 6)) + " ")
		case st.List != NoList:
			level := st.Indent
			for len(numbers) <= level {
				numbers = append(numbers, 0)
			}
			numbers = numbers[:level+1]
			b.WriteString(quote + strings.Repeat("    ", level))
			switch st.List {
			case Numbers:
				numbers[level]++
				b.WriteString(strconv.Itoa(numbers[level]) + ". ")
			case Tasks:
				if st.Checked {
					b.WriteString("- [x] ")
				} else {
					b.WriteString("- [ ] ")
				}
				numbers[level] = 0
			default:
				b.WriteString("- ")
				numbers[level] = 0
			}
		default:
			b.WriteString(quote)
		}
		if st.Heading > 0 {
			// A heading's closing #s are not its text: escape them.
			var h strings.Builder
			writeMarkdownRuns(&h, p.Runs)
			s := h.String()
			if k := len(strings.TrimRight(s, "#")); k < len(s) {
				s = s[:k] + "\\" + s[k:]
			}
			b.WriteString(s)
			continue
		}
		writeMarkdownRuns(&b, p.Runs)
	}
	return b.String()
}

// plainOf returns the text of a paragraph.
func plainOf(p Paragraph) string {
	var b strings.Builder
	for _, r := range p.Runs {
		b.WriteString(r.Text)
	}
	return b.String()
}

// mark is a Markdown mark around runs: a link (by its URL), bold, italic,
// strikethrough or underline, from the outside in. Code is a span of each
// run, fenced as its backticks need.
type mark struct {
	kind byte // 'a', 'b', 'i', 's', 'u'
	url  string
}

func marksOf(a Attributes) []mark {
	var m []mark
	if a.Link != "" {
		m = append(m, mark{kind: 'a', url: a.Link})
	}
	if a.Bold() {
		m = append(m, mark{kind: 'b'})
	}
	if a.Italic {
		m = append(m, mark{kind: 'i'})
	}
	if a.Strikethrough {
		m = append(m, mark{kind: 's'})
	}
	if a.Underline {
		m = append(m, mark{kind: 'u'})
	}
	return m
}

// open returns the mark's opening, as HTML when html: what Markdown
// cannot open or close where it is, as bold "foo." before "bar".
func (m mark) open(html bool) string {
	switch m.kind {
	case 'a':
		return "["
	case 'b':
		if html {
			return "<strong>"
		}
		return "**"
	case 'i':
		if html {
			return "<em>"
		}
		return "*"
	case 's':
		if html {
			return "<del>"
		}
		return "~~"
	case 'u':
		return "<u>"
	}
	return ""
}

func (m mark) close(html bool) string {
	switch m.kind {
	case 'a':
		return "](" + markdownURL(m.url) + ")"
	case 'u':
		return "</u>"
	}
	if html {
		return "</" + m.open(true)[1:]
	}
	return m.open(false)
}

// writeMarkdownRuns writes runs with their marks, opening and closing
// them as they change, whitespace outside them so that they flank words
// as CommonMark wants. Where a delimiter cannot open or close (bold
// "foo." right before "bar" is not **foo.**bar), its mark is written as
// HTML, which Markdown keeps: the runs written are read back to check,
// and those still read otherwise are written in HTML whole. What would
// start a block at the start of the paragraph is escaped, as the text
// of an item or a heading too.
func writeMarkdownRuns(b *strings.Builder, runs []Run) {
	s, delims := renderRuns(runs, nil)
	if len(delims) > 0 && !readsAs(s, runs) {
		html := map[delimKey]bool{}
		for _, d := range delims {
			if !d.flanks(s) {
				html[d.key] = true
			}
		}
		if len(html) > 0 {
			s2, _ := renderRuns(runs, html)
			if readsAs(s2, runs) {
				b.WriteString(s2)
				return
			}
		}
		for _, d := range delims {
			html[d.key] = true
		}
		s, _ = renderRuns(runs, html)
	}
	b.WriteString(s)
}

// delimKey names a mark written in Markdown: the run it opens at and its
// kind.
type delimKey struct {
	run  int
	kind byte
}

// delim is where a mark's delimiters are in what renderRuns wrote.
type delim struct {
	key                 delimKey
	char                byte
	open, close, length int
}

// flanks reports whether the delimiters can open and close where they
// are, as CommonMark's rules of flanking say, the runs of the same
// character around them taken together.
func (d delim) flanks(s string) bool {
	return canFlank(s, d.open, d.length, d.char, true) && canFlank(s, d.close, d.length, d.char, false)
}

func canFlank(s string, at, n int, c byte, left bool) bool {
	from, to := at, at+n
	for from > 0 && s[from-1] == c {
		from--
	}
	for to < len(s) && s[to] == c {
		to++
	}
	before, after := ' ', ' '
	if from > 0 {
		before, _ = utf8.DecodeLastRuneInString(s[:from])
	}
	if to < len(s) {
		after, _ = utf8.DecodeRuneInString(s[to:])
	}
	if !left {
		before, after = after, before
	}
	// Left-flanking: not before whitespace, and not before punctuation
	// unless after whitespace or punctuation; right-flanking mirrored.
	return !unicode.IsSpace(after) && (!isMarkdownPunct(after) || unicode.IsSpace(before) || isMarkdownPunct(before))
}

func isMarkdownPunct(r rune) bool { return unicode.IsPunct(r) || unicode.IsSymbol(r) }

// renderRuns writes runs, the marks of html in HTML, and returns where
// the delimiters of the others are.
func renderRuns(runs []Run, html map[delimKey]bool) (string, []delim) {
	var b strings.Builder
	type opened struct {
		mark
		key delimKey
		at  int
	}
	var open []opened
	var delims []delim
	pending := ""
	first := true
	closeTo := func(n int) {
		for len(open) > n {
			o := open[len(open)-1]
			h := html[o.key]
			if c := o.close(h); !h && (o.kind == 'b' || o.kind == 'i' || o.kind == 's') {
				delims = append(delims, delim{key: o.key, char: c[0], open: o.at, close: b.Len(), length: len(c)})
			}
			b.WriteString(o.close(h))
			open = open[:len(open)-1]
		}
	}
	for i, r := range runs {
		if at := r.Attributes.Attachment; at != nil {
			closeTo(0)
			b.WriteString(pending)
			pending = ""
			b.WriteString("![" + escapeMarkdown(at.Name) + "](" + markdownURL(at.URL) + ")")
			first = false
			continue
		}
		core := strings.TrimFunc(r.Text, unicode.IsSpace)
		if core == "" {
			pending += r.Text
			continue
		}
		lead := r.Text[:strings.Index(r.Text, core)]
		trail := r.Text[len(lead)+len(core):]
		want := marksOf(r.Attributes)
		k := 0
		for k < len(open) && k < len(want) && open[k].mark == want[k] {
			k++
		}
		closeTo(k)
		if first {
			// Spaces at the start would make a block of code.
			lead = strings.NewReplacer(" ", "&#32;", "\t", "&#9;").Replace(pending + lead)
			pending = ""
		}
		b.WriteString(pending + lead)
		for _, m := range want[k:] {
			key := delimKey{i, m.kind}
			open = append(open, opened{m, key, b.Len()})
			b.WriteString(m.open(html[key]))
		}
		if r.Attributes.Code {
			b.WriteString(codeSpan(core))
		} else {
			s := escapeMarkdown(core)
			if first {
				s = escapeBlockStart(s)
			}
			if strings.HasSuffix(s, "!") {
				// Not an image with a link after it.
				s = s[:len(s)-1] + "\\!"
			}
			b.WriteString(s)
		}
		first = false
		pending = trail
	}
	closeTo(0)
	b.WriteString(pending)
	return b.String(), delims
}

// readsAs reports whether s reads back as runs, as far as Markdown keeps
// them: their text, and the marks of all but whitespace.
func readsAs(s string, runs []Run) bool {
	var got []Run
	for _, p := range ParseMarkdown(s).Paragraphs() {
		if got != nil {
			return false
		}
		got = append(p.Runs, Run{})[:len(p.Runs)]
	}
	return markKey(got) == markKey(runs)
}

// markKey returns the text of runs with the marks of each character
// Markdown keeps, but none of whitespace outside code, nor of spaces
// ending the paragraph, which it does not show.
func markKey(runs []Run) string {
	var b strings.Builder
	type marks struct {
		bold, italic, strike, under, code bool
		link                              string
		at                                Attachment
	}
	var all []rune
	var ms []marks
	for _, r := range runs {
		a := r.Attributes
		m := marks{a.Bold(), a.Italic, a.Strikethrough, a.Underline, a.Code, a.Link, Attachment{}}
		if a.Attachment != nil {
			m.at = Attachment{Name: a.Attachment.Name, URL: a.Attachment.URL}
		}
		for _, c := range r.Text {
			all = append(all, c)
			ms = append(ms, m)
		}
	}
	end := len(all)
	for end > 0 && all[end-1] == ' ' && !ms[end-1].code {
		end--
	}
	for i, c := range all[:end] {
		m := ms[i]
		if unicode.IsSpace(c) && !m.code {
			m = marks{}
		}
		fmt.Fprintf(&b, "%c%v\x00", c, m)
	}
	return b.String()
}

// codeSpan returns s as a code span, fenced by more backticks than any
// run of them in it.
func codeSpan(s string) string {
	longest, n := 0, 0
	for _, r := range s {
		if r == '`' {
			n++
			longest = max(longest, n)
		} else {
			n = 0
		}
	}
	fence := strings.Repeat("`", longest+1)
	if strings.HasPrefix(s, "`") || strings.HasSuffix(s, "`") {
		s = " " + s + " "
	}
	return fence + s + fence
}

// escapeMarkdown escapes what Markdown would read as marks.
func escapeMarkdown(s string) string {
	var b strings.Builder
	for i, r := range s {
		switch r {
		case '\\', '*', '_', '`', '[', ']', '<', '~':
			b.WriteByte('\\')
		case '&':
			// Not the start of an entity, as &amp; or &#38;.
			if i+1 < len(s) && (s[i+1] == '#' || unicode.IsLetter(rune(s[i+1]))) {
				b.WriteByte('\\')
			}
		}
		b.WriteRune(r)
	}
	return b.String()
}

// escapeBlockStart escapes the start of a body paragraph that Markdown
// would read as a heading, a list, a quotation or a rule.
func escapeBlockStart(s string) string {
	switch {
	case strings.HasPrefix(s, "#"), strings.HasPrefix(s, ">"), strings.HasPrefix(s, "-"), strings.HasPrefix(s, "+"), strings.HasPrefix(s, "="):
		return "\\" + s
	}
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i > 0 && i < len(s) && (s[i] == '.' || s[i] == ')') {
		return s[:i] + "\\" + s[i:]
	}
	return s
}

// markdownURL returns a URL as the destination of a link.
func markdownURL(u string) string {
	if strings.ContainsAny(u, " ()<>") {
		return "<" + strings.NewReplacer("<", "%3C", ">", "%3E").Replace(u) + ">"
	}
	return u
}
