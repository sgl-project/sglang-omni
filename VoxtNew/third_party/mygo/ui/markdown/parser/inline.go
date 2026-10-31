package parser

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// InlineKind is what an inline is.
type InlineKind uint8

// Kinds of inlines.
const (
	// Text is literal text, in Text.
	Text InlineKind = iota
	// SoftBreak is a line ending within a paragraph, shown as a space.
	SoftBreak
	// LineBreak is a hard line break.
	LineBreak
	// Code is a code span, its text in Text.
	Code
	Emphasis
	Strong
	Strikethrough
	// Link holds the inlines of its text, to Dest, titled Title.
	Link
	// Image holds the inlines of its description, from Dest.
	Image
	// HTML is raw HTML, in Text.
	HTML
	// Math is TeX, in Text, Display or within the line.
	Math
	// FootnoteRef refers to the footnote labelled Text.
	FootnoteRef
)

var inlineNames = [...]string{"text", "soft break", "line break", "code", "emphasis", "strong", "strikethrough", "link", "image", "HTML", "math", "footnote reference"}

func (k InlineKind) String() string {
	if int(k) < len(inlineNames) {
		return inlineNames[k]
	}
	return "unknown"
}

// Inline is an inline of a block's content: inlines are in document
// order, an inline's children right after it, up to its Next.
type Inline struct {
	Kind InlineKind
	// Text is the literal of Text, Code, HTML and Math, and the label of
	// a FootnoteRef.
	Text string
	// Dest and Title are a link's or an image's.
	Dest, Title string
	// Start and End are the offsets of the inline's bytes in the source.
	Start, End int
	// Next is the index after the inline and its children.
	Next int
	// Auto tells a link written as its address, as <https://…> or
	// www.example.com.
	Auto bool
	// Display tells math shown on its own lines, between $$ or \[ \].
	Display bool
	// Pending tells, while the document is streaming, a link or an image
	// whose destination is still being written.
	Pending bool
}

// Inlines returns the inlines of block i, a paragraph, a heading or a
// table cell, parsing them the first time.
func (d *Document) Inlines(i int) []Inline {
	b := &d.blocks[i]
	if b.inlOK {
		return b.inl
	}
	switch b.Kind {
	case Paragraph, Heading, TableCell:
	default:
		return nil
	}
	b.inlOK = true
	text, _ := d.contentOf(i)
	if b.segB > b.segA && i != d.tail {
		at := int(d.segs[b.segA].start)
		for _, o := range d.oldInl {
			if o.at == at && o.text == text && !o.healed && slices.Equal(o.segs, d.segsOf(b)) {
				b.inl, b.labels, b.rev = o.inl, o.labels, o.rev
				return b.inl
			}
		}
	}
	b.healed = d.streaming && i == d.tail
	if b.healed {
		d.healedAt = i
	}
	b.inl, b.labels = d.inp.parse(d, i, b.healed)
	d.revs++
	b.rev = d.revs
	return b.inl
}

// inode is an inline while the parser builds the tree.
type inode struct {
	kind        InlineKind
	text        string
	dest, title string
	start, end  int
	parent      int32
	first, last int32
	prev, next  int32
	auto        bool
	display     bool
	plain       bool // a text node others may merge into
}

type delim struct {
	c                 byte
	n, orig           int
	node              int32
	prev, next        int32
	canOpen, canClose bool
	at                int
}

type bracket struct {
	node         int32
	prev         int32
	prevDelim    int32
	index        int
	image        bool
	bracketAfter bool
}

const maxBackticks = 1000

type inlineParser struct {
	d     *Document
	ext   Extensions
	s     string
	pos   int
	nodes []inode
	dl    []delim
	last  int32 // the last delimiter
	br    []bracket
	lastB int32 // the last bracket
	// linked is the number of brackets when a link last closed: those
	// before it, but for images, open no link, as links do not nest.
	linked int32

	backticks        [maxBackticks + 1]int
	scannedBackticks bool
	html             htmlMemo
	noMath           int // no closing $$ after this offset
	// noDollar, noParen and noBracket are the offsets after which no $,
	// \) or \] closes math, so that openers repeated stay linear.
	noDollar, noParen, noBracket int
	// openMath is where $$ opens math that nothing closes, or -1, and hide
	// a bracket healing leaves out, as the text of a link being written.
	openMath, hide int
	labels         []string
	// leftover are the openers no closer matched, for healing.
	leftover []delim
	probe    bool
	out      []Inline
}

func (ip *inlineParser) reset() {}

// parse parses the inlines of block i; heal completes what the end of
// the block leaves open, as the document streams.
func (ip *inlineParser) parse(d *Document, i int, heal bool) ([]Inline, []string) {
	text, _ := d.contentOf(i)
	// Paragraphs and headings are trimmed of spaces and tabs.
	a, b := 0, len(text)
	for a < b && (text[a] == ' ' || text[a] == '\t' || text[a] == '\n') {
		a++
	}
	for b > a && (text[b-1] == ' ' || text[b-1] == '\t' || text[b-1] == '\n') {
		b--
	}
	s := text[a:b]
	ip.d, ip.ext = d, d.ext
	ip.labels = nil
	ip.hide = -1
	healFrom := len(s)
	built := false
	if heal {
		s, healFrom, built = ip.heal(s)
	}
	if !built {
		ip.run(s)
	}
	out := ip.flatten(d, i, a, healFrom)
	labels := ip.labels
	ip.labels = nil
	return out, labels
}

// run parses s into the node tree, node 0 its root.
func (ip *inlineParser) run(s string) {
	ip.s, ip.pos = s, 0
	ip.nodes = append(ip.nodes[:0], inode{parent: -1, first: -1, last: -1, prev: -1, next: -1, end: len(s)})
	ip.dl = ip.dl[:0]
	ip.br = ip.br[:0]
	ip.last, ip.lastB, ip.linked = -1, -1, 0
	ip.scannedBackticks = false
	ip.html.reset()
	ip.noMath = -1
	ip.noDollar, ip.noParen, ip.noBracket, ip.openMath = -1, -1, -1, -1
	ip.leftover = ip.leftover[:0]
	for ip.pos < len(s) {
		ip.parseInline()
	}
	ip.processEmphasis(0, true)
	ip.joinTexts(0)
	if ip.ext&ExtAutolinks != 0 && strings.IndexByte(s, '@') >= 0 {
		ip.emailLinks(0, false)
	}
	ip.d.steps += ip.html.steps
}

func (ip *inlineParser) newNode(kind InlineKind, text string, start, end int) int32 {
	ip.nodes = append(ip.nodes, inode{kind: kind, text: text, start: start, end: end, parent: -1, first: -1, last: -1, prev: -1, next: -1})
	return int32(len(ip.nodes) - 1)
}

func (ip *inlineParser) appendChild(parent, child int32) {
	n := &ip.nodes[child]
	n.parent = parent
	p := &ip.nodes[parent]
	n.prev, n.next = p.last, -1
	if p.last >= 0 {
		ip.nodes[p.last].next = child
	} else {
		p.first = child
	}
	p.last = child
}

func (ip *inlineParser) unlink(c int32) {
	n := &ip.nodes[c]
	if n.prev >= 0 {
		ip.nodes[n.prev].next = n.next
	} else if n.parent >= 0 {
		ip.nodes[n.parent].first = n.next
	}
	if n.next >= 0 {
		ip.nodes[n.next].prev = n.prev
	} else if n.parent >= 0 {
		ip.nodes[n.parent].last = n.prev
	}
	n.parent, n.prev, n.next = -1, -1, -1
}

func (ip *inlineParser) insertAfter(c, sib int32) {
	ip.unlink(sib)
	n := &ip.nodes[c]
	s := &ip.nodes[sib]
	s.parent, s.prev, s.next = n.parent, c, n.next
	if n.next >= 0 {
		ip.nodes[n.next].prev = sib
	} else if n.parent >= 0 {
		ip.nodes[n.parent].last = sib
	}
	ip.nodes[c].next = sib
}

func (ip *inlineParser) insertBefore(c, sib int32) {
	ip.unlink(sib)
	n := &ip.nodes[c]
	s := &ip.nodes[sib]
	s.parent, s.next, s.prev = n.parent, c, n.prev
	if n.prev >= 0 {
		ip.nodes[n.prev].next = sib
	} else if n.parent >= 0 {
		ip.nodes[n.parent].first = sib
	}
	ip.nodes[c].prev = sib
}

// addText adds the subject's [a, b) as text, merged into the text before
// it when that ends at a.
func (ip *inlineParser) addText(a, b int) {
	if a >= b {
		return
	}
	if l := ip.nodes[0].last; l >= 0 {
		n := &ip.nodes[l]
		if n.kind == Text && n.plain && n.end == a && n.text == ip.s[n.start:n.end] {
			n.end = b
			n.text = ip.s[n.start:b]
			return
		}
	}
	c := ip.newNode(Text, ip.s[a:b], a, b)
	ip.nodes[c].plain = true
	ip.appendChild(0, c)
}

// addLiteral adds text that is not the subject's bytes [a, b), as an
// escaped character's.
func (ip *inlineParser) addLiteral(text string, a, b int) int32 {
	c := ip.newNode(Text, text, a, b)
	ip.appendChild(0, c)
	return c
}

var special = func() (t [256]bool) {
	for _, c := range "\n\\`*_[]!<&~$:w" {
		t[c] = true
	}
	return
}()

func (ip *inlineParser) parseInline() {
	s := ip.s
	c := s[ip.pos]
	switch c {
	case '\n':
		ip.parseNewline()
	case '\\':
		ip.parseBackslash()
	case '`':
		ip.parseBackticks()
	case '*', '_':
		ip.handleDelim(c)
	case '~':
		if ip.ext&ExtStrikethrough != 0 {
			ip.handleDelim(c)
		} else {
			ip.addText(ip.pos, ip.pos+1)
			ip.pos++
		}
	case '[':
		text := "["
		if ip.pos == ip.hide {
			text = ""
		}
		n := ip.addLiteral(text, ip.pos, ip.pos+1)
		ip.addBracket(n, ip.pos+1, false)
		ip.pos++
	case '!':
		if ip.pos+1 < len(s) && s[ip.pos+1] == '[' {
			text := "!["
			if ip.pos == ip.hide {
				text = ""
			}
			n := ip.addLiteral(text, ip.pos, ip.pos+2)
			ip.addBracket(n, ip.pos+2, true)
			ip.pos += 2
		} else {
			ip.addText(ip.pos, ip.pos+1)
			ip.pos++
		}
	case ']':
		ip.parseCloseBracket()
	case '<':
		ip.parseAngle()
	case '&':
		if n, text := scanEntity(s, ip.pos); n > 0 {
			ip.addLiteral(text, ip.pos, ip.pos+n)
			ip.pos += n
		} else {
			ip.addText(ip.pos, ip.pos+1)
			ip.pos++
		}
	case '$':
		if ip.ext&ExtMath == 0 || !ip.parseDollar() {
			ip.addText(ip.pos, ip.pos+1)
			ip.pos++
		}
	case ':':
		if ip.ext&ExtAutolinks == 0 || !ip.urlLink() {
			ip.addText(ip.pos, ip.pos+1)
			ip.pos++
		}
	case 'w':
		if ip.ext&ExtAutolinks == 0 || !ip.wwwLink() {
			ip.parseString()
		}
	default:
		ip.parseString()
	}
}

// parseString adds the text up to the next byte that may start syntax.
func (ip *inlineParser) parseString() {
	s := ip.s
	j := ip.pos + 1
	for j < len(s) {
		c := s[j]
		if special[c] {
			if c == 'w' && (!strings.HasPrefix(s[j:], "www.") || ip.ext&ExtAutolinks == 0) {
				j++
				continue
			}
			if c == ':' && (!strings.HasPrefix(s[j:], "://") || ip.ext&ExtAutolinks == 0) {
				j++
				continue
			}
			if c == '$' && ip.ext&ExtMath == 0 {
				j++
				continue
			}
			if c == '~' && ip.ext&ExtStrikethrough == 0 {
				j++
				continue
			}
			break
		}
		j++
	}
	ip.d.steps += j - ip.pos
	ip.addText(ip.pos, j)
	ip.pos = j
}

func (ip *inlineParser) parseNewline() {
	ip.pos++
	hard := false
	if l := ip.nodes[0].last; l >= 0 {
		n := &ip.nodes[l]
		if n.kind == Text && strings.HasSuffix(n.text, " ") {
			t := strings.TrimRight(n.text, " ")
			hard = len(n.text)-len(t) >= 2
			n.text = t
			if n.plain {
				n.end = n.start + len(t)
			}
		}
	}
	kind := SoftBreak
	if hard {
		kind = LineBreak
	}
	c := ip.newNode(kind, "", ip.pos-1, ip.pos)
	ip.appendChild(0, c)
	for ip.pos < len(ip.s) && (ip.s[ip.pos] == ' ' || ip.s[ip.pos] == '\t') {
		ip.pos++
	}
}

func (ip *inlineParser) parseBackslash() {
	s := ip.s
	ip.pos++
	if ip.pos < len(s) {
		switch c := s[ip.pos]; {
		case c == '\n':
			ip.pos++
			n := ip.newNode(LineBreak, "", ip.pos-2, ip.pos)
			ip.appendChild(0, n)
			for ip.pos < len(s) && (s[ip.pos] == ' ' || s[ip.pos] == '\t') {
				ip.pos++
			}
			return
		case (c == '(' || c == '[') && ip.ext&ExtMath != 0:
			end, none := `\)`, &ip.noParen
			if c == '[' {
				end, none = `\]`, &ip.noBracket
			}
			k := -1
			if *none < 0 || ip.pos < *none {
				if k = strings.Index(s[ip.pos+1:], end); k < 0 {
					*none = ip.pos
					ip.d.steps += len(s) - ip.pos - 1
				} else {
					ip.d.steps += k
				}
			}
			if k > 0 {
				tex := s[ip.pos+1 : ip.pos+1+k]
				n := ip.newNode(Math, tex, ip.pos-1, ip.pos+1+k+2)
				ip.nodes[n].display = c == '['
				ip.appendChild(0, n)
				ip.pos += 1 + k + 2
				return
			}
			ip.addLiteral(s[ip.pos:ip.pos+1], ip.pos-1, ip.pos+1)
			ip.pos++
			return
		case isEscapable(c):
			ip.addLiteral(s[ip.pos:ip.pos+1], ip.pos-1, ip.pos+1)
			ip.pos++
			return
		}
	}
	ip.addText(ip.pos-1, ip.pos)
}

func (ip *inlineParser) parseBackticks() {
	s := ip.s
	start := ip.pos
	for ip.pos < len(s) && s[ip.pos] == '`' {
		ip.pos++
	}
	n := ip.pos - start
	after := ip.pos
	if end := ip.scanCloseBackticks(n); end > 0 {
		content := s[after : end-n]
		if strings.IndexByte(content, '\n') >= 0 {
			content = strings.ReplaceAll(content, "\n", " ")
		}
		if len(content) >= 2 && content[0] == ' ' && content[len(content)-1] == ' ' && strings.Trim(content, " ") != "" {
			content = content[1 : len(content)-1]
		}
		c := ip.newNode(Code, content, start, end)
		ip.appendChild(0, c)
		ip.pos = end
		return
	}
	ip.pos = after
	ip.addText(start, after)
}

// scanCloseBackticks returns the offset after a run of n backticks from
// pos, or 0, remembering the last run of each length so that unmatched
// openers do not take quadratic time, as cmark does.
func (ip *inlineParser) scanCloseBackticks(n int) int {
	if n > maxBackticks {
		return 0
	}
	if ip.scannedBackticks && ip.backticks[n] <= ip.pos {
		return 0
	}
	s := ip.s
	i := ip.pos
	for {
		k := strings.IndexByte(s[i:], '`')
		if k < 0 {
			break
		}
		i += k
		j := i
		for j < len(s) && s[j] == '`' {
			j++
		}
		m := j - i
		if m <= maxBackticks {
			ip.backticks[m] = i
		}
		if m == n {
			ip.d.steps += j - ip.pos
			return j
		}
		i = j
	}
	ip.d.steps += len(s) - ip.pos
	ip.scannedBackticks = true
	return 0
}

// parseDollar reads math between $$ (or $ with DollarMath) at pos.
func (ip *inlineParser) parseDollar() bool {
	s := ip.s
	start := ip.pos
	if strings.HasPrefix(s[start:], "$$") {
		if ip.noMath >= 0 && start >= ip.noMath {
			return false
		}
		k := strings.Index(s[start+2:], "$$")
		if k < 0 {
			ip.d.steps += len(s) - start - 2
			ip.noMath = start
			if ip.openMath < 0 {
				ip.openMath = start
			}
			return false
		}
		ip.d.steps += k
		if strings.Trim(s[start+2:start+2+k], " \t\n") == "" {
			return false
		}
		n := ip.newNode(Math, s[start+2:start+2+k], start, start+2+k+2)
		ip.nodes[n].display = true
		ip.appendChild(0, n)
		ip.pos = start + 2 + k + 2
		return true
	}
	if ip.ext&ExtDollarMath == 0 {
		return false
	}
	// A dollar followed by a space or a digit is money, not math.
	if start+1 >= len(s) || s[start+1] == ' ' || s[start+1] == '\t' || s[start+1] == '\n' || isASCIIDigit(s[start+1]) {
		return false
	}
	if ip.noDollar >= 0 && start >= ip.noDollar {
		return false
	}
	j := start + 1
	defer func() {
		ip.d.steps += j - start
		if ip.pos == start {
			ip.noDollar = start
		}
	}()
	for ; j < len(s); j++ {
		switch s[j] {
		case '\\':
			j++
		case '\n':
			if j+1 < len(s) && s[j+1] == '\n' {
				return false
			}
		case '$':
			if s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\n' || j+1 < len(s) && isASCIIDigit(s[j+1]) {
				continue
			}
			n := ip.newNode(Math, s[start+1:j], start, j+1)
			ip.appendChild(0, n)
			ip.pos = j + 1
			return true
		}
	}
	return false
}

// parseAngle reads an autolink or raw HTML at <.
func (ip *inlineParser) parseAngle() {
	s := ip.s
	if n, dest, email := scanAutolink(s, ip.pos); n > 0 {
		inner := s[ip.pos+1 : ip.pos+n-1]
		l := ip.newNode(Link, "", ip.pos, ip.pos+n)
		ip.nodes[l].dest = dest
		ip.nodes[l].auto = true
		_ = email
		ip.appendChild(0, l)
		t := ip.newNode(Text, inner, ip.pos+1, ip.pos+n-1)
		ip.appendChild(l, t)
		ip.pos += n
		return
	}
	if n := scanHTMLTag(s, ip.pos, &ip.html); n > 0 {
		h := ip.newNode(HTML, s[ip.pos:ip.pos+n], ip.pos, ip.pos+n)
		ip.appendChild(0, h)
		ip.pos += n
		return
	}
	ip.addText(ip.pos, ip.pos+1)
	ip.pos++
}

// scanAutolink reads <scheme:…> or <email> at i: its length, and its
// destination.
func scanAutolink(s string, i int) (int, string, bool) {
	j := i + 1
	// A URI: a scheme of 2 to 32 characters, a colon, no spaces or <>.
	k := j
	for k < len(s) && k-j < 33 && (isASCIILetter(s[k]) || k > j && (isASCIIDigit(s[k]) || s[k] == '+' || s[k] == '.' || s[k] == '-')) {
		k++
	}
	if n := k - j; n >= 2 && n <= 32 && k < len(s) && s[k] == ':' {
		e := k + 1
		for e < len(s) && s[e] > ' ' && s[e] != '<' && s[e] != '>' {
			e++
		}
		if e < len(s) && s[e] == '>' {
			return e + 1 - i, s[j:e], false
		}
	}
	// An email address.
	k = j
	for k < len(s) && (isASCIILetter(s[k]) || isASCIIDigit(s[k]) || strings.IndexByte(".!#$%&'*+/=?^_`{|}~-", s[k]) >= 0) {
		k++
	}
	if k == j || k >= len(s) || s[k] != '@' {
		return 0, "", false
	}
	e := k + 1
	for {
		// A label: alphanumerics and inner hyphens, at most 63.
		l := e
		for e < len(s) && (isASCIILetter(s[e]) || isASCIIDigit(s[e]) || s[e] == '-') && e-l < 64 {
			e++
		}
		if e == l || e-l > 63 || s[l] == '-' || s[e-1] == '-' {
			return 0, "", false
		}
		if e < len(s) && s[e] == '.' {
			e++
			continue
		}
		break
	}
	if e < len(s) && s[e] == '>' {
		return e + 1 - i, "mailto:" + s[j:e], true
	}
	return 0, "", false
}

// scanDelims reads a run of delimiter c at pos: its length, and whether
// it can open and close emphasis.
func (ip *inlineParser) scanDelims(c byte) (n int, canOpen, canClose bool) {
	s := ip.s
	i := ip.pos
	for i < len(s) && s[i] == c {
		i++
	}
	n = i - ip.pos
	before, after := '\n', '\n'
	if ip.pos > 0 {
		before, _ = utf8.DecodeLastRuneInString(s[:ip.pos])
	}
	if i < len(s) {
		after, _ = utf8.DecodeRuneInString(s[i:])
	}
	afterSpace, afterPunct := isUnicodeSpace(after), isPunct(after)
	beforeSpace, beforePunct := isUnicodeSpace(before), isPunct(before)
	left := !afterSpace && (!afterPunct || beforeSpace || beforePunct)
	right := !beforeSpace && (!beforePunct || afterSpace || afterPunct)
	switch c {
	case '_':
		canOpen = left && (!right || beforePunct)
		canClose = right && (!left || afterPunct)
	case '~':
		canOpen, canClose = left, right
		// A tilde between letters or digits is not strikethrough, as
		// 20~25, which language models write for ranges.
		if n == 1 && isAlnum(before) && isAlnum(after) {
			canOpen, canClose = false, false
		}
	default:
		canOpen, canClose = left, right
	}
	return
}

func isAlnum(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }

func isUnicodeSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\f' || unicode.Is(unicode.Zs, r)
}

// isPunct reports whether r is punctuation or a symbol, as CommonMark
// counts them for flanking.
func isPunct(r rune) bool {
	if r < utf8.RuneSelf {
		return isEscapable(byte(r))
	}
	return unicode.IsPunct(r) || unicode.IsSymbol(r)
}

func (ip *inlineParser) handleDelim(c byte) {
	n, canOpen, canClose := ip.scanDelims(c)
	start := ip.pos
	ip.pos += n
	node := ip.addLiteral(ip.s[start:ip.pos], start, ip.pos)
	if c == '~' && n > 2 {
		return
	}
	if canOpen || canClose {
		ip.dl = append(ip.dl, delim{c: c, n: n, orig: n, node: node, prev: ip.last, next: -1, canOpen: canOpen, canClose: canClose, at: start})
		k := int32(len(ip.dl) - 1)
		if ip.last >= 0 {
			ip.dl[ip.last].next = k
		}
		ip.last = k
	}
}

func (ip *inlineParser) removeDelim(k int32) {
	d := &ip.dl[k]
	if d.prev >= 0 {
		ip.dl[d.prev].next = d.next
	}
	if d.next >= 0 {
		ip.dl[d.next].prev = d.prev
	} else {
		ip.last = d.prev
	}
	d.prev, d.next = -1, -1
	d.n = -1 // removed
}

// processEmphasis matches the delimiters at or after offset bottom, as
// CommonMark's algorithm does, openersBottom keeping it linear; final
// notes the openers left unmatched, for healing.
func (ip *inlineParser) processEmphasis(bottom int, final bool) {
	var openersBottom [3][6]int
	for i := range openersBottom {
		for j := range openersBottom[i] {
			openersBottom[i][j] = bottom
		}
	}
	closer := int32(-1)
	for c := ip.last; c >= 0 && ip.dl[c].at >= bottom; c = ip.dl[c].prev {
		closer = c
		ip.d.steps++
	}
	for closer >= 0 {
		ip.d.steps++
		cd := &ip.dl[closer]
		if !cd.canClose {
			closer = cd.next
			continue
		}
		ci := delimClass(cd.c, cd.canOpen)
		opener := cd.prev
		found := false
		for opener >= 0 && ip.dl[opener].at >= openersBottom[cd.orig%3][ci] {
			ip.d.steps++
			od := &ip.dl[opener]
			odd := (cd.canOpen || od.canClose) && cd.orig%3 != 0 && (od.orig+cd.orig)%3 == 0
			if od.c == cd.c && od.canOpen && !odd {
				found = true
				break
			}
			opener = od.prev
		}
		if !found {
			old := closer
			closer = cd.next
			openersBottom[ip.dl[old].orig%3][ci] = ip.dl[old].at
			if !ip.dl[old].canOpen {
				ip.removeDelim(old)
			}
			continue
		}
		if cd.c == '~' {
			closer = ip.insertStrike(opener, closer)
			continue
		}
		closer = ip.insertEmph(opener, closer)
	}
	// What remains matched nothing.
	for ip.last >= 0 && ip.dl[ip.last].at >= bottom {
		ip.d.steps++
		if d := ip.dl[ip.last]; final && d.canOpen && d.n > 0 {
			ip.leftover = append(ip.leftover, d)
		}
		ip.removeDelim(ip.last)
	}
}

func delimClass(c byte, canOpen bool) int {
	k := 0
	switch c {
	case '_':
		k = 2
	case '~':
		k = 4
	}
	if canOpen {
		k++
	}
	return k
}

// insertEmph wraps what is between opener and closer in emphasis, one or
// two delimiters of each, and returns the next closer to look at.
func (ip *inlineParser) insertEmph(opener, closer int32) int32 {
	od, cd := &ip.dl[opener], &ip.dl[closer]
	use := 1
	if cd.n >= 2 && od.n >= 2 {
		use = 2
	}
	on, cn := od.node, cd.node
	od.n -= use
	cd.n -= use
	o, c := &ip.nodes[on], &ip.nodes[cn]
	o.text = o.text[:len(o.text)-use]
	o.end -= use
	c.text = c.text[:len(c.text)-use]
	c.start += use
	kind := Emphasis
	if use == 2 {
		kind = Strong
	}
	emph := ip.newNode(kind, "", ip.nodes[on].end, ip.nodes[cn].start)
	for t := ip.nodes[on].next; t >= 0 && t != cn; {
		ip.d.steps++
		next := ip.nodes[t].next
		ip.unlink(t)
		ip.appendChild(emph, t)
		t = next
	}
	ip.insertAfter(on, emph)
	// The delimiters between are gone.
	for k := ip.dl[closer].prev; k >= 0 && k != opener; {
		ip.d.steps++
		prev := ip.dl[k].prev
		ip.removeDelim(k)
		k = prev
	}
	if ip.dl[opener].n == 0 {
		ip.unlink(on)
		ip.removeDelim(opener)
	}
	if ip.dl[closer].n == 0 {
		ip.unlink(cn)
		next := ip.dl[closer].next
		ip.removeDelim(closer)
		return next
	}
	return closer
}

// insertStrike makes strikethrough of what is between two runs of
// tildes of the same length.
func (ip *inlineParser) insertStrike(opener, closer int32) int32 {
	next := ip.dl[closer].next
	od, cd := ip.dl[opener], ip.dl[closer]
	if od.n == cd.n {
		on, cn := od.node, cd.node
		st := ip.newNode(Strikethrough, "", ip.nodes[on].start, ip.nodes[cn].end)
		for t := ip.nodes[on].next; t >= 0 && t != cn; {
			ip.d.steps++
			n := ip.nodes[t].next
			ip.unlink(t)
			ip.appendChild(st, t)
			t = n
		}
		ip.insertAfter(on, st)
		ip.unlink(on)
		ip.unlink(cn)
	}
	for k := closer; k >= 0 && k != opener; {
		ip.d.steps++
		prev := ip.dl[k].prev
		ip.removeDelim(k)
		k = prev
	}
	ip.removeDelim(opener)
	return next
}

func (ip *inlineParser) addBracket(node int32, index int, image bool) {
	if ip.lastB >= 0 {
		ip.br[ip.lastB].bracketAfter = true
	}
	ip.br = append(ip.br, bracket{node: node, prev: ip.lastB, prevDelim: ip.last, index: index, image: image})
	ip.lastB = int32(len(ip.br) - 1)
}

func (ip *inlineParser) popBracket() {
	ip.lastB = ip.br[ip.lastB].prev
}

func (ip *inlineParser) parseCloseBracket() {
	s := ip.s
	ip.pos++
	start := ip.pos // after ]
	if ip.lastB < 0 {
		ip.addText(start-1, start)
		return
	}
	op := ip.br[ip.lastB]
	if !op.image && ip.lastB < ip.linked {
		ip.addText(start-1, start)
		ip.popBracket()
		return
	}
	var dest, title string
	matched := false
	// An inline link.
	if ip.pos < len(s) && s[ip.pos] == '(' {
		save := ip.pos
		ip.pos++
		ip.spnl()
		if d, ok := ip.parseLinkDestination(); ok {
			dest = d
			before := ip.pos
			ip.spnl()
			ok := true
			if ip.pos > before {
				if t, tok := ip.parseLinkTitle(); tok {
					title = t
				} else {
					ip.pos = before
					ip.spnl()
				}
			}
			ip.spnl()
			if ok && ip.pos < len(s) && s[ip.pos] == ')' {
				ip.pos++
				matched = true
			}
		}
		if !matched {
			ip.pos = save
		}
	}
	if !matched {
		// A reference: [text][label], [text][] or [text].
		before := ip.pos
		n := ip.parseLinkLabel()
		var label string
		has := false
		if n > 2 {
			label, has = s[before:before+n], true
		} else if !op.bracketAfter {
			label, has = s[op.index-1:start], true
		}
		if n == 0 {
			ip.pos = start
		}
		if has {
			ip.d.steps += len(label)
			key := normalizeLabel(label[1 : len(label)-1])
			if key != "" {
				ip.noteLabel(key)
				if r, ok := ip.d.refs[key]; ok {
					dest, title, matched = r.dest, r.title, true
				}
			}
		}
	}
	if matched {
		kind := Link
		if op.image {
			kind = Image
		}
		openStart := ip.nodes[op.node].start
		l := ip.newNode(kind, "", openStart, ip.pos)
		ip.nodes[l].dest, ip.nodes[l].title = dest, title
		for t := ip.nodes[op.node].next; t >= 0; {
			ip.d.steps++
			next := ip.nodes[t].next
			ip.unlink(t)
			ip.appendChild(l, t)
			t = next
		}
		ip.appendChild(0, l)
		ip.processEmphasis(op.index, false)
		ip.popBracket()
		ip.unlink(op.node)
		if !op.image {
			// The brackets open before it are below len(ip.br), as
			// brackets are added at the end.
			ip.linked = int32(len(ip.br))
		}
		return
	}
	// A footnote reference: [^label], when the footnote is defined.
	if ip.ext&ExtFootnotes != 0 && !op.image && !op.bracketAfter && op.index < start-1 && s[op.index] == '^' && start-1 > op.index+1 && start-1-op.index <= 1000 {
		ip.d.steps += start - 1 - op.index
		key := normalizeLabel(s[op.index+1 : start-1])
		ip.noteLabel("^" + key)
		if _, ok := ip.d.notes[key]; ok {
			ip.pos = start
			f := ip.newNode(FootnoteRef, s[op.index+1:start-1], ip.nodes[op.node].start, start)
			ip.processEmphasis(op.index, false)
			for t := ip.nodes[op.node].next; t >= 0; {
				ip.d.steps++
				next := ip.nodes[t].next
				ip.unlink(t)
				t = next
			}
			ip.insertBefore(op.node, f)
			ip.unlink(op.node)
			ip.popBracket()
			return
		}
	}
	ip.popBracket()
	ip.pos = start
	ip.addText(start-1, start)
}

func (ip *inlineParser) noteLabel(key string) {
	if !containsString(ip.labels, key) {
		ip.labels = append(ip.labels, key)
	}
}

// spnl skips spaces and tabs, and one line ending with those around it.
func (ip *inlineParser) spnl() {
	s := ip.s
	for ip.pos < len(s) && (s[ip.pos] == ' ' || s[ip.pos] == '\t') {
		ip.pos++
	}
	if ip.pos < len(s) && s[ip.pos] == '\n' {
		ip.pos++
		for ip.pos < len(s) && (s[ip.pos] == ' ' || s[ip.pos] == '\t') {
			ip.pos++
		}
	}
}

// parseLinkLabel returns the length of a link label at pos, [ and ]
// included, or 0.
func (ip *inlineParser) parseLinkLabel() int {
	s := ip.s
	i := ip.pos
	if i >= len(s) || s[i] != '[' {
		return 0
	}
	j := i + 1
	for j < len(s) && j-i <= 1001 {
		switch s[j] {
		case '\\':
			j++
		case '[':
			ip.d.steps += j - i
			return 0
		case ']':
			ip.d.steps += j - i
			if j-i > 1000 {
				return 0
			}
			ip.pos = j + 1
			return j + 1 - i
		}
		j++
	}
	ip.d.steps += j - i
	return 0
}

// parseLinkDestination reads a link destination at pos, <…> or bare
// with balanced parentheses.
func (ip *inlineParser) parseLinkDestination() (string, bool) {
	s := ip.s
	i := ip.pos
	if i < len(s) && s[i] == '<' {
		for j := i + 1; j < len(s); j++ {
			switch s[j] {
			case '\\':
				if j+1 < len(s) && isEscapable(s[j+1]) {
					j++
				}
			case '\n', '<':
				ip.d.steps += j - i
				return "", false
			case '>':
				ip.d.steps += j - i
				ip.pos = j + 1
				return unescapeString(s[i+1 : j]), true
			}
		}
		ip.d.steps += len(s) - i
		return "", false
	}
	parens := 0
	j := i
loop:
	for j < len(s) {
		c := s[j]
		switch {
		case c == '\\' && j+1 < len(s) && isEscapable(s[j+1]):
			j += 2
		case c == '(':
			parens++
			if parens > 32 {
				ip.d.steps += j - i
				return "", false
			}
			j++
		case c == ')':
			if parens < 1 {
				break loop
			}
			parens--
			j++
		case c <= ' ' || c == 0x7f:
			break loop
		default:
			j++
		}
	}
	ip.d.steps += j - i
	if j == i && (j >= len(s) || s[j] != ')') {
		return "", false
	}
	if parens != 0 {
		return "", false
	}
	ip.pos = j
	return unescapeString(s[i:j]), true
}

// parseLinkTitle reads a link title at pos: "…", '…' or (…).
func (ip *inlineParser) parseLinkTitle() (string, bool) {
	s := ip.s
	i := ip.pos
	if i >= len(s) {
		return "", false
	}
	end := s[i]
	switch end {
	case '"', '\'':
	case '(':
		end = ')'
	default:
		return "", false
	}
	for j := i + 1; j < len(s); j++ {
		switch c := s[j]; {
		case c == '\\' && j+1 < len(s):
			j++
		case c == end:
			ip.d.steps += j - i
			ip.pos = j + 1
			return unescapeString(s[i+1 : j]), true
		case c == '(' && end == ')':
			ip.d.steps += j - i
			return "", false
		}
	}
	ip.d.steps += len(s) - i
	return "", false
}

// NormalizeLabel normalizes a label as CommonMark matches link
// references and footnotes: whitespace collapsed and case folded.
func NormalizeLabel(s string) string { return normalizeLabel(s) }

// normalizeLabel normalizes a link label for matching: whitespace
// collapsed and case folded.
func normalizeLabel(s string) string {
	var b strings.Builder
	space := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			space = b.Len() > 0
			continue
		}
		if space {
			b.WriteByte(' ')
			space = false
		}
		b.WriteByte(c)
	}
	s = b.String()
	s = strings.ToLower(s)
	if strings.Contains(s, "ß") {
		s = strings.ReplaceAll(s, "ß", "ss")
	}
	return strings.ToUpper(s)
}

// parseReference reads a link reference definition at the start of s,
// adding it to the document unless its label is defined, and returns its
// length, or 0. at is the offset of s in the source.
func (ip *inlineParser) parseReference(d *Document, s string, at int) int {
	saveS, savePos := ip.s, ip.pos
	defer func() { ip.s, ip.pos = saveS, savePos }()
	ip.s, ip.pos = s, 0
	n := ip.parseLinkLabel()
	if n == 0 {
		return 0
	}
	label := s[1 : n-1]
	if ip.pos >= len(s) || s[ip.pos] != ':' {
		return 0
	}
	ip.pos++
	ip.spnl()
	dest, ok := ip.parseLinkDestination()
	if !ok {
		return 0
	}
	beforeTitle := ip.pos
	ip.spnl()
	title, hasTitle := "", false
	if ip.pos != beforeTitle {
		title, hasTitle = ip.parseLinkTitle()
	}
	if !hasTitle {
		ip.pos = beforeTitle
	}
	atEnd := func() bool {
		j := ip.pos
		for j < len(s) && (s[j] == ' ' || s[j] == '\t') {
			j++
		}
		if j == len(s) || s[j] == '\n' {
			if j < len(s) {
				j++
			}
			ip.pos = j
			return true
		}
		return false
	}
	if !atEnd() {
		if !hasTitle {
			return 0
		}
		title, hasTitle = "", false
		ip.pos = beforeTitle
		if !atEnd() {
			return 0
		}
	}
	key := normalizeLabel(label)
	if key == "" {
		return 0
	}
	d.define(definition{key, linkRef{dest: dest, title: title, at: at}})
	return ip.pos
}

// flatten turns the tree into the block's inlines, in source offsets:
// content offsets from base, those past healFrom (text healing added)
// at healFrom.
func (ip *inlineParser) flatten(d *Document, block, base, healFrom int) []Inline {
	if ip.nodes[0].first < 0 {
		return nil
	}
	out := make([]Inline, 0, len(ip.nodes)-1)
	m := d.sourceMap(block)
	srcOf := func(off int) int {
		return m.source(base + min(off, healFrom))
	}
	var walk func(n int32)
	walk = func(n int32) {
		for c := ip.nodes[n].first; c >= 0; c = ip.nodes[c].next {
			nd := &ip.nodes[c]
			if nd.kind == Text && nd.text == "" {
				continue
			}
			d.steps++
			i := len(out)
			out = append(out, Inline{Kind: nd.kind, Text: nd.text, Dest: nd.dest, Title: nd.title, Start: srcOf(nd.start), End: srcOf(nd.end), Auto: nd.auto, Display: nd.display})
			if nd.end > healFrom && (nd.kind == Link || nd.kind == Image) {
				out[i].Pending = true
			}
			walk(c)
			out[i].Next = len(out)
		}
	}
	walk(0)
	return out
}

// joinTexts joins the runs of text among the children of node n, and
// below it, into one node each: a slice of the subject when their texts
// are its bytes one after another, else the texts joined once, so that a
// run of many, as delimiters left unmatched, takes linear time.
func (ip *inlineParser) joinTexts(n int32) {
	for c := ip.nodes[n].first; c >= 0; c = ip.nodes[c].next {
		ip.d.steps++
		if ip.nodes[c].kind != Text {
			ip.joinTexts(c)
			continue
		}
		last := c
		for nx := ip.nodes[c].next; nx >= 0 && ip.nodes[nx].kind == Text; nx = ip.nodes[nx].next {
			ip.d.steps++
			last = nx
		}
		if last == c {
			continue
		}
		first := &ip.nodes[c]
		exact := true
		var b strings.Builder
		for k := c; ; k = ip.nodes[k].next {
			nd := &ip.nodes[k]
			if nd.text != ip.s[nd.start:min(nd.end, len(ip.s))] || k != c && nd.start != ip.nodes[nd.prev].end {
				exact = false
			}
			if k == last {
				break
			}
		}
		end := ip.nodes[last].end
		if exact {
			first.text = ip.s[first.start:end]
		} else {
			for k := c; ; k = ip.nodes[k].next {
				b.WriteString(ip.nodes[k].text)
				if k == last {
					break
				}
			}
			first.text = b.String()
		}
		first.end, first.plain = end, false
		for k := ip.nodes[c].next; k >= 0; {
			nx := ip.nodes[k].next
			ip.unlink(k)
			if k == last {
				break
			}
			k = nx
		}
	}
}
