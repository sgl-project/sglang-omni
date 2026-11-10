package markdown

import (
	"strings"
	"unicode/utf8"

	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown/parser"
)

// A paragraph is the spans of its text, cut into pieces at its links,
// which are elements of their own inside it (ui.Link), with where each
// span's text is in the source, for copying.

// piece is a run of spans: a link's, when link >= 0 (an index into the
// block's inlines), else plain text.
type piece struct {
	link  int
	spans []ui.Span
}

// runMark notes that the text of a paragraph from rune comes from the
// source's bytes [src, end): exact when it is those bytes, else written
// otherwise there, as an entity or an escape.
type runMark struct {
	rune, src, end int
	exact          bool
}

// para is what a paragraph, heading or cell shows.
type para struct {
	pieces []piece
	marks  []runMark
	runes  int
	fading bool // its spans fade in: build them anew next frame
	// width is the width of its text on one line, once measured.
	width    float32
	measured bool
}

// spanStyle is the style inlines set as the builder descends.
type spanStyle struct {
	weight              int
	italic, strike      bool
	underline           bool
	code                bool
	sub                 bool
	mark                bool
	color               Color
	size                float32
	font                string
	inLink, pendingLink bool
}

type spanBuilder struct {
	d     *parser.Document
	th    *Theme
	block int
	in    []parser.Inline
	p     *para
	cur   *piece
	fade  func(src int) float32 // the opacity of text at a source offset, while revealing
	style spanStyle
	// html is the stack of inline HTML tags open, as <b>.
	html []string
	// notes numbers footnotes by their definitions' order.
	notes func(label string) int
	// linkIndex is the inline of the link being built.
	linkIndex int
}

// build makes the paragraph of block i, of base style s.
func (sb *spanBuilder) build(p *para, base spanStyle) {
	*p = para{pieces: p.pieces[:0], marks: p.marks[:0]}
	sb.p = p
	sb.in = sb.d.Inlines(sb.block)
	sb.html = sb.html[:0]
	sb.style = base
	sb.cur = nil
	sb.inlines(0, len(sb.in), base)
	// A paragraph's pieces keep their own slices: they outlive the build.
	for i := range p.pieces {
		p.pieces[i].spans = append([]ui.Span(nil), p.pieces[i].spans...)
	}
}

func (sb *spanBuilder) piece(link int) *piece {
	if sb.cur == nil || sb.cur.link != link {
		sb.p.pieces = append(sb.p.pieces, piece{link: link})
		sb.cur = &sb.p.pieces[len(sb.p.pieces)-1]
	}
	return sb.cur
}

// add adds text, which comes from the source's [src, end) (exact when it
// is those bytes), in style s.
func (sb *spanBuilder) add(text string, src, end int, exact bool, s spanStyle, link int) {
	if text == "" {
		return
	}
	if sb.fade != nil && exact {
		// Text revealed at different times fades in on its own: cut it
		// where its opacity changes.
		for len(text) > 0 {
			a := sb.fade(src)
			n := len(text)
			for k := 1; k < len(text); k++ {
				if sb.fade(src+k) != a {
					n = k
					break
				}
			}
			for n < len(text) && !utf8.RuneStart(text[n]) {
				n++
			}
			sb.addSpan(text[:n], src, src+n, exact, s, link, a)
			text, src = text[n:], src+n
		}
		return
	}
	a := float32(1)
	if sb.fade != nil {
		a = sb.fade(src)
	}
	sb.addSpan(text, src, end, exact, s, link, a)
}

// addAligned adds text that comes from the source's [src, end) written
// with escapes or entities: as many runs as are its bytes, and the rest,
// each escape or entity, of its own, so that copying maps its runes to
// the source's.
func (sb *spanBuilder) addAligned(text string, src, end int, s spanStyle, link int) {
	source := sb.d.Source()[src:end]
	i, j := 0, 0 // in source and in text
	run := 0     // where the run of the same bytes started, in text
	flush := func() {
		if j > run {
			sb.add(text[run:j], src+i-(j-run), src+i, true, s, link)
		}
	}
	for i < len(source) && j < len(text) {
		switch {
		case source[i] == '\\' && i+1 < len(source) && isEscapable(source[i+1]) && source[i+1] == text[j]:
			flush()
			sb.add(text[j:j+1], src+i, src+i+2, false, s, link)
			i, j = i+2, j+1
		case source[i] == '&' && entityAt(source, i, text[j:]) > 0:
			n, lit := parser.Entity(source[i:])
			flush()
			sb.add(lit, src+i, src+i+n, false, s, link)
			i, j = i+n, j+len(lit)
		case source[i] == text[j]:
			i++
			j++
			continue
		default:
			goto rest
		}
		run = j
	}
	flush()
	run = j
rest:
	if j < len(text) {
		if j > run {
			flush()
		}
		sb.add(text[j:], src+i, end, false, s, link)
	}
}

// entityAt returns the length of the entity at i of source that text
// starts with, decoded, or 0.
func entityAt(source string, i int, text string) int {
	n, lit := parser.Entity(source[i:])
	if n == 0 || !strings.HasPrefix(text, lit) {
		return 0
	}
	return n
}

func (sb *spanBuilder) addSpan(text string, src, end int, exact bool, s spanStyle, link int, alpha float32) {
	th := sb.th
	sp := ui.Span{Text: text, Font: th.Font, Size: th.FontSize, Weight: s.weight, Italic: s.italic, Strikethrough: s.strike, Underline: s.underline}
	if s.size != 0 {
		sp.Size = s.size
	}
	if s.font != "" {
		sp.Font = s.font
	}
	if s.code {
		sp.Font = th.CodeFont
		sp.Size = th.em(th.CodeSize)
		if s.size != 0 {
			sp.Size = s.size * th.CodeSize
		}
		sp.Background = th.CodeBackground
		sp.Radius = max(3, sp.Size/4)
		if !s.inLink {
			sp.Color = th.Code
		}
	}
	if s.sub {
		sp.Size *= 0.75
	}
	if s.mark {
		sp.Background = th.Mark
	}
	if s.color.A > 0 {
		sp.Color = s.color
	}
	if alpha < 1 {
		// Not quite transparent: a color of alpha 0 is none, the text's.
		alpha = max(alpha, 0.02)
		if sp.Color.A == 0 {
			sp.Color = th.Text
			if s.inLink {
				sp.Color = th.Link
			}
		}
		sp.Color = sp.Color.Alpha(alpha)
		sp.Background = sp.Background.Alpha(alpha)
	}
	if sb.fade != nil {
		sb.p.fading = sb.p.fading || alpha < 1
	}
	pc := sb.piece(link)
	pc.spans = append(pc.spans, sp)
	sb.p.marks = append(sb.p.marks, runMark{rune: sb.p.runes, src: src, end: end, exact: exact})
	sb.p.runes += utf8.RuneCountInString(text)
}

func (sb *spanBuilder) inlines(from, to int, s spanStyle) {
	in := sb.in
	link := -1
	if s.inLink {
		link = sb.linkIndex
	}
	for k := from; k < to; k = in[k].Next {
		n := &in[k]
		switch n.Kind {
		case parser.Text:
			if n.End-n.Start == len(n.Text) && sb.d.Source()[n.Start:n.End] == n.Text {
				sb.add(n.Text, n.Start, n.End, true, sb.htmlStyle(s), link)
			} else {
				sb.addAligned(n.Text, n.Start, n.End, sb.htmlStyle(s), link)
			}
		case parser.SoftBreak:
			sb.add(" ", n.Start, n.End, false, s, link)
		case parser.LineBreak:
			sb.add("\n", n.Start, n.End, false, s, link)
		case parser.Code:
			cs := sb.htmlStyle(s)
			cs.code = true
			// The code's text is its source's bytes but for the spaces
			// stripped at its ends and line endings as spaces.
			a, z := codeStart(sb.d.Source(), n), n.End-(codeStart(sb.d.Source(), n)-n.Start)
			if z-a == len(n.Text)+2 && sb.d.Source()[a] == ' ' {
				a++
			}
			sb.add(n.Text, a, min(a+len(n.Text), z), z-a == len(n.Text), cs, link)
		case parser.Emphasis:
			cs := s
			cs.italic = true
			sb.inlines(k+1, n.Next, cs)
		case parser.Strong:
			cs := s
			cs.weight = sb.th.StrongWeight
			if cs.color.A == 0 && !cs.inLink && sb.th.Strong != sb.th.Text {
				cs.color = sb.th.Strong
			}
			sb.inlines(k+1, n.Next, cs)
		case parser.Strikethrough:
			cs := s
			cs.strike = true
			sb.inlines(k+1, n.Next, cs)
		case parser.Link:
			if s.inLink {
				sb.inlines(k+1, n.Next, s)
				continue
			}
			cs := s
			cs.inLink, cs.pendingLink = true, n.Pending
			cs.color = Color{} // the link's
			save := sb.linkIndex
			sb.linkIndex = k
			sb.inlines(k+1, n.Next, cs)
			sb.linkIndex = save
		case parser.Image:
			// Images within text show their description; a paragraph of
			// images alone shows the images (see imagesOnly).
			cs := s
			cs.color = sb.th.Muted
			alt := parser.PlainText(in[k+1 : n.Next])
			if alt == "" {
				alt = "image"
			}
			sb.add("["+alt+"]", n.Start, n.End, false, cs, link)
		case parser.HTML:
			if !sb.inlineHTML(n.Text) {
				sb.add(n.Text, n.Start, n.End, true, s, link)
			}
		case parser.Math:
			cs := s
			cs.font = sb.th.CodeFont
			cs.color = sb.th.Muted
			if !cs.italic {
				cs.size = sb.th.em(sb.th.CodeSize)
			}
			sb.add(n.Text, mathStart(sb.d.Source(), n), n.End, false, cs, link)
		case parser.FootnoteRef:
			cs := s
			cs.sub = true
			cs.color = sb.th.Link
			num := 0
			if sb.notes != nil {
				num = sb.notes(n.Text)
			}
			label := n.Text
			if num > 0 {
				label = itoa(num)
			}
			sb.add("["+label+"]", n.Start, n.End, false, cs, -2-k)
		}
	}
}

// codeStart returns where a code span's text starts in the source.
func codeStart(src string, n *parser.Inline) int {
	i := n.Start
	for i < n.End && src[i] == '`' {
		i++
	}
	return i
}

func mathStart(src string, n *parser.Inline) int {
	i := n.Start
	for i < n.End && (src[i] == '$' || src[i] == '\\' || src[i] == '(' || src[i] == '[') && i-n.Start < 2 {
		i++
	}
	return i
}

func itoa(n int) string {
	if n < 10 {
		return string(rune('0' + n))
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// inlineHTML applies the inline HTML tags of formatting language models
// write, <b>, <i>, <code>, <br> and the like, and reports whether tag was
// one: others show as text.
func (sb *spanBuilder) inlineHTML(tag string) bool {
	t := strings.ToLower(strings.Trim(tag, "<>/ \t\n"))
	if i := strings.IndexAny(t, " \t\n"); i >= 0 {
		t = t[:i]
	}
	closing := strings.HasPrefix(tag, "</")
	switch t {
	case "br":
		sb.add("\n", 0, 0, false, sb.style, -1)
		return true
	case "b", "strong", "i", "em", "u", "ins", "s", "del", "strike", "code", "kbd", "samp", "mark", "sub", "sup", "small":
	default:
		return false
	}
	if closing {
		for i := len(sb.html) - 1; i >= 0; i-- {
			if sb.html[i] == t {
				sb.html = append(sb.html[:i], sb.html[i+1:]...)
				break
			}
		}
	} else if !strings.HasSuffix(tag, "/>") {
		sb.html = append(sb.html, t)
	}
	return true
}

// htmlStyle returns s with the inline HTML tags open applied.
func (sb *spanBuilder) htmlStyle(s spanStyle) spanStyle {
	for _, t := range sb.html {
		switch t {
		case "b", "strong":
			s.weight = sb.th.StrongWeight
		case "i", "em":
			s.italic = true
		case "u", "ins":
			s.underline = true
		case "s", "del", "strike":
			s.strike = true
		case "code", "kbd", "samp":
			s.code = true
		case "mark":
			s.mark = true
		case "sub", "sup", "small":
			s.sub = true
		}
	}
	return s
}

// imagesOnly reports whether the inlines are images, and the spaces
// between them, alone: a paragraph showing them as images.
func imagesOnly(in []parser.Inline) bool {
	n := 0
	for k := 0; k < len(in); k = in[k].Next {
		switch in[k].Kind {
		case parser.Image:
			n++
		case parser.SoftBreak, parser.LineBreak:
		case parser.Text:
			if strings.TrimSpace(in[k].Text) != "" {
				return false
			}
		default:
			return false
		}
	}
	return n > 0
}

// sourceAt returns the source offset of rune r of the paragraph's text:
// for the end of a selection (end), that of the text before it, so that
// what follows, as an emphasis's opening *, stays out.
func (p *para) sourceAt(src string, r int, end bool) int {
	if len(p.marks) == 0 {
		return 0
	}
	lo, hi := 0, len(p.marks)
	for lo < hi {
		m := (lo + hi) / 2
		if p.marks[m].rune <= r {
			lo = m + 1
		} else {
			hi = m
		}
	}
	k := max(lo-1, 0)
	m := p.marks[k]
	if r <= m.rune {
		if end && k > 0 {
			return p.marks[k-1].end
		}
		return m.src
	}
	if !m.exact {
		return m.end
	}
	off := m.src
	for n := m.rune; n < r && off < m.end; n++ {
		_, w := utf8.DecodeRuneInString(src[off:])
		off += w
	}
	return off
}
