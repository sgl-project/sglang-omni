package richtext

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"html"
	"image"
	"strconv"
	"strings"
	"unicode"

	"github.com/egoist/mygo/ui"
	xhtml "golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

// HTML returns t as HTML: paragraphs, headings, lists (task lists with
// check boxes), quotations and blocks of code, and runs in the tags and
// styles of their attributes, as browsers, mail and word processors paste
// it.
func (t Text) HTML() string {
	var b strings.Builder
	b.WriteString(`<meta charset="utf-8">`)
	var paras []Paragraph
	for _, p := range t.Paragraphs() {
		paras = append(paras, p)
	}
	writeBlocks(&b, paras)
	return b.String()
}

// writeBlocks writes paragraphs as HTML blocks; those of a quotation in a
// blockquote, whatever they are.
func writeBlocks(b *strings.Builder, paras []Paragraph) {
	// open is the stack of lists open: their tags.
	var open []string
	closeLists := func(depth int) {
		for len(open) > depth {
			b.WriteString("</li></" + open[len(open)-1] + ">")
			open = open[:len(open)-1]
		}
	}
	for i := 0; i < len(paras); i++ {
		p := paras[i]
		st := p.Style
		if st.List == NoList || st.Quote {
			closeLists(0)
		}
		align := ""
		switch st.Align {
		case ui.Center:
			align = ` style="text-align:center"`
		case ui.End:
			align = ` style="text-align:right"`
		}
		switch {
		case st.Quote:
			j := i
			var quoted []Paragraph
			for ; j < len(paras) && paras[j].Style.Quote; j++ {
				q := paras[j]
				q.Style.Quote = false
				quoted = append(quoted, q)
			}
			b.WriteString("<blockquote>")
			writeBlocks(b, quoted)
			b.WriteString("</blockquote>")
			i = j - 1
		case st.List != NoList:
			tag := "ul"
			if st.List == Numbers {
				tag = "ol"
			}
			depth := st.Indent + 1
			if len(open) >= depth && open[depth-1] != tag {
				closeLists(depth - 1)
			}
			switch {
			case len(open) >= depth:
				closeLists(depth)
				b.WriteString("</li>")
			default:
				for len(open) < depth {
					b.WriteString("<" + tag + ">")
					open = append(open, tag)
					if len(open) < depth {
						b.WriteString("<li>")
					}
				}
			}
			b.WriteString("<li" + align + ">")
			if st.List == Tasks {
				if st.Checked {
					b.WriteString(`<input type="checkbox" checked disabled> `)
				} else {
					b.WriteString(`<input type="checkbox" disabled> `)
				}
			}
			writeRuns(b, p.Runs)
		case st.CodeBlock:
			b.WriteString("<pre><code>")
			for ; i < len(paras) && paras[i].Style.CodeBlock; i++ {
				for _, r := range paras[i].Runs {
					b.WriteString(html.EscapeString(r.Text))
				}
				if i+1 < len(paras) && paras[i+1].Style.CodeBlock {
					b.WriteByte('\n')
				}
			}
			i--
			b.WriteString("</code></pre>")
		default:
			tag := blockTag(st)
			b.WriteString("<" + tag + align + ">")
			if len(p.Runs) == 0 {
				b.WriteString("<br>")
			}
			writeRuns(b, p.Runs)
			b.WriteString("</" + tag + ">")
		}
	}
	closeLists(0)
}

// imageSide returns the width or height an img's attribute gives, or 0.
func imageSide(s string) float32 {
	v, err := strconv.ParseFloat(strings.TrimSuffix(strings.TrimSpace(s), "px"), 32)
	if err != nil {
		return 0
	}
	return side(float32(v))
}

// side returns v as an attachment's width or height: 0, for its picture's,
// unless it is a size, not negative, infinite or not a number.
func side(v float32) float32 {
	if !(v > 0) || v > 1e5 {
		return 0
	}
	return v
}

// maxImagePixels is how many pixels a pasted picture may have, so a small
// file that claims a huge size takes no gigabytes to decode.
const maxImagePixels = 1 << 25

// smallImage reports whether the picture of data is small enough to decode.
func smallImage(data []byte) bool {
	c, _, err := image.DecodeConfig(bytes.NewReader(data))
	return err == nil && c.Width > 0 && c.Height > 0 && int64(c.Width)*int64(c.Height) <= maxImagePixels
}

// safeLink returns a link's URL, or "" for one that runs a script where a
// browser opens it: javascript:, vbscript: and data:, written as browsers
// read them, with spaces before and tabs or newlines within.
func safeLink(s string) string {
	s = strings.TrimFunc(s, func(r rune) bool { return r <= ' ' })
	scheme := strings.Map(func(r rune) rune {
		if r == '\t' || r == '\n' || r == '\r' {
			return -1
		}
		return unicode.ToLower(r)
	}, s)
	for _, bad := range []string{"javascript:", "vbscript:", "data:"} {
		if strings.HasPrefix(scheme, bad) {
			return ""
		}
	}
	return s
}

// blockTag returns the tag of a paragraph of style st: p, or h1 to h6.
func blockTag(st ParagraphStyle) string {
	if st.Heading > 0 {
		return "h" + strconv.Itoa(min(st.Heading, 6))
	}
	return "p"
}

// writeRuns writes runs as HTML: a link around the tags of the formats,
// and a span with the styles of fonts and colors.
func writeRuns(b *strings.Builder, runs []Run) {
	for i := 0; i < len(runs); i++ {
		r := runs[i]
		a := r.Attributes
		if link := safeLink(a.Link); link != "" {
			b.WriteString(`<a href="` + html.EscapeString(link) + `">`)
			for ; i < len(runs) && runs[i].Attributes.Link == a.Link; i++ {
				writeRun(b, runs[i])
			}
			i--
			b.WriteString("</a>")
			continue
		}
		writeRun(b, r)
	}
}

func writeRun(b *strings.Builder, r Run) {
	a := r.Attributes
	if at := a.Attachment; at != nil {
		src := at.URL
		w, h := at.size()
		fmt.Fprintf(b, `<img src="%s" alt="%s" width="%d" height="%d">`, html.EscapeString(src), html.EscapeString(at.Name), int(w+0.5), int(h+0.5))
		return
	}
	var style []string
	if a.Font != "" {
		style = append(style, "font-family:"+cssFamily(a.Font))
	}
	if a.Size > 0 {
		style = append(style, "font-size:"+strconv.FormatFloat(float64(a.Size), 'g', 4, 32)+"px")
	}
	if a.Color.A > 0 {
		style = append(style, "color:"+cssColor(a.Color))
	}
	if a.Background.A > 0 {
		style = append(style, "background-color:"+cssColor(a.Background))
	}
	if a.Weight > 0 && a.Weight != 400 && a.Weight != 700 {
		style = append(style, "font-weight:"+strconv.Itoa(a.Weight))
	}
	var tags []string
	if a.Weight == 700 {
		tags = append(tags, "b")
	}
	if a.Italic {
		tags = append(tags, "i")
	}
	if a.Underline {
		tags = append(tags, "u")
	}
	if a.Strikethrough {
		tags = append(tags, "s")
	}
	if a.Code {
		tags = append(tags, "code")
	}
	if len(style) > 0 {
		b.WriteString(`<span style="` + html.EscapeString(strings.Join(style, ";")) + `">`)
	}
	for _, t := range tags {
		b.WriteString("<" + t + ">")
	}
	b.WriteString(html.EscapeString(r.Text))
	for k := len(tags) - 1; k >= 0; k-- {
		b.WriteString("</" + tags[k] + ">")
	}
	if len(style) > 0 {
		b.WriteString("</span>")
	}
}

// cssColor returns c as CSS.
func cssColor(c ui.Color) string {
	if c.A == 255 {
		return fmt.Sprintf("#%02x%02x%02x", c.R, c.G, c.B)
	}
	return fmt.Sprintf("rgba(%d,%d,%d,%.3g)", c.R, c.G, c.B, float32(c.A)/255)
}

// cssFamily returns a family as CSS, quoted when it has a space.
func cssFamily(f string) string {
	if strings.ContainsAny(f, " ,") && !strings.ContainsAny(f, `"'`) {
		return `'` + f + `'`
	}
	return f
}

// parseCSSColor reads a CSS color: #rgb, #rrggbb, rgb() and rgba().
func parseCSSColor(s string) (ui.Color, bool) {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" || s == "transparent" || s == "inherit" || s == "initial" {
		return ui.Color{}, false
	}
	if strings.HasPrefix(s, "#") {
		h := s[1:]
		if len(h) == 3 {
			h = string([]byte{h[0], h[0], h[1], h[1], h[2], h[2]})
		}
		if len(h) != 6 && len(h) != 8 {
			return ui.Color{}, false
		}
		v, err := strconv.ParseUint(h, 16, 32)
		if err != nil {
			return ui.Color{}, false
		}
		if len(h) == 8 {
			return ui.Color{R: uint8(v >> 24), G: uint8(v >> 16), B: uint8(v >> 8), A: uint8(v)}, true
		}
		return ui.RGB(uint8(v>>16), uint8(v>>8), uint8(v)), true
	}
	if inner, ok := strings.CutPrefix(s, "rgba("); ok {
		s = "rgb(" + inner
	}
	if inner, ok := strings.CutPrefix(s, "rgb("); ok {
		inner = strings.TrimSuffix(inner, ")")
		parts := strings.FieldsFunc(inner, func(r rune) bool { return r == ',' || r == ' ' || r == '/' })
		if len(parts) < 3 {
			return ui.Color{}, false
		}
		var c [4]float64
		c[3] = 1
		for i, p := range parts[:min(len(parts), 4)] {
			f, err := strconv.ParseFloat(strings.TrimSuffix(p, "%"), 64)
			if err != nil {
				return ui.Color{}, false
			}
			if strings.HasSuffix(p, "%") {
				f = f / 100 * 255
				if i == 3 {
					f /= 255
				}
			}
			c[i] = f
		}
		if c[3] <= 0 {
			return ui.Color{}, false
		}
		return ui.RGBA(uint8(min(c[0], 255)), uint8(min(c[1], 255)), uint8(min(c[2], 255)), float32(min(c[3], 1))), true
	}
	return ui.Color{}, false
}

// htmlBuilder builds a Text from the nodes of an HTML document.
type htmlBuilder struct {
	paras []Paragraph
	// cur is the paragraph being built, and started whether it has begun.
	cur     Paragraph
	started bool
	// space is set after whitespace, which collapses into one space.
	space bool
	// lists are the lists open: their kinds.
	lists []ListKind
	style ParagraphStyle
	pre   int
}

// ParseHTML returns the attributed text of an HTML document or fragment,
// as a paste from a browser, mail or a word processor brings: paragraphs,
// headings, lists, task lists, quotations, blocks of code, alignment,
// bold, italic, underline, strikethrough, code and links. Fonts, sizes and
// colors are left out, as in Notes, so that what is pasted takes the
// editor's.
func ParseHTML(s string) Text {
	root, err := xhtml.Parse(strings.NewReader(s))
	if err != nil {
		return NewText(s)
	}
	b := htmlBuilder{space: true}
	b.node(root, Attributes{})
	b.flush()
	if len(b.paras) == 0 {
		return Text{}
	}
	// Leading and trailing empty paragraphs are the document's margins.
	paras := b.paras
	for len(paras) > 1 && len(paras[len(paras)-1].Runs) == 0 {
		paras = paras[:len(paras)-1]
	}
	return TextOf(paras...)
}

// flush ends the paragraph being built.
func (b *htmlBuilder) flush() {
	if !b.started {
		return
	}
	p := b.cur
	// Trailing whitespace of a paragraph is not shown.
	if n := len(p.Runs); n > 0 && b.pre == 0 {
		p.Runs[n-1].Text = strings.TrimRight(p.Runs[n-1].Text, " ")
		if p.Runs[n-1].Text == "" {
			p.Runs = p.Runs[:n-1]
		}
	}
	b.paras = append(b.paras, p)
	b.cur, b.started, b.space = Paragraph{}, false, true
}

// begin starts a paragraph of style, unless one is being built.
func (b *htmlBuilder) begin() {
	if !b.started {
		b.cur = Paragraph{Style: b.style}
		b.started = true
	}
}

// text adds text in attributes a.
func (b *htmlBuilder) text(s string, a Attributes) {
	if b.pre > 0 {
		for i, line := range strings.Split(s, "\n") {
			if i > 0 {
				b.begin()
				b.flush()
			}
			if line != "" {
				b.begin()
				b.cur.Runs = append(b.cur.Runs, Run{Text: line, Attributes: a})
			}
		}
		return
	}
	var out strings.Builder
	for _, r := range s {
		if r == ' ' || r == '\n' || r == '\t' || r == '\r' || r == '\f' {
			if !b.space {
				out.WriteByte(' ')
				b.space = true
			}
			continue
		}
		if r == ' ' {
			r = ' '
		}
		out.WriteRune(r)
		b.space = false
	}
	if out.Len() == 0 {
		return
	}
	b.begin()
	t := out.String()
	if len(b.cur.Runs) == 0 {
		t = strings.TrimLeft(t, " ")
		if t == "" {
			return
		}
	}
	b.cur.Runs = append(b.cur.Runs, Run{Text: t, Attributes: a})
}

// node adds n and what it holds, in attributes a.
func (b *htmlBuilder) node(n *xhtml.Node, a Attributes) {
	switch n.Type {
	case xhtml.TextNode:
		b.text(n.Data, a)
		return
	case xhtml.ElementNode:
	default:
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			b.node(c, a)
		}
		return
	}
	saved := b.style
	block := false
	switch n.DataAtom {
	case atom.Script, atom.Style, atom.Head, atom.Title, atom.Template, atom.Noscript:
		return
	case atom.Br:
		b.begin()
		b.flush()
		return
	case atom.Img:
		src, alt := attr(n, "src"), attr(n, "alt")
		at := &Attachment{Name: alt, URL: src}
		at.Width, at.Height = imageSide(attr(n, "width")), imageSide(attr(n, "height"))
		if data, ok := strings.CutPrefix(src, "data:"); ok {
			if _, enc, ok := strings.Cut(data, ";base64,"); ok {
				if raw, err := base64.StdEncoding.DecodeString(enc); err == nil && smallImage(raw) {
					if bm, err := ui.DecodeBitmap(raw); err == nil {
						at.Image = bm
						at.URL = ""
					}
				}
			}
		}
		b.begin()
		x := a.extending()
		x.Attachment = at
		b.cur.Runs = append(b.cur.Runs, Run{Text: string(ObjectReplacement), Attributes: x})
		b.space = false
		return
	case atom.Input:
		if strings.EqualFold(attr(n, "type"), "checkbox") {
			b.begin()
			b.cur.Style.List = Tasks
			_, checked := hasAttr(n, "checked")
			b.cur.Style.Checked = checked
			b.space = true
		}
		return
	case atom.B, atom.Strong:
		a.Weight = 700
	case atom.I, atom.Em, atom.Cite, atom.Dfn:
		a.Italic = true
	case atom.U, atom.Ins:
		a.Underline = true
	case atom.S, atom.Strike, atom.Del:
		a.Strikethrough = true
	case atom.Code, atom.Kbd, atom.Samp, atom.Tt:
		if b.pre == 0 {
			a.Code = true
		}
	case atom.A:
		a.Link = safeLink(attr(n, "href"))
	case atom.Ul, atom.Ol:
		b.flush()
		kind := Bullets
		if n.DataAtom == atom.Ol {
			kind = Numbers
		}
		b.lists = append(b.lists, kind)
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			b.node(c, a)
		}
		b.lists = b.lists[:len(b.lists)-1]
		b.flush()
		b.style = saved
		return
	case atom.Li:
		b.flush()
		kind := Bullets
		if k := len(b.lists); k > 0 {
			kind = b.lists[k-1]
		}
		b.style = ParagraphStyle{List: kind, Indent: max(len(b.lists)-1, 0), Align: saved.Align, Quote: saved.Quote}
		block = true
	case atom.P, atom.Div, atom.Section, atom.Article, atom.Header, atom.Footer, atom.Main, atom.Aside, atom.Nav,
		atom.Figure, atom.Figcaption, atom.Tr, atom.Dt, atom.Dd, atom.Address, atom.Center:
		b.flush()
		block = true
	case atom.H1, atom.H2, atom.H3, atom.H4, atom.H5, atom.H6:
		b.flush()
		b.style = ParagraphStyle{Heading: int(n.Data[1] - '0'), Align: saved.Align, Quote: saved.Quote}
		block = true
	case atom.Blockquote:
		b.flush()
		b.style.Quote = true
		block = true
	case atom.Pre:
		b.flush()
		b.style = ParagraphStyle{CodeBlock: true, Indent: saved.Indent, Quote: saved.Quote}
		b.pre++
		defer func() { b.pre-- }()
		block = true
	case atom.Td, atom.Th:
		b.text(" ", a)
	}
	if style := attr(n, "style"); style != "" {
		a = applyCSS(style, a, &b.style, block)
	}
	if block {
		switch strings.ToLower(attr(n, "align")) {
		case "center":
			b.style.Align = ui.Center
		case "right":
			b.style.Align = ui.End
		}
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		b.node(c, a)
	}
	if block {
		if n.DataAtom == atom.P && !b.started && b.pre == 0 {
			// An empty paragraph shows as one.
			b.begin()
		}
		b.flush()
		b.style = saved
	}
}

// applyCSS applies the declarations of a style attribute to a run's
// attributes and, for a block, its paragraph's style: weights, italics,
// lines and alignment; fonts, sizes and colors are left out.
func applyCSS(style string, a Attributes, ps *ParagraphStyle, block bool) Attributes {
	for _, decl := range strings.Split(style, ";") {
		name, value, ok := strings.Cut(decl, ":")
		if !ok {
			continue
		}
		name = strings.TrimSpace(strings.ToLower(name))
		value = strings.TrimSpace(strings.ToLower(value))
		value = strings.TrimSuffix(strings.TrimSpace(strings.TrimSuffix(value, "!important")), ";")
		switch name {
		case "font-weight":
			switch value {
			case "bold", "bolder":
				a.Weight = 700
			case "normal", "lighter":
				a.Weight = 0
			default:
				if w, err := strconv.Atoi(value); err == nil {
					if w >= 600 {
						a.Weight = 700
					} else {
						a.Weight = 0
					}
				}
			}
		case "font-style":
			a.Italic = value == "italic" || value == "oblique"
		case "text-decoration", "text-decoration-line":
			a.Underline = a.Underline || strings.Contains(value, "underline")
			a.Strikethrough = a.Strikethrough || strings.Contains(value, "line-through")
			if value == "none" {
				a.Underline, a.Strikethrough = false, false
			}
		case "text-align":
			if block {
				switch value {
				case "center":
					ps.Align = ui.Center
				case "right", "end":
					ps.Align = ui.End
				case "left", "start", "justify":
					ps.Align = ui.Start
				}
			}
		}
	}
	return a
}

// attr returns the attribute key of n.
func attr(n *xhtml.Node, key string) string {
	v, _ := hasAttr(n, key)
	return v
}

func hasAttr(n *xhtml.Node, key string) (string, bool) {
	for _, a := range n.Attr {
		if strings.EqualFold(a.Key, key) {
			return a.Val, true
		}
	}
	return "", false
}
