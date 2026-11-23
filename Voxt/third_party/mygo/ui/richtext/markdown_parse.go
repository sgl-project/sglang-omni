package richtext

import (
	"strings"

	"github.com/egoist/mygo/ui/markdown/parser"
)

// ParseMarkdown returns the attributed text of Markdown, CommonMark with
// GitHub's extensions and footnotes, as Markdown writes it: headings,
// lists and task lists, quotations, blocks of code, bold, italic,
// strikethrough, code, links, images, and the HTML tags of these marks
// and of underlines, <u>. A list item's paragraphs after its first are
// indented under it; a table's rows are paragraphs of their cells apart
// by tabs, the header's bold. Math, which the editor does not show, stays
// as written: \[ is an escaped bracket, as everywhere but in math.
func ParseMarkdown(s string) Text {
	d := parser.New(parser.GFM | parser.ExtFootnotes)
	d.Set(s)
	m := mdReader{d: d}
	m.blocks(0, ParagraphStyle{}, -1)
	if len(m.paras) == 0 {
		return Text{}
	}
	return TextOf(m.paras...)
}

// mdReader reads the blocks and inlines of a Markdown document into
// paragraphs.
type mdReader struct {
	d     *parser.Document
	paras []Paragraph
	// html counts the inline HTML tags open, by the attribute they turn
	// on: <u>, <s>, <b> and <i> and their synonyms.
	html [4]int
}

// blocks adds the children of block i in style st; depth is the level of
// the list they are in, -1 outside lists.
func (m *mdReader) blocks(i int, st ParagraphStyle, depth int) {
	for k := range m.d.Children(i) {
		m.block(k, st, depth)
	}
}

func (m *mdReader) block(i int, st ParagraphStyle, depth int) {
	b := m.d.Block(i)
	switch b.Kind {
	case parser.Paragraph:
		m.inlinePara(st, m.d.Inlines(i), Attributes{})
	case parser.Heading:
		st.Heading = b.Level
		m.inlinePara(st, m.d.Inlines(i), Attributes{})
	case parser.CodeBlock:
		st.CodeBlock, st.Heading = true, 0
		n := m.d.LineCount(i)
		for k := range n {
			line, _ := m.d.Line(i, k)
			p := Paragraph{Style: st}
			if line != "" {
				p.Runs = []Run{{Text: line}}
			}
			m.paras = append(m.paras, p)
		}
		if n == 0 {
			m.paras = append(m.paras, Paragraph{Style: st})
		}
	case parser.HTMLBlock:
		for _, p := range ParseHTML(m.d.Lines(i)).Paragraphs() {
			p.Style.Quote = p.Style.Quote || st.Quote
			p.Style.Indent += st.Indent
			m.paras = append(m.paras, p)
		}
	case parser.Quote:
		st.Quote = true
		m.blocks(i, st, depth)
	case parser.List:
		kind := Bullets
		if b.Ordered {
			kind = Numbers
		}
		for it := range m.d.Children(i) {
			ist := ParagraphStyle{List: kind, Indent: depth + 1, Quote: st.Quote}
			if task := m.d.Block(it).Task; task != parser.NoTask {
				ist.List, ist.Checked = Tasks, task == parser.Checked
			}
			m.item(it, ist, depth+1)
		}
	case parser.Table:
		for row := range m.d.Children(i) {
			p := Paragraph{Style: st}
			var a Attributes
			if m.d.Block(row).Header {
				a.Weight = 700
			}
			first := true
			for cell := range m.d.Children(row) {
				if !first {
					p.Runs = addMDRun(p.Runs, Run{Text: "\t", Attributes: a})
				}
				first = false
				for _, r := range m.runs(m.d.Inlines(cell), a) {
					p.Runs = addMDRun(p.Runs, r)
				}
			}
			m.paras = append(m.paras, p)
		}
	case parser.FootnoteDefinition:
		from := len(m.paras)
		m.blocks(i, st, depth)
		if from < len(m.paras) {
			p := &m.paras[from]
			p.Runs = append([]Run{{Text: "[" + b.Label() + "] "}}, p.Runs...)
		}
	}
}

// item adds a list item: its first paragraph in the item's style ist, and
// what follows indented under it.
func (m *mdReader) item(i int, ist ParagraphStyle, depth int) {
	under := ParagraphStyle{Indent: depth + 1, Quote: ist.Quote}
	first := true
	for k := range m.d.Children(i) {
		b := m.d.Block(k)
		if first {
			first = false
			if b.Kind == parser.Paragraph || b.Kind == parser.Heading {
				m.block(k, ist, depth)
				continue
			}
			// An item starting with code or a list has an empty line of
			// its own, as it shows.
			m.paras = append(m.paras, Paragraph{Style: ist})
		}
		m.block(k, under, depth)
	}
	if first {
		m.paras = append(m.paras, Paragraph{Style: ist})
	}
}

// inlinePara adds a paragraph of inlines in style st; a hard line break
// starts another, indented under a list item rather than an item of its
// own.
func (m *mdReader) inlinePara(st ParagraphStyle, in []parser.Inline, a Attributes) {
	runs := m.runs(in, a)
	p := Paragraph{Style: st}
	for _, r := range runs {
		for {
			line, rest, br := strings.Cut(r.Text, "\n")
			if line != "" {
				p.Runs = append(p.Runs, Run{Text: line, Attributes: r.Attributes})
			}
			if !br {
				break
			}
			m.paras = append(m.paras, p)
			if st.List != NoList {
				st = ParagraphStyle{Indent: st.Indent + 1, Quote: st.Quote}
			}
			p = Paragraph{Style: st}
			r.Text = rest
		}
	}
	m.paras = append(m.paras, p)
}

// runs returns the runs of inlines, in attributes a; hard line breaks
// are newlines.
func (m *mdReader) runs(in []parser.Inline, a Attributes) []Run {
	var out []Run
	m.html = [4]int{}
	m.inlines(&out, in, 0, len(in), a)
	return out
}

func (m *mdReader) inlines(out *[]Run, in []parser.Inline, from, to int, a Attributes) {
	add := func(s string, a Attributes) {
		if m.html[0] > 0 {
			a.Underline = true
		}
		if m.html[1] > 0 {
			a.Strikethrough = true
		}
		if m.html[2] > 0 {
			a.Weight = 700
		}
		if m.html[3] > 0 {
			a.Italic = true
		}
		*out = addMDRun(*out, Run{Text: s, Attributes: a})
	}
	for k := from; k < to; k = in[k].Next {
		n := &in[k]
		x := a
		switch n.Kind {
		case parser.Text:
			add(n.Text, a)
		case parser.SoftBreak:
			add(" ", a)
		case parser.LineBreak:
			add("\n", a)
		case parser.Code:
			x.Code = true
			add(n.Text, x)
		case parser.Emphasis:
			x.Italic = true
			m.inlines(out, in, k+1, n.Next, x)
		case parser.Strong:
			x.Weight = 700
			m.inlines(out, in, k+1, n.Next, x)
		case parser.Strikethrough:
			x.Strikethrough = true
			m.inlines(out, in, k+1, n.Next, x)
		case parser.Link:
			x.Link = safeLink(n.Dest)
			m.inlines(out, in, k+1, n.Next, x)
		case parser.Image:
			x = a.extending()
			x.Attachment = &Attachment{Name: parser.PlainText(in[k+1 : n.Next]), URL: safeLink(n.Dest)}
			*out = append(*out, Run{Text: string(ObjectReplacement), Attributes: x})
		case parser.HTML:
			m.tag(n.Text, out, a)
		case parser.FootnoteRef:
			add("["+n.Text+"]", a)
		}
	}
}

// tag applies an inline HTML tag: the opening and closing tags of
// underlines, strikethrough, bold and italic, and line breaks. Other
// tags are left out, as a browser shows none.
func (m *mdReader) tag(s string, out *[]Run, a Attributes) {
	s = strings.ToLower(strings.TrimSpace(s))
	if !strings.HasPrefix(s, "<") || !strings.HasSuffix(s, ">") {
		return
	}
	s = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(s, "<"), ">"))
	closing := strings.HasPrefix(s, "/")
	s = strings.TrimPrefix(s, "/")
	s = strings.TrimSuffix(s, "/")
	name, _, _ := strings.Cut(strings.TrimSpace(s), " ")
	which := -1
	switch name {
	case "br":
		*out = append(*out, Run{Text: "\n", Attributes: a})
		return
	case "u", "ins":
		which = 0
	case "s", "strike", "del":
		which = 1
	case "b", "strong":
		which = 2
	case "i", "em":
		which = 3
	}
	if which < 0 {
		return
	}
	if closing {
		m.html[which] = max(m.html[which]-1, 0)
	} else {
		m.html[which]++
	}
}

// addMDRun appends r to runs, joined to the last when their attributes
// are the same.
func addMDRun(runs []Run, r Run) []Run {
	if n := len(runs); n > 0 && runs[n-1].Attributes == r.Attributes && r.Attributes.Attachment == nil {
		runs[n-1].Text += r.Text
		return runs
	}
	return append(runs, r)
}
