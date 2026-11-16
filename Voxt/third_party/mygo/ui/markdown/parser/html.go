package parser

import (
	"strconv"
	"strings"
)

// HTML returns the document as HTML, as cmark-gfm renders it, raw HTML
// included as written.
func (d *Document) HTML() string {
	var w htmlWriter
	w.d = d
	w.blocks(0)
	w.footnotes()
	return w.b.String()
}

type htmlWriter struct {
	d     *Document
	b     strings.Builder
	notes []string // footnote labels in the order of their first reference
}

func (w *htmlWriter) cr() {
	if s := w.b.Len(); s > 0 {
		str := w.b.String()
		if str[s-1] != '\n' {
			w.b.WriteByte('\n')
		}
	}
}

func (w *htmlWriter) blocks(parent int) {
	for i := range w.d.Children(parent) {
		w.block(i)
	}
}

func (w *htmlWriter) tight(i int) bool {
	b := w.d.Block(i)
	if b.Parent <= 0 {
		return false
	}
	p := w.d.Block(b.Parent)
	if p.Kind != Item || p.Parent < 0 {
		return false
	}
	return w.d.Block(p.Parent).Tight
}

func (w *htmlWriter) block(i int) {
	d := w.d
	b := d.Block(i)
	switch b.Kind {
	case Paragraph:
		if w.tight(i) {
			w.inlines(i)
			return
		}
		w.cr()
		w.b.WriteString("<p>")
		w.inlines(i)
		w.b.WriteString("</p>\n")
	case Heading:
		w.cr()
		tag := "h" + strconv.Itoa(b.Level)
		w.b.WriteString("<" + tag + ">")
		w.inlines(i)
		w.b.WriteString("</" + tag + ">\n")
	case ThematicBreak:
		w.cr()
		w.b.WriteString("<hr />\n")
	case CodeBlock:
		w.cr()
		w.b.WriteString("<pre><code")
		if lang := b.Language(); lang != "" {
			w.b.WriteString(` class="language-`)
			escapeHTML(&w.b, lang)
			w.b.WriteString(`"`)
		}
		w.b.WriteString(">")
		escapeHTML(&w.b, d.Lines(i))
		w.b.WriteString("</code></pre>\n")
	case HTMLBlock:
		w.cr()
		w.b.WriteString(d.Lines(i))
		w.cr()
	case MathBlock:
		w.cr()
		w.b.WriteString(`<div class="math display">`)
		escapeHTML(&w.b, strings.TrimSuffix(d.Lines(i), "\n"))
		w.b.WriteString("</div>\n")
	case Quote:
		w.cr()
		w.b.WriteString("<blockquote>\n")
		w.blocks(i)
		w.cr()
		w.b.WriteString("</blockquote>\n")
	case List:
		w.cr()
		if b.Ordered {
			if b.Number != 1 {
				w.b.WriteString(`<ol start="` + strconv.Itoa(b.Number) + `">` + "\n")
			} else {
				w.b.WriteString("<ol>\n")
			}
		} else {
			w.b.WriteString("<ul>\n")
		}
		w.blocks(i)
		if b.Ordered {
			w.b.WriteString("</ol>\n")
		} else {
			w.b.WriteString("</ul>\n")
		}
	case Item:
		w.cr()
		w.b.WriteString("<li>")
		switch b.Task {
		case Unchecked:
			w.b.WriteString(`<input type="checkbox" disabled="" /> `)
		case Checked:
			w.b.WriteString(`<input type="checkbox" checked="" disabled="" /> `)
		}
		w.blocks(i)
		w.b.WriteString("</li>\n")
	case Table:
		w.cr()
		w.b.WriteString("<table>\n")
		body := false
		for r := range d.Children(i) {
			row := d.Block(r)
			if row.Header {
				w.b.WriteString("<thead>\n")
			} else if !body {
				w.b.WriteString("<tbody>\n")
				body = true
			}
			w.b.WriteString("<tr>\n")
			k := 0
			for c := range d.Children(r) {
				tag := "td"
				if row.Header {
					tag = "th"
				}
				w.b.WriteString("<" + tag)
				if k < len(b.Align) {
					switch b.Align[k] {
					case AlignLeft:
						w.b.WriteString(` align="left"`)
					case AlignCenter:
						w.b.WriteString(` align="center"`)
					case AlignRight:
						w.b.WriteString(` align="right"`)
					}
				}
				w.b.WriteString(">")
				w.inlines(c)
				w.b.WriteString("</" + tag + ">\n")
				k++
			}
			w.b.WriteString("</tr>\n")
			if row.Header {
				w.b.WriteString("</thead>\n")
			}
		}
		if body {
			w.b.WriteString("</tbody>\n")
		}
		w.b.WriteString("</table>\n")
	case FootnoteDefinition:
		// Listed at the end, in the order of their references.
	}
}

// Lines returns the content of a code, HTML or math block, each line
// ending with a newline.
func (d *Document) Lines(i int) string {
	segs := d.segsOf(&d.blocks[i])
	n := 0
	for _, s := range segs {
		n += int(s.pad) + int(s.end-s.start) + 1
	}
	var sb strings.Builder
	sb.Grow(n)
	for _, s := range segs {
		for range s.pad {
			sb.WriteByte(' ')
		}
		sb.WriteString(d.src[s.start:s.end])
		sb.WriteByte('\n')
	}
	return sb.String()
}

// LineCount returns the number of lines of a code, HTML or math block.
func (d *Document) LineCount(i int) int { b := &d.blocks[i]; return int(b.segB - b.segA) }

// Line returns line k of a code, HTML or math block, without its newline,
// and the offset in the source of its first byte.
func (d *Document) Line(i, k int) (string, int) {
	s := d.segs[int(d.blocks[i].segA)+k]
	line := d.src[s.start:s.end]
	if s.pad > 0 {
		line = strings.Repeat(" ", int(s.pad)) + line
	}
	return line, int(s.start)
}

func (w *htmlWriter) inlines(i int) {
	in := w.d.Inlines(i)
	w.inline(in, 0, len(in))
}

func (w *htmlWriter) inline(in []Inline, from, to int) {
	for k := from; k < to; k = in[k].Next {
		n := &in[k]
		switch n.Kind {
		case Text:
			escapeHTML(&w.b, n.Text)
		case SoftBreak:
			w.b.WriteString("\n")
		case LineBreak:
			w.b.WriteString("<br />\n")
		case Code:
			w.b.WriteString("<code>")
			escapeHTML(&w.b, n.Text)
			w.b.WriteString("</code>")
		case Emphasis:
			w.b.WriteString("<em>")
			w.inline(in, k+1, n.Next)
			w.b.WriteString("</em>")
		case Strong:
			w.b.WriteString("<strong>")
			w.inline(in, k+1, n.Next)
			w.b.WriteString("</strong>")
		case Strikethrough:
			w.b.WriteString("<del>")
			w.inline(in, k+1, n.Next)
			w.b.WriteString("</del>")
		case Link:
			w.b.WriteString(`<a href="`)
			escapeHref(&w.b, n.Dest)
			w.b.WriteString(`"`)
			if n.Title != "" {
				w.b.WriteString(` title="`)
				escapeHTML(&w.b, n.Title)
				w.b.WriteString(`"`)
			}
			w.b.WriteString(">")
			w.inline(in, k+1, n.Next)
			w.b.WriteString("</a>")
		case Image:
			w.b.WriteString(`<img src="`)
			escapeHref(&w.b, n.Dest)
			w.b.WriteString(`" alt="`)
			escapeHTML(&w.b, PlainText(in[k+1:n.Next]))
			w.b.WriteString(`"`)
			if n.Title != "" {
				w.b.WriteString(` title="`)
				escapeHTML(&w.b, n.Title)
				w.b.WriteString(`"`)
			}
			w.b.WriteString(" />")
		case HTML:
			w.b.WriteString(n.Text)
		case Math:
			if n.Display {
				w.b.WriteString(`<span class="math display">`)
			} else {
				w.b.WriteString(`<span class="math inline">`)
			}
			escapeHTML(&w.b, n.Text)
			w.b.WriteString("</span>")
		case FootnoteRef:
			num := w.note(n.Text)
			id := strconv.Itoa(num)
			w.b.WriteString(`<sup class="footnote-ref"><a href="#fn-` + id + `" id="fnref-` + id + `" data-footnote-ref>` + id + `</a></sup>`)
		}
	}
}

// note returns the number of footnote label, numbering it at its first
// reference.
func (w *htmlWriter) note(label string) int {
	key := normalizeLabel(label)
	for i, l := range w.notes {
		if l == key {
			return i + 1
		}
	}
	w.notes = append(w.notes, key)
	return len(w.notes)
}

func (w *htmlWriter) footnotes() {
	if len(w.notes) == 0 {
		return
	}
	w.cr()
	w.b.WriteString("<section class=\"footnotes\" data-footnotes>\n<ol>\n")
	for k := 0; k < len(w.notes); k++ {
		i := w.d.notes[w.notes[k]]
		id := strconv.Itoa(k + 1)
		w.b.WriteString(`<li id="fn-` + id + `">` + "\n")
		w.blocks(i)
		w.cr()
		w.b.WriteString(`<a href="#fnref-` + id + `" class="footnote-backref">↩</a>` + "\n</li>\n")
	}
	w.b.WriteString("</ol>\n</section>\n")
}

// PlainText returns the text of inlines without their markup: the text
// of links and emphasis, code as its text, line breaks as spaces.
func PlainText(in []Inline) string {
	var b strings.Builder
	for k := range in {
		switch n := &in[k]; n.Kind {
		case Text, Code, HTML, Math:
			b.WriteString(n.Text)
		case SoftBreak, LineBreak:
			b.WriteByte(' ')
		case FootnoteRef:
			b.WriteString("[^" + n.Text + "]")
		}
	}
	return b.String()
}

func escapeHTML(b *strings.Builder, s string) {
	last := 0
	for i := 0; i < len(s); i++ {
		var r string
		switch s[i] {
		case '&':
			r = "&amp;"
		case '<':
			r = "&lt;"
		case '>':
			r = "&gt;"
		case '"':
			r = "&quot;"
		case 0:
			r = "�"
		default:
			continue
		}
		b.WriteString(s[last:i])
		b.WriteString(r)
		last = i + 1
	}
	b.WriteString(s[last:])
}

// hrefSafe are the bytes a URL keeps as they are, as cmark's houdini.
var hrefSafe = func() (t [256]bool) {
	for c := 'a'; c <= 'z'; c++ {
		t[c] = true
	}
	for c := 'A'; c <= 'Z'; c++ {
		t[c] = true
	}
	for c := '0'; c <= '9'; c++ {
		t[c] = true
	}
	for _, c := range "-_.+!*(),%#@?=;:/,+$~" {
		t[c] = true
	}
	return
}()

func escapeHref(b *strings.Builder, s string) {
	const hex = "0123456789ABCDEF"
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case hrefSafe[c]:
			b.WriteByte(c)
		case c == '&':
			b.WriteString("&amp;")
		case c == '\'':
			b.WriteString("&#x27;")
		default:
			b.WriteByte('%')
			b.WriteByte(hex[c>>4])
			b.WriteByte(hex[c&15])
		}
	}
}
