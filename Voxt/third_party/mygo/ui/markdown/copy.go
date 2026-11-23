package markdown

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown/parser"
)

// Copying: as text, a view joins its paragraphs with blank lines, a code
// block's lines and a tight list's items with newlines and a table's
// cells with tabs; as Markdown, it copies the source between the
// selection's ends, with the syntax of emphasis, code and links cut by
// them completed.

func (st *viewState) elem(id uint64) (elemInfo, bool) {
	for _, e := range st.elems {
		if e.id == id {
			return e, true
		}
	}
	return elemInfo{}, false
}

// copyText returns the text of the selected parts.
func (st *viewState) copyText(parts []ui.SelectedText) string {
	var b strings.Builder
	var prev elemInfo
	for k, p := range parts {
		inf, ok := st.elem(p.ID)
		if k > 0 {
			if ok {
				b.WriteString(st.separator(prev, inf))
			} else {
				b.WriteString("\n")
			}
		}
		b.WriteString(runeSlice(p.Text, p.Start, p.End))
		prev = inf
	}
	return b.String()
}

// copyData returns the selection as Copy puts it on the clipboard: as
// text, as HTML and as Markdown.
func (st *viewState) copyData(parts []ui.SelectedText) transfer.Data {
	md := st.copyMarkdown(parts)
	d := parser.New(st.doc.Extensions())
	d.Set(md)
	return transfer.New(transfer.NewItem(
		transfer.Bytes(transfer.Text, []byte(st.copyText(parts))),
		transfer.Bytes(transfer.HTML, []byte(`<meta charset="utf-8">`+d.HTML())),
		transfer.Bytes(transfer.Markdown, []byte(md)),
	))
}

// markdownData returns Markdown as Copy as Markdown puts it on the
// clipboard: as text, and as Markdown.
func markdownData(md string) transfer.Data {
	return transfer.New(transfer.NewItem(
		transfer.Bytes(transfer.Text, []byte(md)),
		transfer.Bytes(transfer.Markdown, []byte(md)),
	))
}

func runeSlice(s string, from, to int) string {
	i, n := 0, 0
	a, z := len(s), len(s)
	for i < len(s) {
		if n == from {
			a = i
		}
		if n == to {
			z = i
			break
		}
		_, w := utf8.DecodeRuneInString(s[i:])
		i += w
		n++
	}
	if from >= n && a == len(s) {
		a = len(s)
	}
	return s[a:max(a, z)]
}

// separator returns what goes between the text of two blocks copied.
func (st *viewState) separator(a, b elemInfo) string {
	d := st.doc
	if a.block == b.block {
		return "\n"
	}
	ab, bb := d.Block(a.block), d.Block(b.block)
	if ab.Kind == parser.TableCell && bb.Kind == parser.TableCell {
		if ab.Parent == bb.Parent {
			return "\t"
		}
		return "\n"
	}
	// Items of the same tight list.
	if l := st.listOf(b.block); l > 0 && d.Block(l).Tight && st.listOf(a.block) == l || st.listOf(a.block) > 0 && d.Block(st.listOf(a.block)).Tight && st.within(b.block, st.listOf(a.block)) {
		return "\n"
	}
	return "\n\n"
}

// listOf returns the nearest list around block i, or 0.
func (st *viewState) listOf(i int) int {
	for p := st.doc.Block(i).Parent; p > 0; p = st.doc.Block(p).Parent {
		if st.doc.Block(p).Kind == parser.List {
			return p
		}
	}
	return 0
}

func (st *viewState) within(i, anc int) bool {
	for p := st.doc.Block(i).Parent; p >= 0; p = st.doc.Block(p).Parent {
		if p == anc {
			return true
		}
	}
	return false
}

// srcAt returns the source offset of rune r of an element's text, for
// the end of a selection when end.
func (st *viewState) srcAt(inf elemInfo, r int, end bool) int {
	d := st.doc
	b := d.Block(inf.block)
	switch b.Kind {
	case parser.CodeBlock, parser.HTMLBlock, parser.MathBlock:
		// Lines from inf.line on, each followed by a newline.
		n := d.LineCount(inf.block)
		for k := max(inf.line, 0); k < n; k++ {
			line, at := d.Line(inf.block, k)
			c := utf8.RuneCountInString(line)
			if r <= c || k == n-1 {
				// Spaces standing for a tab partly taken by indentation
				// are not the source's.
				pad := linePad(d.Source(), line, at)
				off := at
				for j := pad; j < r && off < len(d.Source()); j++ {
					_, w := utf8.DecodeRuneInString(d.Source()[off:])
					off += w
				}
				return off
			}
			r -= c + 1
		}
		return b.Start
	}
	if inf.block < len(st.paras) && st.paras[inf.block].ok {
		return st.paras[inf.block].p.sourceAt(d.Source(), r, end)
	}
	return b.Start
}

// copyMarkdown returns the Markdown of the selected parts, or of the
// whole document when none is selected.
func (st *viewState) copyMarkdown(parts []ui.SelectedText) string {
	d := st.doc
	if len(parts) == 0 {
		return d.Source()
	}
	first, last := parts[0], parts[len(parts)-1]
	fi, ok1 := st.elem(first.ID)
	li, ok2 := st.elem(last.ID)
	if !ok1 || !ok2 {
		return st.copyText(parts)
	}
	fb, lb := d.Block(fi.block), d.Block(li.block)
	a := st.srcAt(fi, first.Start, false)
	b := st.srcAt(li, last.End, true)
	atStart := first.Start == 0 && fi.line <= 0
	atEnd := last.End >= utf8.RuneCountInString(last.Text)
	if lb.Kind == parser.CodeBlock && atEnd {
		// Only the last chunk of lines ends the block.
		atEnd = li.line+chunkLines >= d.LineCount(li.block)
	}
	// From a block's start, or to its end, its syntax too: a heading's #,
	// a list's marker, a quote's >, a table row's pipes, a code block's
	// fences when it is selected whole.
	switch {
	case !atStart:
	case fb.Kind == parser.TableCell:
		// A row's pipes go with it when cells after it are selected too.
		if fi.block == fb.Parent+1 && li.block != fi.block {
			a = lineStartOf(d.Source(), fb.Start)
		}
	case fb.Kind == parser.CodeBlock:
		if fb.Fenced && atEnd && fi.block == li.block {
			a = lineStartOf(d.Source(), fb.Start)
		}
	default:
		a = lineStartOf(d.Source(), fb.Start)
	}
	switch {
	case !atEnd:
	case lb.Kind == parser.TableCell:
		if lb.Next == d.Block(lb.Parent).Next && li.block != fi.block {
			b = lineEndOf(d.Source(), lb.End)
		}
	case lb.Kind == parser.CodeBlock:
		if lb.Fenced && !lb.Open && atStart && fi.block == li.block {
			b = lb.End
		}
	default:
		b = lineEndOf(d.Source(), lb.End)
	}
	if b < a {
		a, b = b, a
	}
	prefix := st.openAt(fi.block, a)
	suffix := st.closeAt(li.block, b)
	return prefix + d.Source()[a:b] + suffix
}

func lineEndOf(src string, off int) int {
	for off < len(src) && src[off] != '\n' && src[off] != '\r' {
		off++
	}
	return off
}

// openAt returns the opening syntax of the inline spans of block i that
// offset at falls inside, outermost first.
func (st *viewState) openAt(i, at int) string {
	in := st.doc.Inlines(i)
	var b strings.Builder
	for k := range in {
		n := &in[k]
		if n.Start < at && at < n.End {
			if open := st.openSyntax(in, k); open != "" {
				b.WriteString(open)
			}
		}
	}
	return b.String()
}

// closeAt returns the closing syntax of the inline spans of block i that
// offset at falls inside, innermost first.
func (st *viewState) closeAt(i, at int) string {
	in := st.doc.Inlines(i)
	var closes []string
	for k := range in {
		n := &in[k]
		if n.Start < at && at < n.End {
			if c := st.closeSyntax(in, k); c != "" {
				closes = append(closes, c)
			}
		}
	}
	slices.Reverse(closes)
	return strings.Join(closes, "")
}

func (st *viewState) openSyntax(in []parser.Inline, k int) string {
	n := &in[k]
	src := st.doc.Source()
	switch n.Kind {
	case parser.Emphasis, parser.Strong, parser.Strikethrough, parser.Link:
		if k+1 < n.Next {
			return src[n.Start:in[k+1].Start]
		}
	case parser.Code:
		return src[n.Start:codeStart(src, n)]
	}
	return ""
}

func (st *viewState) closeSyntax(in []parser.Inline, k int) string {
	n := &in[k]
	src := st.doc.Source()
	switch n.Kind {
	case parser.Emphasis, parser.Strong, parser.Strikethrough, parser.Link:
		if k+1 < n.Next {
			// The end of its last child.
			end := in[k+1].End
			for j := k + 1; j < n.Next; j = in[j].Next {
				end = in[j].End
			}
			return src[end:n.End]
		}
	case parser.Code:
		ticks := codeStart(src, n) - n.Start
		return src[n.End-ticks : n.End]
	}
	return ""
}

// linePad returns how many spaces a code line starts with that are not
// the source's at, standing for a tab partly taken by indentation.
func linePad(src, line string, at int) int {
	pad := 0
	for pad < len(line) && line[pad] == ' ' && !strings.HasPrefix(src[at:], line[pad:]) {
		pad++
	}
	return pad
}

func lineStartOf(src string, off int) int {
	for off > 0 && src[off-1] != '\n' && src[off-1] != '\r' {
		off--
	}
	return off
}

// isEscapable reports whether a backslash escapes c: ASCII punctuation.
func isEscapable(c byte) bool {
	return c >= '!' && c <= '/' || c >= ':' && c <= '@' || c >= '[' && c <= '`' || c >= '{' && c <= '~'
}

func isAlnum(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }
