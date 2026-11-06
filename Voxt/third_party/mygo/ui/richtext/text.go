package richtext

import (
	"github.com/egoist/mygo/ui"
	"iter"
	"slices"
	"strings"
	"sync/atomic"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/rope"
)

// Text is text whose runs have attributes of their own, in
// paragraphs that have a style of their own: what an Editor edits,
// as Foundation's AttributedString is what SwiftUI's TextEditor edits. It
// is a value: methods return a new text and leave the one they are called
// on as it was, sharing what did not change with it, so that keeping a
// version (for undo, to save it later, to compare) costs a pointer, and a
// version can be read from any goroutine while others are made. The zero
// value is the empty text.
//
// Offsets count runes, newlines included, as Element.TextSelection does.
// Paragraphs end at newlines; the style of a paragraph and the attributes
// of its newline belong to it, as in word processors, so that an empty
// paragraph remembers the attributes typing gets there. Runs are kept as
// long as they can be: neighbors with the same attributes are one run.
type Text struct {
	text  *rope.Node
	paras *seq[richPara]
}

// Attributes are the attributes of a run of an Text. What
// they leave zero takes the style of the editor showing the text (its
// font, size and color), and Italic, Underline, Strikethrough and Code only
// turn those on. They are comparable: runs with equal attributes are one.
type Attributes struct {
	// Font is a family, as for Element.Font.
	Font string
	// Size is in DIPs.
	Size float32
	// Weight goes from 100 (thin) to 900 (black): 700 is bold.
	Weight        int
	Italic        bool
	Underline     bool
	Strikethrough bool
	// Color colors the text, Background highlights it.
	Color, Background ui.Color
	// Code shows inline code, in the editor's monospaced font. Typing at
	// its edges does not continue it.
	Code bool
	// Link is a URL the run links to. Typing at its edges does not
	// continue it, as in NSTextView.
	Link string
	// Tag is the app's own mark over the run, as a mention or a comment's
	// ID: typing at its edges does not continue it, and the app styles it
	// with EditorOptions.TagStyle.
	Tag string
	// Attachment is what an object replacement character (U+FFFC) shows:
	// an image, or a drawing of the app's, inside the line. Only that
	// character takes it.
	Attachment *Attachment
}

// Bold reports whether the run is bold: its weight is 600 or more.
func (a Attributes) Bold() bool { return a.Weight >= 600 }

// extends reports whether typing right after a run with the attributes
// continues it. Links, code and tags hold only what they hold, as
// NSTextView and the expand rules of Peritext and Loro leave them; every
// other attribute carries on to what is typed after it.
func (a Attributes) extending() Attributes {
	a.Link, a.Code, a.Tag, a.Attachment = "", false, "", nil
	return a
}

// Attachment is an object inside an Text, as an image in a note:
// one object replacement character (U+FFFC) whose Attributes hold it,
// which the editor lays out as a box of its size inside the line, on the
// baseline, and selects, copies and deletes as one character.
type Attachment struct {
	// Image is drawn in the box, unless Draw draws it.
	Image ui.ImageSource
	// Width and Height are the size of the box in DIPs; zero takes the
	// image's size. A box wider than the editor is scaled down to fit.
	Width, Height float32
	// Name describes the attachment to assistive technology, and stands
	// for it in plain text: an image's alternative text, a file's name.
	Name string
	// URL is where the attachment comes from, as a file's or an image's,
	// which HTML and Markdown write for it.
	URL string
	// Draw draws the attachment in r, instead of Image, as an app's chip
	// for a mention or a file.
	Draw func(p *ui.Painter, r ui.Rect)
	// Data is the app's.
	Data any
}

// size returns the size of the box of the attachment.
func (a *Attachment) size() (w, h float32) {
	w, h = a.Width, a.Height
	if a.Image != nil && (w <= 0 || h <= 0) {
		iw, ih := imageSize(a.Image)
		switch {
		case w <= 0 && h <= 0:
			w, h = iw, ih
		case w <= 0 && ih > 0:
			w = h * iw / ih
		case h <= 0 && iw > 0:
			h = w * ih / iw
		}
	}
	return max(w, 1), max(h, 1)
}

// imageSize returns the size an image shows at, in DIPs.
func imageSize(src ui.ImageSource) (float32, float32) {
	switch s := src.(type) {
	case *ui.Bitmap:
		w, h := s.Size()
		return float32(w), float32(h)
	case *ui.SVG:
		return s.Size()
	}
	return 0, 0
}

// ObjectReplacement is the character an Attachment takes in a text.
const ObjectReplacement = '￼'

// ListKind is what a paragraph is an item of.
type ListKind uint8

const (
	NoList ListKind = iota
	// Bullets is an unordered list.
	Bullets
	// Numbers is an ordered list, numbered from 1 among the items of one
	// level next to each other.
	Numbers
	// Tasks is a checklist, whose items are Checked or not.
	Tasks
)

// normalized returns the style with its levels in range, as text pasted
// or loaded may have any.
func (s ParagraphStyle) normalized() ParagraphStyle {
	s.Heading = max(0, min(s.Heading, 6))
	s.Indent = max(0, min(s.Indent, maxIndent))
	if s.List > Tasks {
		s.List = NoList
	}
	if s.List != Tasks {
		s.Checked = false
	}
	return s
}

// ParagraphStyle is the style of a paragraph of an Text: what it
// is (a heading, an item of a list, a quote, a block of code) and how its
// lines align. Lists are flat, as in Quill and Automerge: an item's Indent
// is its level, and items next to each other make a list.
type ParagraphStyle struct {
	// Heading is the level of a heading, from 1, or 0 for a body
	// paragraph.
	Heading int      `json:"heading,omitempty"`
	List    ListKind `json:"list,omitempty"`
	// Checked marks a task done.
	Checked bool `json:"checked,omitempty"`
	// Indent is how many levels the paragraph is indented: the level of a
	// list's item, from 0.
	Indent int `json:"indent,omitempty"`
	// Align aligns its lines: Start (the default), Center or End.
	Align ui.Align `json:"align,omitempty"`
	// Quote and CodeBlock show the paragraph as a quotation, or as code in
	// the monospaced font, its spaces kept.
	Quote     bool `json:"quote,omitempty"`
	CodeBlock bool `json:"codeBlock,omitempty"`
}

// Run is a run of text with its attributes.
type Run struct {
	Text       string
	Attributes Attributes
}

// Paragraph is a paragraph of an Text: its style and its runs,
// without its newline.
type Paragraph struct {
	Style ParagraphStyle
	Runs  []Run
}

// richPara is a paragraph of an Text. Its runs hold its runes
// and, last, its end: its newline, or for the last paragraph the end of the
// text, whose attributes are those typing gets in an empty paragraph. They
// are as long as they can be, none empty: n+1 runes in all.
type richPara struct {
	id    uint64
	style ParagraphStyle
	n     int // runes, without the end
	runs  []attrRun
}

type attrRun struct {
	n int
	a Attributes
}

// paraIDs numbers the paragraphs of every text, so that a paragraph keeps
// its identity as the text changes around it, for layouts and, one day,
// collaboration.
var paraIDs atomic.Uint64

func newPara(style ParagraphStyle, runs []attrRun) richPara {
	p := richPara{id: paraIDs.Add(1), style: style.normalized(), runs: runs}
	for _, r := range runs {
		p.n += r.n
	}
	p.n--
	return p
}

// emptyParas is the paragraph of the empty text.
var emptyParas = seqLeaf([]richPara{{runs: []attrRun{{n: 1}}}})

func (t Text) seq() *seq[richPara] {
	if t.paras == nil {
		return emptyParas
	}
	return t.paras
}

// NewText returns s as an Text, without attributes.
func NewText(s string) Text {
	return TextOf(Paragraph{Runs: []Run{{Text: s}}})
}

// TextOf returns an Text of paragraphs. A newline in a
// run starts another paragraph of the same style.
//
//	ui.TextOf(
//		ui.Paragraph{Style: ui.ParagraphStyle{Heading: 1}, Runs: []ui.Run{{Text: "Groceries"}}},
//		ui.Paragraph{Style: ui.ParagraphStyle{List: ui.Tasks}, Runs: []ui.Run{{Text: "Milk"}}},
//	)
func TextOf(paragraphs ...Paragraph) Text {
	if len(paragraphs) == 0 {
		return Text{}
	}
	var lines []string
	var paras []richPara
	for _, p := range paragraphs {
		var line strings.Builder
		var runs []attrRun
		var last Attributes
		for _, r := range p.Runs {
			last = r.Attributes
			rest := r.Text
			for {
				part, after, more := strings.Cut(rest, "\n")
				line.WriteString(part)
				runs = appendRuns(runs, part, r.Attributes)
				if !more {
					break
				}
				lines = append(lines, line.String())
				paras = append(paras, newPara(p.Style, appendRun(runs, attrRun{1, r.Attributes.extending()})))
				line.Reset()
				runs = nil
				rest = after
			}
		}
		lines = append(lines, line.String())
		paras = append(paras, newPara(p.Style, appendRun(runs, attrRun{1, last.extending()})))
	}
	return Text{text: rope.Build(strings.Join(lines, "\n")), paras: seqBuild(paras)}
}

// appendRuns appends the runs of s with attributes a: an object
// replacement character is a run of its own, where a holds an attachment,
// which no other rune takes.
func appendRuns(runs []attrRun, s string, a Attributes) []attrRun {
	if s == "" {
		return runs
	}
	if a.Attachment == nil {
		return appendRun(runs, attrRun{utf8.RuneCountInString(s), a})
	}
	plain := a
	plain.Attachment = nil
	for _, r := range s {
		if r == ObjectReplacement {
			runs = appendRun(runs, attrRun{1, a})
		} else {
			runs = appendRun(runs, attrRun{1, plain})
		}
	}
	return runs
}

// appendRun appends r, joining it to the last run when they have the same
// attributes: runs holding an attachment are never joined.
func appendRun(runs []attrRun, r attrRun) []attrRun {
	if r.n <= 0 {
		return runs
	}
	if k := len(runs) - 1; k >= 0 && runs[k].a == r.a && r.a.Attachment == nil {
		runs[k].n += r.n
		return runs
	}
	return append(runs, r)
}

// cutRuns returns the runs from rune from up to rune to of runs, appended
// to dst.
func cutRuns(dst, runs []attrRun, from, to int) []attrRun {
	at := 0
	for _, r := range runs {
		a, z := max(at, from), min(at+r.n, to)
		if a < z {
			dst = appendRun(dst, attrRun{z - a, r.a})
		}
		at += r.n
		if at >= to {
			break
		}
	}
	return dst
}

// Len returns the number of runes of the text, newlines included.
func (t Text) Len() int { return rope.Sum(t.text).Runes }

// String returns the text without its attributes.
func (t Text) String() string { return rope.TextOf(t.text).String() }

// ParagraphCount returns the number of paragraphs: one more than the
// newlines.
func (t Text) ParagraphCount() int { return rope.Sum(t.text).Lines + 1 }

// Same reports whether t and u are the same version of a text, without
// comparing them: a text is the same as what it was copied from.
func (t Text) Same(u Text) bool { return t.text == u.text && t.paras == u.paras }

// Equal reports whether t and u hold the same text with the same
// attributes and paragraph styles.
func (t Text) Equal(u Text) bool {
	if t.Same(u) {
		return true
	}
	if t.Len() != u.Len() || t.ParagraphCount() != u.ParagraphCount() || t.String() != u.String() {
		return false
	}
	equal := true
	us := u.seq()
	seqEach(t.seq(), 0, func(i int, p *richPara) bool {
		q := seqAt(us, i)
		equal = p.style == q.style && slices.Equal(p.runs, q.runs)
		return equal
	})
	return equal
}

// paraAt returns the paragraph holding rune i, whose newline is its.
func (t Text) paraAt(i int) int {
	i = max(0, min(i, t.Len()))
	return rope.Prefix(t.text, rope.ByteOfRune(t.text, i)).Lines
}

// paraStart returns the rune paragraph p starts at.
func (t Text) paraStart(p int) int {
	if p <= 0 {
		return 0
	}
	return rope.Prefix(t.text, rope.LineStart(t.text, p)).Runes
}

// ParagraphRange returns where the paragraph holding rune i starts and
// ends, before its newline.
func (t Text) ParagraphRange(i int) (start, end int) {
	p := t.paraAt(i)
	start = t.paraStart(p)
	return start, start + seqAt(t.seq(), p).n
}

// AttributesAt returns the attributes of rune i, or those of the end of
// the text for its length.
func (t Text) AttributesAt(i int) Attributes {
	p := t.paraAt(i)
	return seqAt(t.seq(), p).attributesAt(i - t.paraStart(p))
}

// attributesAt returns the attributes of rune i of the paragraph, its end
// for its length.
func (p *richPara) attributesAt(i int) Attributes {
	at := 0
	for _, r := range p.runs {
		at += r.n
		if i < at {
			return r.a
		}
	}
	return p.runs[len(p.runs)-1].a
}

// ParagraphStyleAt returns the style of the paragraph holding rune i.
func (t Text) ParagraphStyleAt(i int) ParagraphStyle {
	return seqAt(t.seq(), t.paraAt(i)).style
}

// Paragraphs returns the paragraphs of the text, with the rune each starts
// at.
func (t Text) Paragraphs() iter.Seq2[int, Paragraph] {
	return func(yield func(int, Paragraph) bool) {
		start := 0
		tx := rope.TextOf(t.text)
		seqEach(t.seq(), 0, func(i int, p *richPara) bool {
			line := tx.Line(i)
			par := Paragraph{Style: p.style}
			off := 0
			for _, r := range p.runs {
				if off >= len(line) {
					break
				}
				end := off
				for k := 0; k < r.n && end < len(line); k++ {
					_, size := utf8.DecodeRuneInString(line[end:])
					end += size
				}
				par.Runs = append(par.Runs, Run{Text: line[off:end], Attributes: r.a})
				off = end
			}
			if !yield(start, par) {
				return false
			}
			start += p.n + 1
			return true
		})
	}
}

// Runs returns the runs of the text from rune start to rune end, with the
// rune each starts at: the newline ending a paragraph is a run of its own,
// or ends the paragraph's last run when it has the same attributes.
func (t Text) Runs(start, end int) iter.Seq2[int, Run] {
	return func(yield func(int, Run) bool) {
		n := t.Len()
		start, end = max(0, min(start, n)), max(0, min(end, n))
		if start >= end {
			return
		}
		tx := rope.TextOf(t.text)
		pa := t.paraAt(start)
		at := t.paraStart(pa)
		off := rope.ByteOfRune(t.text, at)
		seqEach(t.seq(), pa, func(_ int, p *richPara) bool {
			for _, r := range p.runs {
				a, z := max(at, start), min(at+r.n, end, n)
				if a < z {
					ba := off
					if a > at {
						ba = rope.ByteOfRune(t.text, a)
					}
					bz := rope.ByteOfRune(t.text, z)
					if !yield(a, Run{Text: tx.Slice(ba, bz), Attributes: r.a}) {
						return false
					}
				}
				at += r.n
				if at >= end {
					return false
				}
				off = rope.ByteOfRune(t.text, min(at, n))
			}
			return true
		})
	}
}

// Slice returns the runes from start to end with their attributes and the
// styles of their paragraphs.
func (t Text) Slice(start, end int) Text {
	n := t.Len()
	start, end = max(0, min(start, n)), max(0, min(end, n))
	if start >= end {
		return Text{}
	}
	pa, pz := t.paraAt(start), t.paraAt(end)
	sa, sz := t.paraStart(pa), t.paraStart(pz)
	paras := make([]richPara, 0, pz-pa+1)
	seqEach(t.seq(), pa, func(i int, p *richPara) bool {
		from, to := 0, p.n+1
		if i == pa {
			from = start - sa
		}
		last := i == pz
		if last {
			to = end - sz
		}
		runs := cutRuns(nil, p.runs, from, to)
		n := to - from - 1
		if last {
			runs = appendRun(runs, attrRun{1, p.runs[len(p.runs)-1].a})
			n++
		}
		paras = append(paras, richPara{id: p.id, style: p.style, n: n, runs: runs})
		return !last
	})
	_, rest := rope.Split(t.text, rope.ByteOfRune(t.text, start))
	mid, _ := rope.Split(rest, rope.ByteOfRune(rest, end-start))
	return Text{text: mid, paras: seqBuild(paras)}
}

// Replace returns the text with the runes from start to end replaced by
// with. Paragraphs joined or split take the style of the one their first
// rune comes from, as AppKit fixes paragraph styles: deleting the newline
// after a heading keeps the heading, and text put at the start of a
// paragraph brings its own paragraph's style.
func (t Text) Replace(start, end int, with Text) Text {
	n := t.Len()
	start, end = max(0, min(start, end, n)), max(0, min(max(start, end), n))
	pa, pz := t.paraAt(start), t.paraAt(end)
	sa, sz := t.paraStart(pa), t.paraStart(pz)
	s := t.seq()
	first, last := seqAt(s, pa), seqAt(s, pz)
	head := cutRuns(nil, first.runs, 0, start-sa)
	tail := cutRuns(nil, last.runs, end-sz, last.n+1)
	headN, tailN := start-sa, last.n-(end-sz)

	var frag []richPara
	seqEach(with.seq(), 0, func(_ int, p *richPara) bool {
		frag = append(frag, *p)
		return true
	})
	out := make([]richPara, 0, len(frag))
	if len(frag) == 1 {
		f := frag[0]
		runs := cutRuns(head, f.runs, 0, f.n)
		runs = cutRuns(runs, tail, 0, tailN+1)
		style := last.style
		switch {
		case headN > 0:
			style = first.style
		case f.n > 0:
			style = f.style
		}
		out = append(out, richPara{id: first.id, style: style, n: headN + f.n + tailN, runs: runs})
	} else {
		f := frag[0]
		style := f.style
		if headN > 0 {
			style = first.style
		}
		out = append(out, richPara{id: first.id, style: style, n: headN + f.n, runs: cutRuns(head, f.runs, 0, f.n+1)})
		for _, f := range frag[1 : len(frag)-1] {
			f.id = paraIDs.Add(1)
			out = append(out, f)
		}
		f = frag[len(frag)-1]
		style, id := last.style, paraIDs.Add(1)
		if f.n > 0 {
			style = f.style
		}
		if pz != pa {
			id = last.id
		}
		runs := cutRuns(nil, f.runs, 0, f.n)
		runs = cutRuns(runs, tail, 0, tailN+1)
		out = append(out, richPara{id: id, style: style, n: f.n + tailN, runs: runs})
	}
	left, rest := rope.Split(t.text, rope.ByteOfRune(t.text, start))
	_, right := rope.Split(rest, rope.ByteOfRune(rest, end-start))
	return Text{
		text:  rope.Join(rope.Join(left, with.text), right),
		paras: seqReplace(s, pa, pz+1, out),
	}
}

// ReplaceString returns the text with the runes from start to end replaced
// by s, which takes attributes a and, for the paragraphs it adds, the style
// of the paragraph holding start.
func (t Text) ReplaceString(start, end int, s string, a Attributes) Text {
	style := t.ParagraphStyleAt(start)
	var frag []richPara
	for rest := s; ; {
		line, after, more := strings.Cut(rest, "\n")
		runs := appendRuns(nil, line, a)
		frag = append(frag, richPara{style: style, n: utf8.RuneCountInString(line), runs: appendRun(runs, attrRun{1, a.extending()})})
		if !more {
			break
		}
		rest = after
	}
	return t.Replace(start, end, Text{text: rope.Build(s), paras: seqBuild(frag)})
}

// Insert returns the text with with inserted at rune i.
func (t Text) Insert(i int, with Text) Text { return t.Replace(i, i, with) }

// Delete returns the text without the runes from start to end.
func (t Text) Delete(start, end int) Text {
	return t.Replace(start, end, Text{})
}

// Append returns the text followed by s with attributes a.
func (t Text) Append(s string, a Attributes) Text {
	n := t.Len()
	return t.ReplaceString(n, n, s, a)
}

// Format returns the text with fn changing the attributes of the runes
// from start to end, newlines included. fn cannot add or take away an
// attachment, which only an object replacement character has.
func (t Text) Format(start, end int, fn func(a *Attributes)) Text {
	n := t.Len()
	start, end = max(0, min(start, n)), max(0, min(end, n))
	if start >= end {
		return t
	}
	pa, pz := t.paraAt(start), t.paraAt(end)
	if end == t.paraStart(pz) && pz > pa && end < n {
		pz-- // up to the newline ending the paragraph before
	}
	at := t.paraStart(pa)
	out := make([]richPara, 0, pz-pa+1)
	changed := false
	seqEach(t.seq(), pa, func(i int, p *richPara) bool {
		from, to := max(start-at, 0), min(end-at, p.n+1)
		if end == n && i == pz {
			// To the end of the text: what typing gets there too.
			to = p.n + 1
		}
		runs := cutRuns(nil, p.runs, 0, from)
		mid := cutRuns(nil, p.runs, from, to)
		for _, r := range mid {
			b := r.a
			fn(&b)
			b.Attachment = r.a.Attachment
			changed = changed || b != r.a
			runs = appendRun(runs, attrRun{r.n, b})
		}
		runs = cutRuns(runs, p.runs, to, p.n+1)
		q := *p
		q.runs = runs
		out = append(out, q)
		at += p.n + 1
		return i < pz
	})
	if !changed {
		return t
	}
	return Text{text: t.text, paras: seqReplace(t.seq(), pa, pz+1, out)}
}

// FormatParagraphs returns the text with fn changing the styles of the
// paragraphs from the one holding start to the one holding end, but for a
// paragraph end only starts, after a selection of whole paragraphs.
func (t Text) FormatParagraphs(start, end int, fn func(p *ParagraphStyle)) Text {
	pa, pz := t.paragraphSpan(start, end)
	out := make([]richPara, 0, pz-pa+1)
	changed := false
	seqEach(t.seq(), pa, func(i int, p *richPara) bool {
		q := *p
		fn(&q.style)
		q.style = q.style.normalized()
		changed = changed || q.style != p.style
		out = append(out, q)
		return i < pz
	})
	if !changed {
		return t
	}
	return Text{text: t.text, paras: seqReplace(t.seq(), pa, pz+1, out)}
}

// paragraphSpan returns the paragraphs a selection from start to end
// formats: those it touches, but for the one after a selection ending at
// its start.
func (t Text) paragraphSpan(start, end int) (int, int) {
	n := t.Len()
	start, end = max(0, min(start, end, n)), max(0, min(max(start, end), n))
	pa, pz := t.paraAt(start), t.paraAt(end)
	if end > start && pz > pa && end == t.paraStart(pz) {
		pz--
	}
	return pa, pz
}
