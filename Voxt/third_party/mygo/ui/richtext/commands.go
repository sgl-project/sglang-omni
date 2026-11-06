package richtext

import (
	"strings"

	"github.com/egoist/mygo/ui"
)

// FormatState is whether a format holds for a selection: all of it, none,
// or some of it, as a toolbar's button shows it.
type FormatState uint8

const (
	FormatOff FormatState = iota
	FormatOn
	FormatMixed
)

// Formatting describes the formats of a document's selection, as
// a toolbar and a Format menu show them: of the runes selected, or of the
// typing attributes at a caret.
type Formatting struct {
	// Attributes are those of the first rune selected, or the typing
	// attributes at a caret.
	Attributes                                   Attributes
	Bold, Italic, Underline, Strikethrough, Code FormatState
	// Link is the link all of the selection is in, "" when none is.
	Link string
	// Paragraph is the style of the first paragraph selected, and
	// ParagraphsMixed whether others differ from it.
	Paragraph       ParagraphStyle
	ParagraphsMixed bool
}

// Formatting returns the formats of the selection.
func (d *Document) Formatting() Formatting {
	d.lock()
	defer d.mu.Unlock()
	return d.formatting()
}

func (d *Document) formatting() Formatting {
	a, z := d.span()
	f := Formatting{Attributes: d.typingAttributes()}
	if a < z {
		// From the first rune that takes a format, not a newline.
		for _, r := range d.text.Runs(a, z) {
			if r.Text != "\n" {
				f.Attributes = r.Attributes
				f.Attributes.Attachment = nil
				break
			}
		}
	}
	state := func(on bool) FormatState {
		if on {
			return FormatOn
		}
		return FormatOff
	}
	at := f.Attributes
	f.Bold, f.Italic, f.Underline, f.Strikethrough, f.Code = state(at.Bold()), state(at.Italic), state(at.Underline), state(at.Strikethrough), state(at.Code)
	f.Link = at.Link
	if a < z {
		mix := func(s *FormatState, on bool) {
			if (*s == FormatOn) != on {
				*s = FormatMixed
			}
		}
		for _, r := range d.text.Runs(a, z) {
			if r.Text == "\n" {
				continue // a newline selected takes no format
			}
			x := r.Attributes
			mix(&f.Bold, x.Bold())
			mix(&f.Italic, x.Italic)
			mix(&f.Underline, x.Underline)
			mix(&f.Strikethrough, x.Strikethrough)
			mix(&f.Code, x.Code)
			if x.Link != f.Link {
				f.Link = ""
			}
		}
	}
	pa, pz := d.text.paragraphSpan(a, z)
	seqEach(d.text.seq(), pa, func(i int, p *richPara) bool {
		if i == pa {
			f.Paragraph = p.style
		} else if p.style != f.Paragraph {
			f.ParagraphsMixed = true
		}
		return i < pz
	})
	return f
}

// Format changes the attributes of the selection with fn, as one step of
// undo, or the typing attributes at a caret.
func (d *Document) Format(fn func(a *Attributes)) {
	d.lock()
	defer d.mu.Unlock()
	d.format(fn)
}

func (d *Document) format(fn func(a *Attributes)) {
	a, z := d.span()
	if a == z {
		t := d.typingAttributes()
		fn(&t)
		t.Attachment = nil
		d.typing = &t
		return
	}
	anchor, caret := d.anchor, d.caret
	d.edit(false, func(e *Edit) { e.Format(a, z, fn) })
	d.anchor, d.caret = anchor, caret
}

// toggle turns a format on for the selection unless all of it has it, then
// off.
func (d *Document) toggle(state func(f Formatting) FormatState, set func(a *Attributes, on bool)) {
	d.lock()
	defer d.mu.Unlock()
	on := state(d.formatting()) != FormatOn
	d.format(func(a *Attributes) { set(a, on) })
}

// ToggleBold makes the selection bold, or not when all of it is, as ⌘B
// does.
func (d *Document) ToggleBold() {
	d.toggle(func(f Formatting) FormatState { return f.Bold }, func(a *Attributes, on bool) {
		if on {
			a.Weight = 700
		} else {
			a.Weight = 0
		}
	})
}

// ToggleItalic, ToggleUnderline, ToggleStrikethrough and ToggleCode turn
// those formats on for the selection, or off when all of it has them.
func (d *Document) ToggleItalic() {
	d.toggle(func(f Formatting) FormatState { return f.Italic }, func(a *Attributes, on bool) { a.Italic = on })
}

func (d *Document) ToggleUnderline() {
	d.toggle(func(f Formatting) FormatState { return f.Underline }, func(a *Attributes, on bool) { a.Underline = on })
}

func (d *Document) ToggleStrikethrough() {
	d.toggle(func(f Formatting) FormatState { return f.Strikethrough }, func(a *Attributes, on bool) { a.Strikethrough = on })
}

func (d *Document) ToggleCode() {
	d.toggle(func(f Formatting) FormatState { return f.Code }, func(a *Attributes, on bool) { a.Code = on })
}

// linkRange returns the runes around i that link to the same URL as the
// rune at i, or i, i.
func (t Text) linkRange(i int) (int, int) {
	url := t.AttributesAt(i).Link
	if url == "" || i >= t.Len() {
		return i, i
	}
	ps, pe := t.ParagraphRange(i)
	a, z := i, i
	for a > ps && t.AttributesAt(a-1).Link == url {
		a--
	}
	for z < pe && t.AttributesAt(z).Link == url {
		z++
	}
	return a, z
}

// SetLink links the selection to url, or takes its link away for "". At a
// caret inside a link, it changes or takes away that link; elsewhere it
// inserts url, linked to itself.
func (d *Document) SetLink(url string) {
	d.lock()
	defer d.mu.Unlock()
	a, z := d.span()
	if a == z {
		la, lz := d.text.linkRange(a)
		if la == lz && a > 0 {
			la, lz = d.text.linkRange(a - 1)
			if lz != a {
				la, lz = a, a
			}
		}
		if la == lz {
			if url != "" {
				attrs := d.typingAttributes()
				attrs.Link = url
				d.edit(false, func(e *Edit) { e.ReplaceString(a, a, url, attrs) })
			}
			return
		}
		a, z = la, lz
	}
	anchor, caret := d.anchor, d.caret
	d.edit(false, func(e *Edit) { e.Format(a, z, func(x *Attributes) { x.Link = url }) })
	d.anchor, d.caret = anchor, caret
}

// FormatParagraphs changes the styles of the paragraphs selected with fn,
// as one step of undo.
func (d *Document) FormatParagraphs(fn func(p *ParagraphStyle)) {
	d.lock()
	defer d.mu.Unlock()
	d.formatParagraphs(fn)
}

func (d *Document) formatParagraphs(fn func(p *ParagraphStyle)) {
	a, z := d.span()
	anchor, caret, typing := d.anchor, d.caret, d.typing
	d.edit(false, func(e *Edit) { e.FormatParagraphs(a, z, fn) })
	d.anchor, d.caret, d.typing = anchor, caret, typing
}

// paragraphsAll reports whether every paragraph selected satisfies fn.
func (d *Document) paragraphsAll(fn func(p ParagraphStyle) bool) bool {
	a, z := d.span()
	pa, pz := d.text.paragraphSpan(a, z)
	all := true
	seqEach(d.text.seq(), pa, func(i int, p *richPara) bool {
		all = fn(p.style)
		return all && i < pz
	})
	return all
}

// SetHeading makes the paragraphs selected headings of level, from 1, or
// body paragraphs for 0.
func (d *Document) SetHeading(level int) {
	d.FormatParagraphs(func(p *ParagraphStyle) {
		p.Heading = max(0, min(level, 6))
		if p.Heading > 0 {
			p.List, p.Checked, p.CodeBlock = NoList, false, false
		}
	})
}

// ToggleList makes the paragraphs selected items of a list of kind, or
// takes them out of it when all of them are.
func (d *Document) ToggleList(kind ListKind) {
	d.lock()
	defer d.mu.Unlock()
	off := d.paragraphsAll(func(p ParagraphStyle) bool { return p.List == kind })
	d.formatParagraphs(func(p *ParagraphStyle) {
		if off {
			p.List, p.Checked = NoList, false
			return
		}
		p.List, p.Heading, p.CodeBlock = kind, 0, false
		if kind != Tasks {
			p.Checked = false
		}
	})
}

// ToggleQuote and ToggleCodeBlock make the paragraphs selected a quotation
// or a block of code, or not when all of them are.
func (d *Document) ToggleQuote() {
	d.lock()
	defer d.mu.Unlock()
	off := d.paragraphsAll(func(p ParagraphStyle) bool { return p.Quote })
	d.formatParagraphs(func(p *ParagraphStyle) { p.Quote = !off })
}

func (d *Document) ToggleCodeBlock() {
	d.lock()
	defer d.mu.Unlock()
	off := d.paragraphsAll(func(p ParagraphStyle) bool { return p.CodeBlock })
	d.formatParagraphs(func(p *ParagraphStyle) {
		p.CodeBlock = !off
		if p.CodeBlock {
			p.Heading, p.List, p.Checked = 0, NoList, false
		}
	})
}

// maxIndent is the deepest a paragraph indents.
const maxIndent = 8

// Indent indents the paragraphs selected by levels, or outdents them for
// a negative count, as Tab and Shift-Tab do in a list.
func (d *Document) Indent(levels int) {
	d.FormatParagraphs(func(p *ParagraphStyle) { p.Indent = max(0, min(p.Indent+levels, maxIndent)) })
}

// SetAlign aligns the lines of the paragraphs selected.
func (d *Document) SetAlign(a ui.Align) {
	d.FormatParagraphs(func(p *ParagraphStyle) { p.Align = a })
}

// ToggleChecked checks the task holding rune i, or unchecks it.
func (d *Document) ToggleChecked(i int) {
	d.lock()
	defer d.mu.Unlock()
	anchor, caret, typing := d.anchor, d.caret, d.typing
	d.edit(false, func(e *Edit) {
		e.FormatParagraphs(i, i, func(p *ParagraphStyle) {
			if p.List == Tasks {
				p.Checked = !p.Checked
			}
		})
	})
	d.anchor, d.caret, d.typing = anchor, caret, typing
}

// InsertText replaces the selection with s, in the typing attributes, as
// typing does but as a step of undo of its own.
func (d *Document) InsertText(s string) {
	d.lock()
	defer d.mu.Unlock()
	a, z := d.span()
	attrs := d.typingAttributes()
	d.edit(false, func(e *Edit) {
		n := e.Text().Len()
		e.ReplaceString(a, z, s, attrs)
		end := z + e.Text().Len() - n
		e.Select(end, end)
	})
}

// Insert replaces the selection with text, as a paste does.
func (d *Document) Insert(text Text) {
	d.lock()
	defer d.mu.Unlock()
	a, z := d.span()
	d.edit(false, func(e *Edit) {
		e.Replace(a, z, text)
		e.Select(a+text.Len(), a+text.Len())
	})
}

// InsertAttachment replaces the selection with an attachment.
func (d *Document) InsertAttachment(a *Attachment) {
	d.lock()
	defer d.mu.Unlock()
	s, z := d.span()
	attrs := d.typingAttributes().extending()
	attrs.Attachment = a
	d.edit(false, func(e *Edit) {
		e.ReplaceString(s, z, string(ObjectReplacement), attrs)
		e.Select(s+1, s+1)
	})
}

// normalizeNewlines turns the newlines of other systems into "\n".
func normalizeNewlines(s string) string {
	if !strings.ContainsAny(s, "\r\u2029") {
		return s
	}
	return strings.NewReplacer("\r\n", "\n", "\r", "\n", " ", "\n").Replace(s)
}
