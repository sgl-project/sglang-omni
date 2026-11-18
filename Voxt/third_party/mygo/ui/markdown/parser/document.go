package parser

import (
	"iter"
	"slices"
	"strings"
	"unsafe"
)

// Extensions are the syntax a Document reads beyond CommonMark.
type Extensions uint32

const (
	// ExtTables are GitHub's tables, of cells between pipes.
	ExtTables Extensions = 1 << iota
	// ExtTaskLists are list items starting with [ ] or [x].
	ExtTaskLists
	// ExtStrikethrough is text between one or two tildes.
	ExtStrikethrough
	// ExtAutolinks are links without angle brackets: www.example.com,
	// https://example.com and someone@example.com.
	ExtAutolinks
	// ExtFootnotes are references as [^1], defined by [^1]: lines.
	ExtFootnotes
	// ExtMath is TeX between $$ and $$, \[ and \], or \( and \), in blocks
	// of their own or within text.
	ExtMath
	// ExtDollarMath is TeX between single dollar signs, as $x^2$, except
	// where the dollars read as amounts of money: not after an opening
	// dollar followed by a space or a digit, nor before a closing one
	// following a space or followed by a digit. It needs ExtMath.
	ExtDollarMath

	// GFM is GitHub Flavored Markdown.
	GFM = ExtTables | ExtTaskLists | ExtStrikethrough | ExtAutolinks
	// Default is GitHub Flavored Markdown with footnotes and math, as
	// language models write it.
	Default = GFM | ExtFootnotes | ExtMath
)

// Kind is what a block is.
type Kind uint8

// Kinds of blocks.
const (
	// Root holds the top-level blocks: block 0 of a document.
	Root Kind = iota
	Paragraph
	// Heading has a Level from 1 to 6.
	Heading
	ThematicBreak
	// CodeBlock is fenced or indented code, whose Info string names the
	// language of a fence.
	CodeBlock
	// HTMLBlock is raw HTML, whose lines are its content.
	HTMLBlock
	// Quote is a block quote, holding blocks.
	Quote
	// List holds items, Ordered or not, Tight or not.
	List
	// Item is an item of a list, holding blocks, with a Task state.
	Item
	// Table holds rows, the first its Header, whose cells align as Align
	// says.
	Table
	TableRow
	// TableCell holds inline content.
	TableCell
	// FootnoteDefinition holds the blocks of a footnote, its Label the
	// footnote's.
	FootnoteDefinition
	// MathBlock is display math, whose lines are its TeX.
	MathBlock
)

var kindNames = [...]string{"root", "paragraph", "heading", "thematic break", "code block", "HTML block", "quote", "list", "item", "table", "table row", "table cell", "footnote definition", "math block"}

func (k Kind) String() string {
	if int(k) < len(kindNames) {
		return kindNames[k]
	}
	return "unknown"
}

// Task is the check box of a task list item.
type Task uint8

const (
	NoTask Task = iota
	Unchecked
	Checked
)

// Align is how a table's column aligns its cells.
type Align uint8

const (
	AlignNone Align = iota
	AlignLeft
	AlignCenter
	AlignRight
)

// Block is a block of a document: blocks are in document order, a block's
// children right after it, up to its Next.
type Block struct {
	Kind Kind
	// Parent is the index of the block holding it, -1 for the root.
	Parent int
	// Next is the index after the block and its descendants.
	Next int
	// Start and End are the offsets of the block's bytes in the source,
	// from its first character to the end of its last line.
	Start, End int
	// Open reports that the block is still open at the end of the
	// source, so that more text can continue it: a fence without its
	// closing fence, or the paragraph being written.
	Open bool

	// Level is a heading's, from 1 to 6.
	Level int
	// Ordered, Number (an ordered list's first), Marker (*, - or + for
	// a bullet list, . or ) after an ordered list's numbers) and Tight
	// (no blank lines between its items and their blocks) describe a
	// list.
	Ordered bool
	Number  int
	Marker  byte
	Tight   bool
	// Task is an item's check box.
	Task Task
	// Fenced tells a fenced code block from an indented one.
	Fenced bool
	// Header tells a table's header row.
	Header bool
	// Align is the alignment of a table's columns.
	Align []Align

	// info is a fenced code block's info string or a footnote's label.
	info string

	// What the parser and the inline parser keep.
	parse blockState
	// segA and segB are the block's content lines in the document's
	// segs, and segMark how many lines it held as the block began.
	segA, segB, segMark int32
	text                string
	textS               int32 // the source offset of text[0] when text is a slice of the source, else -1
	inl                 []Inline
	inlOK               bool
	// labels are those its inlines looked up as references, to parse
	// them anew when a definition of one comes or goes.
	labels []string
	rev    uint32
	// healed tells inlines parsed completed, as the end of a streaming
	// document's.
	healed bool
}

// seg is a line of a leaf block's content: pad spaces (from a tab partly
// consumed by indentation), the source's bytes [start, end), and a
// newline unless cell.
type seg struct {
	start, end int32
	pad        int8
	cell       bool // a table cell: no newline, and \| is |
}

// Info returns a fenced code block's info string, unescaped, its
// language first (see Language).
func (b *Block) Info() string {
	if b.Kind != CodeBlock {
		return ""
	}
	return b.info
}

// Label returns a footnote definition's label.
func (b *Block) Label() string {
	if b.Kind != FootnoteDefinition {
		return ""
	}
	return b.info
}

// Language returns the language of a fenced code block, the first word
// of its info string.
func (b *Block) Language() string {
	s := b.Info()
	if i := strings.IndexAny(s, " \t"); i >= 0 {
		s = s[:i]
	}
	return s
}

// Revision changes when the block's inlines are parsed anew, as when a
// link reference defined later in the document resolves its links: views
// keep what they made of a block while it stays the same.
func (b *Block) Revision() uint32 { return b.rev }

// Healed reports whether the block's inlines were completed as the tail
// of a streaming document: their closers are not in the source, and
// change as it grows.
func (b *Block) Healed() bool { return b.healed }

// Document is Markdown parsed, which takes text appended to it, or edits,
// and parses only what they change.
type Document struct {
	src string
	// buf holds the source as text is appended: its bytes before the
	// source's length never change, so that the source is a string over
	// them, and appending copies only the text appended.
	buf    []byte
	ext    Extensions
	blocks []Block
	segs   []seg
	refs   map[string]linkRef
	// defs are the link reference definitions in order, duplicates
	// included, which take over as edits remove those before them.
	defs  []definition
	notes map[string]int // footnote label → definition block
	// restart is the offset of the line where parsing restarts as text
	// is appended: the start of the outermost block open after the last
	// complete line, or that line's end; restartAt is the index of the
	// first block parsed from there.
	restart   int
	restartAt int
	// streaming completes unclosed syntax at the end of the source (see
	// SetStreaming), and tail is the block holding the end.
	streaming bool
	tail      int
	// cut is where the source parsed for display ends: text held back
	// while streaming lies past it.
	cut int
	// oldInl keeps the inlines of blocks parsed again as text is
	// appended, to reuse.
	oldInl []oldInline
	// healedAt is the block whose inlines were parsed healed, or -1.
	healedAt int
	// revs counts the inlines parsed, to give each a Revision of its own.
	revs uint32
	inp  inlineParser
	// steps counts the work parsing does beyond reading the source once:
	// the bytes scanned ahead or back, and the delimiters, brackets, nodes
	// and lines walked, as the loops that could take quadratic time do it.
	// Tests check that it grows linearly with the source.
	steps int
	// marks is the buffer of the last sourceMap.
	marks []int
}

type oldInline struct {
	at     int // the offset of the block's content
	rev    uint32
	text   string
	segs   []seg // its lines, as offsets move with them
	inl    []Inline
	labels []string
	healed bool
}

// definition is a link reference definition of a label.
type definition struct {
	key string
	linkRef
}

// define adds a definition, which defines its label unless one before
// it did.
func (d *Document) define(df definition) {
	d.defs = append(d.defs, df)
	if _, ok := d.refs[df.key]; !ok {
		d.refs[df.key] = df.linkRef
	}
}

// linkRef is a link reference definition.
type linkRef struct {
	dest, title string
	at          int // the offset of the definition
}

// Parse parses src with the Default extensions.
func Parse(src string) *Document {
	d := New(Default)
	d.Set(src)
	return d
}

// New returns an empty document reading ext.
func New(ext Extensions) *Document {
	d := &Document{ext: ext, healedAt: -1}
	d.inp.d = d
	d.reset()
	return d
}

func (d *Document) reset() {
	d.blocks = append(d.blocks[:0], Block{Kind: Root, Parent: -1, Next: 1, textS: -1})
	d.segs = d.segs[:0]
	d.blocks[0].parse.open = true
	d.blocks[0].parse.lastChild = -1
	d.refs = map[string]linkRef{}
	d.defs = d.defs[:0]
	d.notes = map[string]int{}
	d.restart, d.restartAt = 0, 1
}

// Source returns the text of the document.
func (d *Document) Source() string { return d.src }

// Footnote returns the FootnoteDefinition block of a label, as a
// FootnoteRef's Text holds it.
func (d *Document) Footnote(label string) (int, bool) {
	i, ok := d.notes[normalizeLabel(label)]
	return i, ok
}

// Extensions returns what the document reads beyond CommonMark.
func (d *Document) Extensions() Extensions { return d.ext }

// Len returns the number of blocks, the root's included.
func (d *Document) Len() int { return len(d.blocks) }

// Block returns block i, 0 being the root. The block stays valid until
// the document changes.
func (d *Document) Block(i int) *Block { return &d.blocks[i] }

// Children returns the indices of the children of block i.
func (d *Document) Children(i int) iter.Seq[int] {
	return func(yield func(int) bool) {
		end := d.blocks[i].Next
		for j := i + 1; j < end; j = d.blocks[j].Next {
			if !yield(j) {
				return
			}
		}
	}
}

// Streaming reports whether the document completes unclosed syntax at
// its end (see SetStreaming).
func (d *Document) Streaming() bool { return d.streaming }

// SetStreaming sets whether the document is text still being written, as
// a language model's reply: its end then shows what it will most likely
// become rather than the half-written syntax it is. The last paragraph
// closes its emphasis, code spans, strikethrough and math, a link whose
// destination is being written is a link (Inline.Pending), a tag being
// written is left out, and lines that could still turn into something
// else are held back: a line of only a few markers, as - or ==, which
// would make the paragraph above a heading, and a table's header until
// its delimiter row comes. Finished blocks are never changed. Turn it off
// once the text is complete, to parse its end exactly as CommonMark does.
func (d *Document) SetStreaming(on bool) {
	if d.streaming == on {
		return
	}
	d.streaming = on
	d.parseTail()
}

// Set makes src the text of the document, parsing only what changed
// since the last: text appended to it, or else the blocks around the
// change (see Edit).
func (d *Document) Set(src string) {
	old := d.src
	switch {
	case src == old:
		return
	case len(src) > len(old) && src[:len(old)] == old:
		// The new source is the string given, copied nowhere.
		d.reserve(src[len(old):])
		d.src = src
		d.parseTail()
		return
	}
	// The common prefix and suffix bound the change.
	p := 0
	for p < len(old) && p < len(src) && old[p] == src[p] {
		p++
	}
	s := 0
	for s < len(old)-p && s < len(src)-p && old[len(old)-1-s] == src[len(src)-1-s] {
		s++
	}
	d.Edit(p, len(old)-s, src[p:len(src)-s])
}

// Append adds text to the end of the document, parsing again only the
// outermost block still open before it, or nothing but the new lines
// when no block is.
func (d *Document) Append(text string) {
	if text == "" {
		return
	}
	d.reserve(text)
	if len(d.buf) != len(d.src) || len(d.src) > 0 && unsafe.StringData(d.src) != unsafe.SliceData(d.buf) {
		d.buf = append(make([]byte, 0, max(2*len(d.src)+len(text), 512)), d.src...)
	}
	if cap(d.buf)-len(d.buf) < len(text) {
		d.buf = append(make([]byte, 0, 2*cap(d.buf)+len(text)), d.buf...)
	}
	d.buf = append(d.buf, text...)
	d.src = unsafe.String(unsafe.SliceData(d.buf), len(d.buf))
	d.parseTail()
}

// reserve makes room for the blocks of text, a large part of a document
// parsed at once.
func (d *Document) reserve(text string) {
	if len(text) < 4096 {
		return
	}
	n := strings.Count(text, "\n")
	d.blocks = slices.Grow(d.blocks, n+n/4+8)
}

// Edit replaces the source's bytes [from, to) with text. Parsing restarts
// at the top-level block before the change and goes on until a top-level
// block starts where one started before it, from which the blocks parsed
// before are kept.
func (d *Document) Edit(from, to int, text string) {
	if from == to && text == "" {
		return
	}
	if from == len(d.src) && to == from {
		d.Append(text)
		return
	}
	d.edit(from, to, text)
}

// String returns the source of the document.
func (d *Document) String() string { return d.src }
