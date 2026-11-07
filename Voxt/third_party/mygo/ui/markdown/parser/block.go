package parser

import (
	"slices"
	"strings"
	"unicode/utf8"
)

// The block parser follows cmark's (blocks.c), line by line with a stack
// of open blocks, over flat blocks with byte offsets: a block is open
// while lines may still go into it, and only the last child of an open
// block can be open. Blocks go in document order as they are added, as
// only open blocks take children.

// blockState is what the parser keeps of a block while it is open.
type blockState struct {
	// firstLine is the offset of the line the block began on, where
	// parsing it anew starts, though definitions may move its Start; and
	// joined, for a table made of a paragraph's last line, is that
	// paragraph, which parsing it anew starts with.
	firstLine int32
	joined    int32
	lastChild int32
	prev      int32
	startLine int32
	// A list's and an item's.
	markerOffset int32
	padding      int32
	// A fence's, and a table's columns.
	fenceLength int32
	fenceOffset int32
	columns     int32

	open          bool
	lastLineBlank bool
	lineChecked   bool
	// quiet tells, for a top-level block, that no block was open as the
	// line it starts on began: parsing from that line gives it anew.
	quiet     bool
	fenceChar byte
	htmlType  int8
	// tableTried notes that a paragraph was no table's header, as cmark
	// tries a paragraph once.
	tableTried bool
	// mathBracket tells a math block ending with \] from one ending with
	// $$, and mathDone one that ended.
	mathBracket bool
	mathDone    bool
}

func (s *blockState) mathClose() string {
	if s.mathBracket {
		return `\]`
	}
	return "$$"
}

const tabStop = 4
const codeIndent = 4

// maxDepth bounds how deep lists and quotes nest, as cmark does, so that
// a hostile input cannot make the parser recurse without end.
const maxDepth = 100

type blockParser struct {
	d   *Document
	src string
	ext Extensions

	// The line: ln is its text without its line ending, from lineStart.
	ln        string
	lineStart int
	lineNum   int
	prevEnd   int // where the line before ended, its line ending excluded

	offset, column           int
	firstNonspace            int
	firstNonspaceColumn      int
	indent                   int
	blank                    bool
	partiallyConsumedTab     bool
	thematicBreakKillPos     int
	current                  int
	lastMatchedContainer     int
	lineQuiet                bool
	resync                   func(line int) bool
	stopped                  bool
	recordRestart            bool
	completeEnd, restartNext int
}

// peek returns the byte of the line at i, '\n' past its end.
func (p *blockParser) peek(i int) byte {
	if i < len(p.ln) {
		return p.ln[i]
	}
	return '\n'
}

func isLineEnd(c byte) bool     { return c == '\n' || c == '\r' }
func isSpaceOrTab(c byte) bool  { return c == ' ' || c == '\t' }
func isASCIIDigit(c byte) bool  { return c >= '0' && c <= '9' }
func isASCIILetter(c byte) bool { return c|0x20 >= 'a' && c|0x20 <= 'z' }

// parseTail parses the source again from the restart point.
func (d *Document) parseTail() {
	at, off := d.restartAt, d.restart
	removed := d.truncate(at, off)
	d.parseShown(nil)
	d.refsChanged(removed, at, off)
	d.unheal()
}

// unheal parses anew the inlines of a block parsed healed that is no
// longer the end of a streaming document.
func (d *Document) unheal() {
	h := d.healedAt
	if h < 0 || h >= len(d.blocks) || h == d.tail && d.streaming {
		return
	}
	if b := &d.blocks[h]; b.healed {
		b.inlOK, b.inl, b.labels, b.healed = false, nil, nil, false
		b.rev++
	}
	d.healedAt = -1
}

// parseShown parses from the restart point to the end of what is shown.
func (d *Document) parseShown(resync func(line int) bool) bool {
	cut := d.displayCut()
	if d.parse(d.restart, cut, resync) {
		return true
	}
	if d.streaming && d.ext&ExtTables != 0 {
		// A table's header waits for its delimiter row.
		if c := d.heldTableRow(cut); c < cut {
			d.truncate(d.restartAt, d.restart)
			d.parse(d.restart, c, nil)
			cut = c
		}
	}
	d.cut = cut
	return false
}

// edit replaces the source's [from, to) with text, and parses again from
// the top-level block before the one holding from, until a top-level
// block starts, with no block open, where one started after the change,
// from which the blocks parsed before are kept.
func (d *Document) edit(from, to int, text string) {
	old := d.src
	delta := len(text) - (to - from)
	prev := 0
	for j, k := 1, 0; j < len(d.blocks); j = d.blocks[j].Next {
		if int(d.blocks[j].parse.firstLine) > from {
			break
		}
		prev, k = k, j
	}
	at, off := 1, 0
	if prev > 0 {
		at, off = d.restartOf(prev)
	}
	tail := append([]Block(nil), d.blocks[at:]...)
	oldSegs := append([]seg(nil), d.segs...)
	oldDefs := append([]definition(nil), d.defs...)
	oldRefs := make(map[string]linkRef, len(d.refs))
	for key, r := range d.refs {
		oldRefs[key] = r
	}
	oldRestart, oldRestartAt, oldTail, oldCut := d.restart, d.restartAt, d.tail, d.cut
	removed := d.truncate(at, off)
	d.src = old[:from] + text + old[to:]
	d.restart, d.restartAt = off, at
	editEnd := from + len(text)
	j, met := 0, -1
	resync := func(line int) bool {
		if line < editEnd {
			return false
		}
		for j < len(tail) && (tail[j].Parent != 0 || int(tail[j].parse.firstLine)+delta < line) {
			j++
		}
		if j < len(tail) && tail[j].parse.quiet && tail[j].parse.joined == 0 && int(tail[j].parse.firstLine) >= to && int(tail[j].parse.firstLine)+delta == line {
			met = j
			return true
		}
		return false
	}
	if d.parseShown(resync) {
		splitAt := int(tail[met].parse.firstLine)
		shift := len(d.blocks) - (at + met)
		for m := met; m < len(tail); m++ {
			b := tail[m]
			if b.Parent > 0 {
				b.Parent += shift
			}
			b.Next += shift
			b.Start += delta
			b.End += delta
			b.parse.firstLine += int32(delta)
			if b.parse.joined > 0 {
				b.parse.joined += int32(shift)
			}
			segs := oldSegs[b.segA:b.segB]
			b.segMark = int32(len(d.segs))
			b.segA = b.segMark
			for _, sg := range segs {
				sg.start += int32(delta)
				sg.end += int32(delta)
				d.segs = append(d.segs, sg)
			}
			b.segB = int32(len(d.segs))
			if b.textS >= 0 {
				b.textS += int32(delta)
			}
			if b.inlOK && len(b.inl) > 0 {
				b.inl = append([]Inline(nil), b.inl...)
				for x := range b.inl {
					b.inl[x].Start += delta
					b.inl[x].End += delta
				}
			}
			b.parse.lastChild, b.parse.prev = -1, -1
			i := len(d.blocks)
			d.blocks = append(d.blocks, b)
			if b.Parent == 0 {
				d.blocks[0].parse.lastChild = int32(i)
			}
			if b.Kind == FootnoteDefinition {
				key := normalizeLabel(b.info)
				if _, ok := d.notes[key]; !ok {
					d.notes[key] = i
				}
			}
		}
		d.blocks[0].Next = len(d.blocks)
		// The definitions of the blocks kept, duplicates included, which
		// take over from those gone.
		for _, df := range oldDefs {
			if df.at >= splitAt {
				df.at += delta
				d.define(df)
			}
		}
		if oldRestart >= splitAt {
			d.restart, d.restartAt = oldRestart+delta, oldRestartAt+shift
		}
		if oldTail >= at+met {
			d.tail = oldTail + shift
		}
		d.cut = oldCut + delta
	}
	d.refsChanged(removed, at, off)
	d.unheal()
}

// truncate drops the blocks from index at, parsed from offset off, and
// the definitions made there, which it returns, keeping their inlines
// to reuse.
func (d *Document) truncate(at, off int) map[string]linkRef {
	clear(d.oldInl)
	d.oldInl = d.oldInl[:0]
	for i := at; i < len(d.blocks); i++ {
		b := &d.blocks[i]
		if b.inlOK && b.segB > b.segA {
			d.oldInl = append(d.oldInl, oldInline{int(d.segs[b.segA].start), b.rev, b.text, slices.Clone(d.segsOf(b)), b.inl, b.labels, b.healed})
		}
	}
	root := &d.blocks[0]
	root.Next = at
	if at < len(d.blocks) {
		root.parse.lastChild = d.blocks[at].parse.prev
		d.segs = d.segs[:d.blocks[at].segMark]
	}
	clear(d.blocks[at:])
	d.blocks = d.blocks[:at]
	k := 0
	for _, df := range d.defs {
		if df.at < off {
			d.defs[k] = df
			k++
		}
	}
	d.defs = d.defs[:k]
	var removed map[string]linkRef
	for k, r := range d.refs {
		if r.at >= off {
			if removed == nil {
				removed = map[string]linkRef{}
			}
			removed[k] = r
			delete(d.refs, k)
		}
	}
	for k, i := range d.notes {
		if i >= at {
			if removed == nil {
				removed = map[string]linkRef{}
			}
			removed["^"+k] = linkRef{at: -1}
			delete(d.notes, k)
		}
	}
	return removed
}

// refsChanged parses anew the inlines of blocks that looked up a
// definition made or unmade by parsing from block at, offset off, which
// removed held before.
func (d *Document) refsChanged(removed map[string]linkRef, at, off int) {
	var changed []string
	for k, r := range removed {
		if strings.HasPrefix(k, "^") {
			if _, ok := d.notes[k[1:]]; !ok {
				changed = append(changed, k)
			}
		} else if n, ok := d.refs[k]; !ok || n.dest != r.dest || n.title != r.title {
			changed = append(changed, k)
		}
	}
	for k, r := range d.refs {
		if r.at >= off {
			if o, ok := removed[k]; !ok || o.dest != r.dest || o.title != r.title {
				changed = append(changed, k)
			}
		}
	}
	for k, i := range d.notes {
		if i >= at {
			if _, ok := removed["^"+k]; !ok {
				changed = append(changed, "^"+k)
			}
		}
	}
	if len(changed) == 0 {
		return
	}
	for k := range d.oldInl {
		for _, l := range d.oldInl[k].labels {
			if containsString(changed, l) {
				d.oldInl[k].text = "" // matches no block
				break
			}
		}
	}
	for i := range d.blocks {
		b := &d.blocks[i]
		if !b.inlOK || len(b.labels) == 0 {
			continue
		}
		for _, l := range b.labels {
			if containsString(changed, l) {
				b.inlOK, b.inl, b.labels = false, nil, nil
				b.rev++
				break
			}
		}
	}
}

func containsString(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

// lineStartOf returns the offset of the start of the line holding off.
func lineStartOf(src string, off int) int {
	for off > 0 && src[off-1] != '\n' && src[off-1] != '\r' {
		off--
	}
	return off
}

// displayCut returns where the source shown ends: while streaming, a
// rune half received, and a last line that is only a few markers, as -
// or ==, wait for what follows them.
func (d *Document) displayCut() int {
	n := len(d.src)
	if !d.streaming {
		return n
	}
	// A rune half received.
	for i := 1; i <= 3 && i <= n; i++ {
		c := d.src[n-i]
		if c < utf8.RuneSelf {
			break
		}
		if utf8.RuneStart(c) {
			if !utf8.FullRuneInString(d.src[n-i:]) {
				n -= i
			}
			break
		}
	}
	ls := lineStartOf(d.src, n)
	line := strings.Trim(d.src[ls:n], " \t")
	if line != "" && len(line) <= 3 && strings.Trim(line, "-=*+#>|_~`:.)0123456789") == "" {
		return ls
	}
	return n
}

// heldTableRow returns where the source shown ends when the paragraph at
// its end ends in lines starting with a pipe, which may be a table's
// header whose delimiter row is coming: before those lines.
func (d *Document) heldTableRow(cut int) int {
	t := d.tail
	if t <= 0 || t >= len(d.blocks) || d.blocks[t].Kind != Paragraph {
		return cut
	}
	b := &d.blocks[t]
	c := cut
	segs := d.segsOf(b)
	for i := len(segs) - 1; i >= 0; i-- {
		s := segs[i]
		line := d.src[s.start:s.end]
		if !strings.HasPrefix(strings.TrimLeft(line, " \t"), "|") {
			break
		}
		c = lineStartOf(d.src, int(s.start))
	}
	return c
}

// parse parses the source from off, a line's start, to end, with no
// block open but the root, adding blocks after those there are.
func (d *Document) parse(off, end int, resync func(line int) bool) bool {
	p := &blockParser{d: d, src: d.src[:end], ext: d.ext, current: 0, resync: resync}
	d.inp.reset()
	d.tail = -1
	d.restart, d.restartAt = off, len(d.blocks)
	src := p.src
	pos := off
	// The next \n and \r, found once each, so that lines ending in one
	// of them alone take linear time.
	nl, cr := -1, -1
	for pos < len(src) {
		// The line ends at the first \n, \r or \r\n.
		if nl < pos {
			if nl = strings.IndexByte(src[pos:], '\n'); nl < 0 {
				nl = len(src)
			} else {
				nl += pos
			}
			d.steps += nl - pos
		}
		if cr < pos {
			if cr = strings.IndexByte(src[pos:], '\r'); cr < 0 {
				cr = len(src)
			} else {
				cr += pos
			}
			d.steps += cr - pos
		}
		lineEnd, next := nl, nl+1
		if cr < nl {
			lineEnd = cr
			next = lineEnd + 1
			if next < len(src) && src[next] == '\n' {
				next++
			}
		}
		if next > len(src) {
			next = len(src)
		}
		// A line ending in \r at the very end may yet be \r\n.
		complete := next > lineEnd && !(next == len(d.src) && d.src[next-1] == '\r')
		if p.resync != nil && p.quiet() && p.resync(pos) {
			for p.current > 0 {
				p.current = p.finalize(p.current)
			}
			return true
		}
		p.processLine(pos, lineEnd)
		pos = next
		if complete {
			p.noteRestart(next)
		}
	}
	// What is still open at the end is open: finalize it for showing.
	last := p.current
	for p.current > 0 {
		b := &d.blocks[p.current]
		b.Open = !(b.Kind == MathBlock && b.parse.mathDone)
		p.current = p.finalize(p.current)
	}
	if last > 0 && last < len(d.blocks) {
		switch d.blocks[last].Kind {
		case Paragraph, Heading:
			d.tail = last
		case Table, TableRow:
			if n := len(d.blocks) - 1; d.blocks[n].Kind == TableCell {
				d.tail = n
			}
		}
	}
	d.blocks[0].Next = len(d.blocks)
	return false
}

// noteRestart notes, after a complete line ending at next, where parsing
// restarts as text is appended.
func (p *blockParser) noteRestart(next int) {
	d := p.d
	top := d.blocks[0].parse.lastChild
	if top > 0 && d.blocks[top].parse.open {
		d.restartAt, d.restart = d.restartOf(int(top))
	} else {
		d.restart = next
		d.restartAt = len(d.blocks)
	}
}

// restartOf returns the block parsing anew starts with to give top-level
// block i again, and the offset of its first line: a table made of a
// paragraph's last line starts with the paragraph.
func (d *Document) restartOf(i int) (int, int) {
	if j := d.blocks[i].parse.joined; j > 0 {
		i = int(j)
	}
	return i, int(d.blocks[i].parse.firstLine)
}

// quiet reports whether no block is open that a line could go into: none,
// or a heading, which no line continues.
func (p *blockParser) quiet() bool {
	top := p.d.blocks[0].parse.lastChild
	if top <= 0 || !p.d.blocks[top].parse.open {
		return true
	}
	b := &p.d.blocks[top]
	return b.Kind == Heading && b.parse.lastChild < 0
}

func (p *blockParser) processLine(start, end int) {
	p.ln = p.src[start:end]
	p.lineStart = start
	p.offset, p.column = 0, 0
	p.firstNonspace, p.firstNonspaceColumn = 0, 0
	p.thematicBreakKillPos = 0
	p.indent = 0
	p.blank = false
	p.partiallyConsumedTab = false
	if start == 0 && strings.HasPrefix(p.ln, "\xef\xbb\xbf") {
		p.offset = 3
	}
	p.lineNum++
	p.lineQuiet = p.quiet()

	allMatched := false
	container, ok := p.checkOpenBlocks(&allMatched)
	if ok {
		p.lastMatchedContainer = container
		p.openNewBlocks(&container, allMatched)
		p.addTextToContainer(container)
	}
	p.prevEnd = end
}

func (p *blockParser) block(i int) *Block { return &p.d.blocks[i] }

// addChild adds a block of kind as the last child of parent, closing
// the blocks that cannot hold it.
func (p *blockParser) addChild(parent int, kind Kind, start int) int {
	// The blocks the line did not match close first, as a block starting
	// leaves no lazy continuation: blocks stay in document order.
	for p.current != parent && p.d.blocks[p.current].parse.open && p.isBelow(p.current, parent) {
		p.current = p.finalize(p.current)
	}
	for !canContain(p.block(parent).Kind, kind) {
		parent = p.finalize(parent)
	}
	d := p.d
	i := len(d.blocks)
	n := int32(len(d.segs))
	d.blocks = append(d.blocks, Block{Kind: kind, Parent: parent, Start: start, End: start, Next: i + 1, textS: -1, segA: n, segB: n, segMark: n})
	b := &d.blocks[i]
	b.parse.open = true
	b.parse.lastChild = -1
	b.parse.startLine = int32(p.lineNum)
	b.parse.firstLine = int32(p.lineStart)
	pb := &d.blocks[parent]
	b.parse.prev = pb.parse.lastChild
	pb.parse.lastChild = int32(i)
	if parent == 0 {
		b.parse.quiet = p.lineQuiet && lineStartOf(d.src, start) == p.lineStart
	}
	return i
}

// isBelow reports whether block c is a descendant of block parent.
func (p *blockParser) isBelow(c, parent int) bool {
	for c >= 0 {
		p.d.steps++
		c = p.d.blocks[c].Parent
		if c == parent {
			return true
		}
	}
	return false
}

func canContain(parent, child Kind) bool {
	switch parent {
	case Root, Quote, Item, FootnoteDefinition:
		return child != Item && child != TableRow && child != TableCell
	case List:
		return child == Item
	case Table:
		return child == TableRow
	case TableRow:
		return child == TableCell
	}
	return false
}

func acceptsLines(k Kind) bool {
	return k == Paragraph || k == Heading || k == CodeBlock || k == MathBlock
}

// removeLast removes block i, the last, an empty paragraph.
func (p *blockParser) removeLast(i int) {
	d := p.d
	b := &d.blocks[i]
	d.blocks[b.Parent].parse.lastChild = b.parse.prev
	d.blocks = d.blocks[:i]
}

// finalize closes block i and returns its parent. The line it ends on
// is the one before the line being parsed, unless it is closed by that
// line, as a fence by its closing fence, or by the end.
func (p *blockParser) finalize(i int) int {
	d := p.d
	b := &d.blocks[i]
	parent := b.Parent
	b.parse.open = false
	b.Next = len(d.blocks)
	switch {
	case b.segB > b.segA:
		b.End = int(d.segs[b.segB-1].end)
	case b.parse.lastChild >= 0:
		b.End = max(b.End, d.blocks[b.parse.lastChild].End)
	}
	switch b.Kind {
	case Paragraph:
		if !p.resolveDefinitions(i) {
			p.removeLast(i)
			return parent
		}
	case CodeBlock:
		if !b.Fenced {
			// Blank lines at the end are not the code's.
			for b.segB > b.segA && strings.Trim(d.src[d.segs[b.segB-1].start:d.segs[b.segB-1].end], " \t") == "" {
				b.segB--
			}
			if b.segB > b.segA {
				b.End = int(d.segs[b.segB-1].end)
			}
		}
	case HTMLBlock:
		// cmark keeps blank lines at the end of HTML blocks of types 1
		// to 5; they render nothing.
	case List:
		b.Tight = true
		for item := p.firstChild(i); item >= 0 && b.Tight; item = p.nextSibling(item) {
			if d.blocks[item].parse.lastLineBlank && p.nextSibling(item) >= 0 {
				b.Tight = false
				break
			}
			for sub := p.firstChild(item); sub >= 0; sub = p.nextSibling(sub) {
				if (p.nextSibling(item) >= 0 || p.nextSibling(sub) >= 0) && p.endsWithBlankLine(sub) {
					b.Tight = false
					break
				}
			}
		}
	case Item:
		if b.parse.lastChild < 0 {
			b.End = b.Start + 1
		}
	}
	if parent >= 0 {
		pb := &d.blocks[parent]
		pb.End = max(pb.End, b.End)
	}
	return parent
}

func (p *blockParser) firstChild(i int) int {
	if p.d.blocks[i].parse.lastChild < 0 {
		return -1
	}
	return i + 1
}

func (p *blockParser) nextSibling(i int) int {
	b := &p.d.blocks[i]
	n := b.Next
	if n < len(p.d.blocks) && p.d.blocks[n].Parent == b.Parent {
		return n
	}
	return -1
}

// endsWithBlankLine reports whether a block ends with a blank line,
// looking into the last items of lists.
func (p *blockParser) endsWithBlankLine(i int) bool {
	for {
		b := &p.d.blocks[i]
		if b.parse.lineChecked {
			return b.parse.lastLineBlank
		}
		b.parse.lineChecked = true
		if (b.Kind == List || b.Kind == Item) && b.parse.lastChild >= 0 {
			i = int(b.parse.lastChild)
			continue
		}
		return b.parse.lastLineBlank
	}
}

// resolveDefinitions takes the link reference definitions off the start
// of paragraph i, and reports whether text remains.
func (p *blockParser) resolveDefinitions(i int) bool {
	d := p.d
	b := &d.blocks[i]
	if b.segB == b.segA {
		return false
	}
	if first := d.segs[b.segA]; d.src[first.start] != '[' || first.pad != 0 {
		return true
	}
	text, _ := d.contentOf(i)
	m := d.sourceMap(i)
	pos := 0
	for pos < len(text) && text[pos] == '[' {
		d.steps++
		n := d.inp.parseReference(d, text[pos:], m.source(pos))
		if n == 0 {
			break
		}
		pos += n
	}
	if pos == 0 {
		return true
	}
	// Drop the lines the definitions took.
	consumed := pos
	for b.segA < b.segB {
		d.steps++
		s := d.segs[b.segA]
		l := int(s.pad) + int(s.end-s.start) + 1
		if consumed < l {
			break
		}
		consumed -= l
		b.segA++
	}
	if b.segA < b.segB && consumed > 0 {
		s := &d.segs[b.segA]
		take := consumed - int(s.pad)
		s.pad = 0
		s.start += int32(max(take, 0))
	}
	b.text, b.textS = "", -1
	if !hasText(d, d.segsOf(b)) {
		b.segA = b.segB
		return false
	}
	b.Start = int(d.segs[b.segA].start)
	return true
}

func hasText(d *Document, segs []seg) bool {
	for _, s := range segs {
		if strings.Trim(d.src[s.start:s.end], " \t") != "" {
			return true
		}
	}
	return false
}

// segsOf returns the content lines of b.
func (d *Document) segsOf(b *Block) []seg { return d.segs[b.segA:b.segB] }

// appendSeg adds a content line to block i, moving its lines to the end
// of the document's first if others came after them.
func (d *Document) appendSeg(i int, s seg) {
	b := &d.blocks[i]
	if int(b.segB) != len(d.segs) {
		n := b.segB - b.segA
		d.segs = append(d.segs, d.segs[b.segA:b.segB]...)
		b.segB = int32(len(d.segs))
		b.segA = b.segB - n
	}
	d.segs = append(d.segs, s)
	b.segB++
}

// addLine adds the rest of the line to the content of block i.
func (p *blockParser) addLine(i int) {
	s := seg{start: int32(p.lineStart + p.offset), end: int32(p.lineStart + len(p.ln))}
	if p.partiallyConsumedTab {
		p.offset++
		s.start++
		s.pad = int8(tabStop - p.column%tabStop)
	}
	if s.start > s.end {
		s.start = s.end
	}
	p.d.appendSeg(i, s)
	p.d.blocks[i].End = int(s.end)
}

func (p *blockParser) findFirstNonspace() {
	charsToTab := tabStop - p.column%tabStop
	if p.firstNonspace <= p.offset {
		p.firstNonspace = p.offset
		p.firstNonspaceColumn = p.column
		for {
			c := p.peek(p.firstNonspace)
			if c == ' ' {
				p.firstNonspace++
				p.firstNonspaceColumn++
				charsToTab--
				if charsToTab == 0 {
					charsToTab = tabStop
				}
			} else if c == '\t' {
				p.firstNonspace++
				p.firstNonspaceColumn += charsToTab
				charsToTab = tabStop
			} else {
				break
			}
		}
	}
	p.indent = p.firstNonspaceColumn - p.column
	p.blank = isLineEnd(p.peek(p.firstNonspace))
}

// advanceOffset advances by count columns (with columns) or bytes. A tab
// partly consumed sets partiallyConsumedTab.
func (p *blockParser) advanceOffset(count int, columns bool) {
	for count > 0 && p.offset < len(p.ln) {
		c := p.ln[p.offset]
		if c == '\t' {
			charsToTab := tabStop - p.column%tabStop
			if columns {
				p.partiallyConsumedTab = charsToTab > count
				adv := min(count, charsToTab)
				p.column += adv
				if !p.partiallyConsumedTab {
					p.offset++
				}
				count -= adv
			} else {
				p.partiallyConsumedTab = false
				p.column += charsToTab
				p.offset++
				count--
			}
		} else {
			p.partiallyConsumedTab = false
			p.offset++
			p.column++
			count--
		}
	}
}

// checkOpenBlocks matches the line against the open blocks' prefixes,
// returning the last that matched, or false when the line is done (a
// closing fence).
func (p *blockParser) checkOpenBlocks(allMatched *bool) (int, bool) {
	d := p.d
	container := 0
	for {
		d.steps++
		lc := d.blocks[container].parse.lastChild
		if lc < 0 || !d.blocks[lc].parse.open {
			break
		}
		container = int(lc)
		b := &d.blocks[container]
		p.findFirstNonspace()
		matched := true
		switch b.Kind {
		case Quote:
			matched = p.blockQuotePrefix()
		case Item:
			if w := int(b.parse.markerOffset + b.parse.padding); p.indent >= w {
				p.advanceOffset(w, true)
			} else if p.blank && b.parse.lastChild >= 0 {
				p.advanceOffset(p.firstNonspace-p.offset, false)
			} else {
				matched = false
			}
		case FootnoteDefinition:
			if p.indent >= 4 {
				p.advanceOffset(4, true)
			} else if !p.blank {
				matched = false
			}
		case CodeBlock:
			if !b.Fenced {
				if p.indent >= codeIndent {
					p.advanceOffset(codeIndent, true)
				} else if p.blank {
					p.advanceOffset(p.firstNonspace-p.offset, false)
				} else {
					matched = false
				}
			} else {
				n := 0
				if p.indent <= 3 && p.peek(p.firstNonspace) == b.parse.fenceChar {
					n = scanCloseFence(p.ln, p.firstNonspace)
				}
				if n >= int(b.parse.fenceLength) {
					// The closing fence ends the block and the line.
					p.advanceOffset(n, false)
					b.End = p.lineStart + len(p.ln)
					p.current = p.finalize(container)
					d.blocks[container].End = p.lineStart + len(p.ln)
					return 0, false
				}
				for k := b.parse.fenceOffset; k > 0 && isSpaceOrTab(p.peek(p.offset)); k-- {
					p.advanceOffset(1, true)
				}
			}
		case MathBlock:
			// Lines go into it until its closing line; one that ended is
			// closed by the next line, as a thematic break.
		case Heading:
			matched = false
		case HTMLBlock:
			if b.parse.htmlType >= 6 {
				matched = !p.blank
			}
		case Paragraph:
			matched = !p.blank
		case Table:
			matched = !p.blank && rowHasCells(p.ln[p.firstNonspace:])
		}
		if !matched {
			return b.Parent, true
		}
	}
	*allMatched = true
	return container, true
}

func (p *blockParser) blockQuotePrefix() bool {
	if p.indent <= 3 && p.peek(p.firstNonspace) == '>' {
		p.advanceOffset(p.indent+1, true)
		if isSpaceOrTab(p.peek(p.offset)) {
			p.advanceOffset(1, true)
		}
		return true
	}
	return false
}

// openNewBlocks opens the blocks the line starts inside container.
func (p *blockParser) openNewBlocks(container *int, allMatched bool) {
	d := p.d
	maybeLazy := d.blocks[p.current].Kind == Paragraph
	depth := 0
	for p.takesBlocks(*container) {
		depth++
		p.findFirstNonspace()
		indented := p.indent >= codeIndent
		c := p.peek(p.firstNonspace)
		cont := *container
		contKind := d.blocks[cont].Kind
		var n int
		switch {
		case !indented && c == '>' && depth < maxDepth:
			start := p.lineStart + p.firstNonspace
			p.advanceOffset(p.firstNonspace+1-p.offset, false)
			if isSpaceOrTab(p.peek(p.offset)) {
				p.advanceOffset(1, true)
			}
			*container = p.addChild(cont, Quote, start)

		case !indented && c == '#' && func() bool { n = scanATXStart(p.ln, p.firstNonspace); return n > 0 }():
			start := p.lineStart + p.firstNonspace
			level := 0
			for p.peek(p.firstNonspace+level) == '#' {
				level++
			}
			p.advanceOffset(p.firstNonspace+n-p.offset, false)
			*container = p.addChild(cont, Heading, start)
			d.blocks[*container].Level = level

		case !indented && (c == '`' || c == '~') && func() bool { n = scanOpenFence(p.ln, p.firstNonspace); return n > 0 }():
			start := p.lineStart + p.firstNonspace
			*container = p.addChild(cont, CodeBlock, start)
			b := &d.blocks[*container]
			b.Fenced = true
			b.parse.fenceChar = c
			b.parse.fenceLength = int32(n)
			b.parse.fenceOffset = int32(p.firstNonspace - p.offset)
			b.info = unescapeString(strings.Trim(p.ln[p.firstNonspace+n:], " \t"))
			b.End = p.lineStart + len(p.ln)
			p.advanceOffset(len(p.ln)-p.offset, false)

		case !indented && c == '<' && func() bool {
			n = scanHTMLBlockStart(p.ln, p.firstNonspace)
			if n == 0 && contKind != Paragraph && !(!allMatched && d.blocks[p.current].Kind == Paragraph) {
				n = scanHTMLBlockStart7(p.ln, p.firstNonspace)
			}
			return n > 0
		}():
			*container = p.addChild(cont, HTMLBlock, p.lineStart+p.firstNonspace)
			d.blocks[*container].parse.htmlType = int8(n)

		case !indented && contKind == Paragraph && (c == '=' || c == '-') && func() bool { n = scanSetextLine(p.ln, p.firstNonspace); return n > 0 }():
			if p.resolveDefinitionsOpen(cont) {
				b := &d.blocks[cont]
				b.Kind = Heading
				b.Level = n
				b.End = p.lineStart + len(p.ln)
				p.advanceOffset(len(p.ln)-p.offset, false)
			}

		case !indented && !(contKind == Paragraph && !allMatched) && p.thematicBreakKillPos <= p.firstNonspace && func() bool { n = p.scanThematicBreak(p.firstNonspace); return n > 0 }():
			*container = p.addChild(cont, ThematicBreak, p.lineStart+p.firstNonspace)
			d.blocks[*container].End = p.lineStart + len(p.ln)
			p.advanceOffset(len(p.ln)-p.offset, false)

		case !indented && p.ext&ExtMath != 0 && (c == '$' || c == '\\') && p.mathStart(container):

		case !indented && p.ext&ExtFootnotes != 0 && c == '[' && depth < maxDepth && func() bool { n = scanFootnoteDef(p.ln, p.firstNonspace); return n > 0 }():
			label := p.ln[p.firstNonspace+2:]
			label = label[:strings.IndexByte(label, ']')]
			start := p.lineStart + p.firstNonspace
			p.advanceOffset(p.firstNonspace+n-p.offset, false)
			*container = p.addChild(cont, FootnoteDefinition, start)
			d.blocks[*container].info = label
			if key := normalizeLabel(label); d.notes[key] == 0 {
				d.notes[key] = *container
			}

		case (!indented || contKind == List) && p.indent < 4 && depth < maxDepth && p.listItemStart(container):

		case indented && !maybeLazy && !p.blank:
			p.advanceOffset(codeIndent, true)
			*container = p.addChild(cont, CodeBlock, p.lineStart+p.offset)

		case !indented && p.ext&ExtTables != 0 && contKind == Paragraph && p.tableStart(container):

		case !indented && p.ext&ExtTables != 0 && contKind == Table && !p.blank:
			p.tableRow(cont)
			*container = cont

		default:
			return
		}
		if b := &d.blocks[*container]; acceptsLines(b.Kind) && !b.parse.mathDone || b.Kind == Table {
			return
		}
		maybeLazy = false
	}
}

// takesBlocks reports whether new blocks may start in block i: not in
// code, HTML or math still taking lines.
func (p *blockParser) takesBlocks(i int) bool {
	b := &p.d.blocks[i]
	switch b.Kind {
	case CodeBlock, HTMLBlock:
		return false
	case MathBlock:
		return b.parse.mathDone
	}
	return true
}

// resolveDefinitionsOpen takes the definitions off the start of the open
// paragraph i before it becomes a setext heading, and reports whether
// text remains.
func (p *blockParser) resolveDefinitionsOpen(i int) bool {
	b := &p.d.blocks[i]
	if b.segB == b.segA {
		return false
	}
	if p.d.src[p.d.segs[b.segA].start] != '[' {
		return true
	}
	return p.resolveDefinitions(i)
}

// listItemStart opens a list item, and its list unless the container is
// a list it continues.
func (p *blockParser) listItemStart(container *int) bool {
	d := p.d
	cont := *container
	interrupts := d.blocks[cont].Kind == Paragraph
	pos := p.firstNonspace
	c := p.peek(pos)
	var ordered bool
	var marker byte
	number := 0
	switch {
	case c == '*' || c == '-' || c == '+':
		pos++
		if !isSpaceOrTab(p.peek(pos)) && !isLineEnd(p.peek(pos)) {
			return false
		}
		if interrupts {
			i := pos
			for isSpaceOrTab(p.peek(i)) {
				i++
			}
			if isLineEnd(p.peek(i)) {
				return false
			}
		}
		marker = c
	case isASCIIDigit(c):
		digits := 0
		for digits < 9 && isASCIIDigit(p.peek(pos)) {
			number = number*10 + int(p.peek(pos)-'0')
			pos++
			digits++
		}
		if interrupts && number != 1 {
			return false
		}
		c = p.peek(pos)
		if c != '.' && c != ')' {
			return false
		}
		pos++
		if !isSpaceOrTab(p.peek(pos)) && !isLineEnd(p.peek(pos)) {
			return false
		}
		if interrupts {
			i := pos
			for isSpaceOrTab(p.peek(i)) {
				i++
			}
			if isLineEnd(p.peek(i)) {
				return false
			}
		}
		ordered, marker = true, c
	default:
		return false
	}
	matched := pos - p.firstNonspace
	start := p.lineStart + p.firstNonspace
	markerOffset := p.indent
	p.advanceOffset(p.firstNonspace+matched-p.offset, false)
	savedTab, savedOffset, savedColumn := p.partiallyConsumedTab, p.offset, p.column
	for p.column-savedColumn <= 5 && isSpaceOrTab(p.peek(p.offset)) {
		p.advanceOffset(1, true)
	}
	i := p.column - savedColumn
	padding := matched + i
	if i >= 5 || i < 1 || isLineEnd(p.peek(p.offset)) {
		padding = matched + 1
		p.offset, p.column, p.partiallyConsumedTab = savedOffset, savedColumn, savedTab
		if i > 0 {
			p.advanceOffset(1, true)
		}
	}
	cb := &d.blocks[cont]
	if cb.Kind != List || cb.Ordered != ordered || cb.Marker != marker {
		cont = p.addChild(cont, List, start)
		l := &d.blocks[cont]
		l.Ordered, l.Marker, l.Number = ordered, marker, number
		l.parse.markerOffset, l.parse.padding = int32(markerOffset), int32(padding)
	}
	item := p.addChild(cont, Item, start)
	it := &d.blocks[item]
	it.Ordered, it.Marker, it.Number = ordered, marker, number
	it.parse.markerOffset, it.parse.padding = int32(markerOffset), int32(padding)
	it.End = p.lineStart + p.offset
	if p.ext&ExtTaskLists != 0 {
		p.taskMarker(item)
	}
	*container = item
	return true
}

// taskMarker reads [ ] or [x] at the start of an item's first line,
// followed by a space or a tab, as its task state.
func (p *blockParser) taskMarker(item int) {
	i := p.offset
	if p.partiallyConsumedTab {
		return
	}
	if p.peek(i) != '[' || p.peek(i+2) != ']' {
		return
	}
	c := p.peek(i + 1)
	if c != ' ' && c != 'x' && c != 'X' || !isSpaceOrTab(p.peek(i+3)) {
		return
	}
	// Text must follow, as GitHub asks.
	j := i + 3
	for isSpaceOrTab(p.peek(j)) {
		j++
	}
	if isLineEnd(p.peek(j)) {
		return
	}
	b := &p.d.blocks[item]
	if c == ' ' {
		b.Task = Unchecked
	} else {
		b.Task = Checked
	}
	p.advanceOffset(4, true)
}

// mathStart opens a block of display math at a line starting with $$ or
// \[, unless its closing delimiter comes before the end of the line with
// text after it, when it is math within a paragraph.
func (p *blockParser) mathStart(container *int) bool {
	rest := p.ln[p.firstNonspace:]
	var open, close string
	switch {
	case strings.HasPrefix(rest, "$$"):
		open, close = "$$", "$$"
	case strings.HasPrefix(rest, `\[`):
		open, close = `\[`, `\]`
	default:
		return false
	}
	body := rest[len(open):]
	done := false
	if j := strings.Index(body, close); j >= 0 {
		if strings.Trim(body[j+len(close):], " \t") != "" {
			return false
		}
		done = true
		body = body[:j]
	}
	start := p.lineStart + p.firstNonspace
	m := p.addChild(*container, MathBlock, start)
	b := &p.d.blocks[m]
	b.parse.mathBracket = close == `\]`
	b.parse.mathDone = done
	bs := start + len(open)
	for body != "" && (body[0] == ' ' || body[0] == '\t') {
		body, bs = body[1:], bs+1
	}
	body = strings.TrimRight(body, " \t")
	if body != "" {
		p.d.appendSeg(m, seg{start: int32(bs), end: int32(bs + len(body))})
		b = &p.d.blocks[m]
	}
	b.End = p.lineStart + len(p.ln)
	p.advanceOffset(len(p.ln)-p.offset, false)
	*container = m
	return true
}

// tableStart turns the last line of the paragraph container into a
// table's header when the line is a delimiter row of as many cells.
func (p *blockParser) tableStart(container *int) bool {
	d := p.d
	cont := *container
	b := &d.blocks[cont]
	if b.parse.tableTried || b.segB == b.segA {
		return false
	}
	line := p.ln[p.firstNonspace:]
	aligns, ok := scanDelimiterRow(line)
	if !ok {
		return false
	}
	h := d.segs[b.segB-1]
	header := d.src[h.start:h.end]
	if h.pad != 0 {
		return false
	}
	cells := splitRow(header)
	if len(cells) != len(aligns) {
		b.parse.tableTried = true
		return false
	}
	parent := b.Parent
	var table int
	if b.segB-b.segA > 1 {
		b.segB--
		first := b.parse.firstLine
		p.current = p.finalize(cont)
		kept := len(d.blocks) > cont
		table = p.addChild(parent, Table, int(h.start))
		// Its header was the paragraph's last line: parsing anew starts
		// with the paragraph.
		t := &d.blocks[table]
		t.parse.firstLine, t.parse.quiet = first, false
		if kept {
			t.parse.joined = int32(cont)
		}
	} else {
		table = cont
		b.Kind = Table
		b.segA = b.segB
		b.text, b.textS = "", -1
	}
	t := &d.blocks[table]
	t.Align = aligns
	t.parse.columns = int32(len(aligns))
	row := p.addChild(table, TableRow, int(h.start))
	d.blocks[row].Header = true
	p.addCells(row, int(h.start), header, cells, len(aligns))
	p.finalize(row)
	d.blocks[table].End = p.lineStart + len(p.ln)
	p.advanceOffset(len(p.ln)-p.offset, false)
	p.current = table
	p.lastMatchedContainer = table
	*container = table
	return true
}

// tableRow adds the line as a row of table t.
func (p *blockParser) tableRow(t int) {
	start := p.lineStart + p.firstNonspace
	line := p.ln[p.firstNonspace:]
	cells := splitRow(line)
	row := p.addChild(t, TableRow, start)
	p.addCells(row, start, line, cells, int(p.d.blocks[t].parse.columns))
	p.finalize(row)
	p.advanceOffset(len(p.ln)-p.offset, false)
}

// addCells adds the cells of a row, as many as the table has columns,
// those missing empty.
func (p *blockParser) addCells(row, base int, line string, cells [][2]int, columns int) {
	for k := 0; k < columns; k++ {
		at := base + len(line)
		if k < len(cells) {
			at = base + cells[k][0]
		}
		c := p.addChild(row, TableCell, at)
		if k < len(cells) && cells[k][1] > cells[k][0] {
			p.d.appendSeg(c, seg{start: int32(base + cells[k][0]), end: int32(base + cells[k][1]), cell: true})
			p.d.blocks[c].End = base + cells[k][1]
		}
		p.finalize(c)
	}
}

// addTextToContainer adds what remains of the line to container, or to
// the paragraph it lazily continues.
func (p *blockParser) addTextToContainer(container int) {
	d := p.d
	p.findFirstNonspace()
	cb := &d.blocks[container]
	if p.blank && cb.parse.lastChild >= 0 {
		d.blocks[cb.parse.lastChild].parse.lastLineBlank = true
	}
	ck := cb.Kind
	lastLineBlank := p.blank && ck != Quote && ck != Heading && ck != ThematicBreak &&
		!(ck == CodeBlock && cb.Fenced) && ck != MathBlock &&
		!(ck == Item && cb.parse.lastChild < 0 && int(cb.parse.startLine) == p.lineNum)
	cb.parse.lastLineBlank = lastLineBlank
	for t := cb.Parent; t >= 0; t = d.blocks[t].Parent {
		d.blocks[t].parse.lastLineBlank = false
	}

	if p.current != p.lastMatchedContainer && container == p.lastMatchedContainer && !p.blank && d.blocks[p.current].Kind == Paragraph {
		// A lazy continuation line.
		p.addLine(p.current)
		return
	}
	for p.current != p.lastMatchedContainer {
		p.current = p.finalize(p.current)
	}
	cb = &d.blocks[container]
	switch {
	case cb.Kind == CodeBlock:
		// A fence's opening line holds its info string, not code.
		if !cb.Fenced || int(cb.parse.startLine) != p.lineNum {
			p.addLine(container)
		}
	case cb.Kind == MathBlock && int(cb.parse.startLine) == p.lineNum:
		// Its opening line, which mathStart took.
	case cb.Kind == MathBlock && !cb.parse.mathDone:
		rest := p.ln[p.offset:]
		if cl := cb.parse.mathClose(); true {
			if j := strings.LastIndex(rest, cl); j >= 0 && strings.Trim(rest[j+len(cl):], " \t") == "" {
				if strings.Trim(rest[:j], " \t") != "" {
					p.addLine(container)
					cb = &d.blocks[container]
					d.segs[cb.segB-1].end = int32(p.lineStart + p.offset + j)
				}
				cb.parse.mathDone = true
				cb.End = p.lineStart + len(p.ln)
			} else {
				p.addLine(container)
			}
		}
	case cb.Kind == HTMLBlock:
		p.addLine(container)
		if htmlBlockEnds(int(cb.parse.htmlType), p.ln[p.firstNonspace:]) {
			container = p.finalize(container)
		}
	case p.blank:
	case acceptsLines(cb.Kind) && cb.Kind != MathBlock:
		if cb.Kind == Heading {
			p.addHeadingText(container)
		} else {
			p.advanceOffset(p.firstNonspace-p.offset, false)
			p.addLine(container)
		}
	default:
		container = p.addChild(container, Paragraph, p.lineStart+p.firstNonspace)
		p.advanceOffset(p.firstNonspace-p.offset, false)
		p.addLine(container)
	}
	p.current = container
}

// addHeadingText adds the text of an ATX heading, without its closing
// sequence of #.
func (p *blockParser) addHeadingText(h int) {
	p.advanceOffset(p.firstNonspace-p.offset, false)
	start, end := p.offset, len(p.ln)
	for end > start && isSpaceOrTab(p.ln[end-1]) {
		end--
	}
	k := end
	for k > start && p.ln[k-1] == '#' {
		k--
	}
	if k == start {
		end = start
	} else if k < end && isSpaceOrTab(p.ln[k-1]) {
		end = k
		for end > start && isSpaceOrTab(p.ln[end-1]) {
			end--
		}
	}
	p.d.appendSeg(h, seg{start: int32(p.lineStart + start), end: int32(p.lineStart + end)})
	p.block(h).End = p.lineStart + len(p.ln)
}

// scanThematicBreak returns the length of a thematic break at i, or 0,
// noting where scanning failed so that later tries on the line stop
// there.
func (p *blockParser) scanThematicBreak(i int) int {
	c := p.peek(i)
	if c != '*' && c != '_' && c != '-' {
		p.thematicBreakKillPos = i
		return 0
	}
	count := 1
	j := i + 1
	for ; j < len(p.ln); j++ {
		if p.ln[j] == c {
			count++
		} else if p.ln[j] != ' ' && p.ln[j] != '\t' {
			break
		}
	}
	if count >= 3 && j == len(p.ln) {
		return j - i + 1
	}
	p.thematicBreakKillPos = j
	return 0
}

// contentOf returns the content of leaf block i, as one string, and the
// source offset of its first byte when it is a slice of the source, else
// -1.
func (d *Document) contentOf(i int) (string, int) {
	b := &d.blocks[i]
	if b.text != "" || b.segB == b.segA {
		return b.text, int(b.textS)
	}
	segs := d.segsOf(b)
	if segs[0].cell {
		s := d.src[segs[0].start:segs[0].end]
		if strings.Contains(s, `\|`) {
			b.text, b.textS = strings.ReplaceAll(s, `\|`, "|"), -1
		} else {
			b.text, b.textS = s, segs[0].start
		}
		return b.text, int(b.textS)
	}
	// A slice of the source when the lines follow each other with \n.
	contiguous := true
	for k, s := range segs {
		if s.pad != 0 {
			contiguous = false
			break
		}
		if k < len(segs)-1 && (int(s.end) >= len(d.src) || d.src[s.end] != '\n' || segs[k+1].start != s.end+1) {
			contiguous = false
			break
		}
	}
	last := segs[len(segs)-1]
	if contiguous {
		// Without the final newline, which nothing needs.
		b.text, b.textS = d.src[segs[0].start:last.end], segs[0].start
		return b.text, int(b.textS)
	}
	var sb strings.Builder
	for k, s := range segs {
		for range s.pad {
			sb.WriteByte(' ')
		}
		sb.WriteString(d.src[s.start:s.end])
		if k < len(segs)-1 {
			sb.WriteByte('\n')
		}
	}
	b.text, b.textS = sb.String(), -1
	return b.text, int(b.textS)
}

// sourceMap maps offsets in the content of a block to the source.
type sourceMap struct {
	d *Document
	i int
	// marks are, for content that is not a slice of the source, the
	// offset in it where each line starts, or in a table cell the offsets
	// of the | written \|.
	marks []int
}

// sourceMap returns the map of block i's content, valid until the
// block or the next map changes. Maps of content joined from lines look
// up an offset by binary search, so that the offsets of all the inlines
// of a paragraph of many lines take n log n time, not quadratic.
func (d *Document) sourceMap(i int) sourceMap {
	b := &d.blocks[i]
	m := sourceMap{d: d, i: i}
	if b.textS >= 0 {
		return m
	}
	marks := d.marks[:0]
	segs := d.segsOf(b)
	if len(segs) > 0 && segs[0].cell {
		src := d.src[segs[0].start:segs[0].end]
		d.steps += len(src)
		o := 0
		for j := 0; j < len(src); j++ {
			if src[j] == '\\' && j+1 < len(src) && src[j+1] == '|' {
				j++
				marks = append(marks, o)
			}
			o++
		}
	} else {
		d.steps += len(segs)
		o := 0
		for _, s := range segs {
			marks = append(marks, o)
			o += int(s.pad) + int(s.end-s.start) + 1
		}
	}
	d.marks, m.marks = marks, marks
	return m
}

// source returns the source offset of byte off of the content.
func (m sourceMap) source(off int) int {
	b := &m.d.blocks[m.i]
	m.d.steps++
	if b.textS >= 0 {
		return int(b.textS) + off
	}
	segs := m.d.segsOf(b)
	if len(segs) == 0 {
		return b.Start
	}
	if s := segs[0]; s.cell {
		// Each \| read as | shifts what follows by one.
		n := int(s.end-s.start) - len(m.marks)
		if off > n {
			return int(s.start) + min(off, int(s.end-s.start))
		}
		k, _ := slices.BinarySearch(m.marks, off)
		return int(s.start) + off + k
	}
	// The line holding off: the last starting at or before it.
	k, _ := slices.BinarySearch(m.marks, off+1)
	s := segs[k-1]
	return int(s.start) + max(min(off-m.marks[k-1]-int(s.pad), int(s.end-s.start)), 0)
}
