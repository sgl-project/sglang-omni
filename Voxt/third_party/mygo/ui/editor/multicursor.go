package editor

import (
	"slices"
	"strings"
	"unicode/utf8"
)

// selectNext adds the next occurrence of the newest selection's text to
// the selections, as Zed's select next: a caret alone selects its word
// first, and looks for whole words then.
func (ed *editor) selectNext(prev bool) bool {
	r := &ed.view.text
	p := ed.primary()
	if p.empty() {
		a, z := wordAt(*r, p.head, ed.wordChars())
		if a == z {
			return false
		}
		sels := slices.Clone(ed.sels)
		for i := range sels {
			if sels[i].id == p.id {
				sels[i].anchor, sels[i].head = a, z
			}
		}
		ed.setSels(sels, revealFit)
		// The next ones are whole words too, as the first was a word.
		ed.wholeWord = true
		return true
	}
	query := r.Slice(p.start(), p.end())
	text := r.String()
	var at int
	if prev {
		at = searchBack(text, query, p.start(), ed.wholeWord, ed.wordChars(), func(i int) bool { return !ed.selected(i, i+len(query)) })
	} else {
		at = searchFwd(text, query, p.end(), ed.wholeWord, ed.wordChars(), func(i int) bool { return !ed.selected(i, i+len(query)) })
	}
	if at < 0 {
		return false
	}
	whole := ed.wholeWord
	ed.setSels(append(slices.Clone(ed.sels), ed.newSel(at, at+len(query))), revealFit)
	ed.wholeWord = whole
	return true
}

func (ed *editor) selected(a, b int) bool {
	for _, s := range ed.sels {
		if s.start() < b && s.end() > a {
			return true
		}
	}
	return false
}

// selectAllMatches selects every occurrence of the newest selection's
// text, or of the word at the caret.
func (ed *editor) selectAllMatches() bool {
	r := &ed.view.text
	p := ed.primary()
	whole := ed.wholeWord
	if p.empty() {
		a, z := wordAt(*r, p.head, ed.wordChars())
		if a == z {
			return false
		}
		p.anchor, p.head = a, z
		whole = true
	}
	query := r.Slice(p.start(), p.end())
	text := r.String()
	var sels []sel
	for at := 0; at <= len(text); {
		i := strings.Index(text[at:], query)
		if i < 0 {
			break
		}
		i += at
		if i != p.start() && (!whole || isWholeWord(text, i, i+len(query), ed.wordChars())) {
			sels = append(sels, ed.newSel(i, i+len(query)))
		}
		at = i + len(query)
	}
	// The original stays the newest.
	ed.nextID++
	p.id = ed.nextID
	sels = append(sels, p)
	ed.setSels(sels, revealNone)
	return true
}

// searchFwd returns the first match of query at or after at that ok
// takes, going round to the start, or -1.
func searchFwd(text, query string, at int, whole bool, extra string, ok func(int) bool) int {
	if query == "" {
		return -1
	}
	at = max(0, min(at, len(text)))
	// The matches starting from from to limit.
	find := func(from, limit int) int {
		for from < limit {
			i := strings.Index(text[from:], query)
			if i < 0 || from+i >= limit {
				return -1
			}
			i += from
			if (!whole || isWholeWord(text, i, i+len(query), extra)) && (ok == nil || ok(i)) {
				return i
			}
			from = i + 1
		}
		return -1
	}
	if i := find(at, len(text)); i >= 0 {
		return i
	}
	return find(0, at)
}

// searchBack returns the last match of query starting before before that
// ok takes, going round to the end, or -1.
func searchBack(text, query string, before int, whole bool, extra string, ok func(int) bool) int {
	if query == "" {
		return -1
	}
	before = max(0, min(before, len(text)))
	// The matches starting from lo to hi, the last first.
	find := func(lo, hi int) int {
		for hi > lo {
			i := strings.LastIndex(text[:min(len(text), hi-1+len(query))], query)
			if i < lo {
				return -1
			}
			if (!whole || isWholeWord(text, i, i+len(query), extra)) && (ok == nil || ok(i)) {
				return i
			}
			hi = i
		}
		return -1
	}
	if i := find(0, before); i >= 0 {
		return i
	}
	return find(before, len(text))
}

// isWholeWord reports whether text from a to b is not inside a word.
func isWholeWord(text string, a, b int, extra string) bool {
	if a > 0 {
		r, _ := utf8.DecodeLastRuneInString(text[:a])
		if classOf(r, extra) == classWord {
			return false
		}
	}
	if b < len(text) {
		r, _ := utf8.DecodeRuneInString(text[b:])
		if classOf(r, extra) == classWord {
			return false
		}
	}
	return true
}

// addCursor adds a caret on the row above (or below) the newest, at the
// same x, as Zed's add selection above: going back the other way takes
// the carets added back.
func (ed *editor) addCursor(dir int) bool {
	if ed.addDir == -dir && len(ed.addStack) > 0 {
		id := ed.addStack[len(ed.addStack)-1]
		ed.addStack = ed.addStack[:len(ed.addStack)-1]
		sels := slices.DeleteFunc(slices.Clone(ed.sels), func(s sel) bool { return s.id == id })
		if len(sels) > 0 {
			ed.sels = mergeSels(sels)
			ed.reveal = revealFit
			return true
		}
	}
	if len(ed.addStack) == 0 {
		ed.addDir = dir
	}
	p := ed.primary()
	r := &ed.view.text
	from := p.head
	// From the edge of the selections in the direction.
	for _, s := range ed.sels {
		if dir < 0 && s.head < from || dir > 0 && s.head > from {
			from = s.head
		}
	}
	line := r.LineOf(from)
	if dir < 0 && line == 0 || dir > 0 && line >= r.LineCount()-1 {
		return false
	}
	goal := p.goal
	if goal < 0 {
		if cr, ok := ed.caretRect(p.head); ok {
			goal = cr.X
		}
	}
	to, _ := ed.verticalMove(from, goal, dir)
	s := ed.newSel(to, to)
	s.goal = goal
	ed.addStack = append(ed.addStack, s.id)
	ed.sels = mergeSels(append(slices.Clone(ed.sels), s))
	ed.reveal = revealFit
	return true
}

// selectLarger grows each selection, as Zed's select larger syntax node
// without a syntax tree: the word, the inside of the string or the
// brackets around it, the brackets with them, the line, then the lines
// between blank lines.
func (ed *editor) selectLarger() bool {
	sels := slices.Clone(ed.sels)
	var before []sel
	for i := range sels {
		s := &sels[i]
		a, z := s.start(), s.end()
		if na, nz := ed.larger(a, z); na < a || nz > z {
			before = append(before, sel{anchor: s.anchor, head: s.head, id: s.id})
			s.anchor, s.head = na, nz
		}
	}
	if len(before) == 0 {
		return false
	}
	larged := ed.larged
	ed.setSels(sels, revealFit)
	// Select Smaller takes them back.
	ed.larged = append(larged, before...)
	return true
}

// larger returns the range around a to z the next step selects: the
// highlighter's syntax node, when it knows the tree, else the word, the
// string, the brackets, the line and the text, in turn.
func (ed *editor) larger(a, z int) (int, int) {
	if ss, ok := ed.highlighter().(SyntaxSelector); ok {
		if n, ok := ss.Enclosing(a, z); ok && n.Start <= a && n.End >= z && n.End-n.Start > z-a {
			return n.Start, n.End
		}
	}
	r := &ed.view.text
	if a == z {
		// The word at the caret, not a run of punctuation.
		wa, wz := wordAt(*r, a, ed.wordChars())
		if c, _ := r.RuneAt(wa); wz > wa && classOf(c, ed.wordChars()) == classWord {
			return wa, wz
		}
	}
	// The string around the selection.
	if sa, sz, ok := ed.stringAround(a, z); ok {
		if sa < a || sz > z {
			return sa, sz
		}
		if sa-1 >= 0 && (sa-1 < a || sz+1 > z) && sa > r.LineStart(r.LineOf(sa)) {
			return sa - 1, sz + 1
		}
	}
	// The brackets around it: their inside, then them. A selection of
	// a pair grows to the pair around it.
	open, close, ok := ed.bracketsAround(a, z)
	if ok && open.Start == a && close.End == z {
		ok = false
		if a > 0 {
			open, close, ok = ed.bracketsAround(a-1, z)
		}
	}
	if ok {
		if open.End <= a && close.Start >= z && (open.End < a || close.Start > z) {
			return open.End, close.Start
		}
		return open.Start, close.End
	}
	la, lz := r.LineStart(r.LineOf(a)), r.LineEnd(r.LineOf(z))
	if la < a || lz > z {
		return la, lz
	}
	return 0, r.Len()
}

// stringAround finds the string token holding a to z, inside its quotes.
func (ed *editor) stringAround(a, z int) (int, int, bool) {
	v := &ed.view
	line := v.text.LineOf(a)
	if v.text.LineOf(z) != line {
		return 0, 0, false
	}
	l := v.line(line)
	from := 0
	for i, t := range l.toks {
		if t.Kind == TokenString || t.Kind == TokenEscape {
			// The run of string tokens.
			end := t.End
			j := i
			for j+1 < len(l.toks) && (l.toks[j+1].Kind == TokenString || l.toks[j+1].Kind == TokenEscape) {
				j++
				end = l.toks[j].End
			}
			sa, sz := l.start+from, l.start+end
			if sa <= a && z <= sz && sz-sa >= 2 {
				return sa + 1, sz - 1, true
			}
		}
		from = t.End
	}
	return 0, 0, false
}

// selectSmaller takes back the selections Select Larger grew.
func (ed *editor) selectSmaller() bool {
	if len(ed.larged) == 0 {
		return false
	}
	sels := slices.Clone(ed.sels)
	changed := false
	for i := range sels {
		for j := len(ed.larged) - 1; j >= 0; j-- {
			if ed.larged[j].id == sels[i].id {
				sels[i].anchor, sels[i].head = ed.larged[j].anchor, ed.larged[j].head
				ed.larged = slices.Delete(ed.larged, j, j+1)
				changed = true
				break
			}
		}
	}
	if changed {
		ed.sels = mergeSels(sels)
		ed.reveal = revealFit
	}
	return changed
}
