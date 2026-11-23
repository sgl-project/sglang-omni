package editor

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// apply makes edits to the buffer as one step of undo, and gives the
// selections after them: after maps each selection, given a function
// moving offsets through the edits; nil moves them as the text around
// them moves.
func (ed *editor) apply(edits []Edit, kind editKind, after func(move func(int, bias) int) []sel) bool {
	b := ed.buf()
	if b == nil || ed.o.readOnly || len(edits) == 0 {
		return false
	}
	before := slices.Clone(ed.sels)
	b.mu.Lock()
	if b.version != ed.view.version {
		// The buffer changed since this editor last looked, as another
		// goroutine edited it: the edit, made for the text before, is
		// left for its command to run again on the text as it is
		// (retry).
		b.mu.Unlock()
		ed.stale = true
		return false
	}
	changes := b.edit(edits, kind, ed, before)
	made := b.version
	b.mu.Unlock()
	if changes == nil {
		// The edits change nothing, as pasting the text selected over
		// it: the selections still go where the edits put them, as the
		// caret after the text pasted.
		if after == nil {
			return false
		}
		sels := after(func(off int, _ bias) int { return off })
		if sels == nil {
			return false
		}
		ed.setSels(mergeSels(sels), revealFit)
		ed.blinkStart = ed.frameNow
		return true
	}
	ed.dropSuggestion()
	move := func(off int, bias bias) int { return mapOffset(off, changes, bias) }
	var sels []sel
	if after != nil {
		sels = after(move)
	}
	ed.sync()
	if sels != nil {
		// Placed in the text the edit made, and moved through the edits
		// made after it before the editor followed.
		if ed.view.version != made {
			entries, _, _ := b.entriesSince(made)
			for _, e := range entries {
				if e.version > ed.view.version {
					break
				}
				for i := range sels {
					sels[i] = mapSel(sels[i], e.changes)
				}
			}
		}
		ed.sels = mergeSels(sels)
	}
	b.setSelectionsAfter(ed, slices.Clone(ed.sels))
	ed.reveal = max(ed.reveal, revealFit)
	ed.blinkStart = ed.frameNow
	ed.needsBuild = true
	return true
}

// replaceEach replaces each selection with the text of fn and puts the
// caret after it, or selects it.
func (ed *editor) replaceEach(kind editKind, selectInserted bool, fn func(i int, s sel) string) bool {
	edits := make([]Edit, len(ed.sels))
	texts := make([]string, len(ed.sels))
	for i, s := range ed.sels {
		texts[i] = fn(i, s)
		edits[i] = Edit{Start: s.start(), End: s.end(), Text: texts[i]}
	}
	sels := slices.Clone(ed.sels)
	return ed.apply(edits, kind, func(move func(int, bias) int) []sel {
		delta := 0
		for i := range sels {
			s := &sels[i]
			start := s.start() + delta
			end := start + len(texts[i])
			delta += len(texts[i]) - (s.end() - s.start())
			if selectInserted {
				s.anchor, s.head = start, end
			} else {
				s.anchor, s.head = end, end
			}
			s.goal = -1
		}
		return sels
	})
}

// insert types text at every selection, closing and wrapping brackets.
func (ed *editor) insert(text string) {
	if ed.o.readOnly {
		return
	}
	if ed.o.autoClose && utf8.RuneCountInString(text) == 1 && ed.autoPair(text) {
		return
	}
	kind := editTyping
	if strings.ContainsAny(text, " \n\t") {
		kind = editOther
	}
	if ed.replaceEach(kind, false, func(int, sel) string { return text }) {
		ed.outdentTyped(text)
	}
	ed.afterTyping(text)
}

// outdentTyped outdents the lines that the text just typed closes a
// block on, as Zed's auto-indent on input does: a closing bracket typed
// first on its line takes the indentation of its opener's line, and a
// line that the language's DecreaseIndent matches once typed, as end,
// one level less than the line above suggests.
func (ed *editor) outdentTyped(t string) {
	l := ed.language()
	if l == nil || strings.Contains(t, "\n") {
		return
	}
	r := &ed.view.text
	var edits []Edit
	last := -1
	for _, s := range ed.sels {
		line := r.LineOf(s.head)
		if !s.empty() || line == last {
			continue
		}
		last = line
		a := r.LineStart(line)
		text := r.Line(line)
		indent := indentOf(text)
		col := s.head - a
		if col < len(indent)+len(t) || text[col-len(t):col] != t {
			continue
		}
		target, ok := ed.closedIndent(line, a+len(indent), text, col, t)
		if ok && indentWidth(target, ed.tabSize()) < indentWidth(indent, ed.tabSize()) {
			edits = append(edits, Edit{Start: a, End: a + len(indent), Text: target})
		}
	}
	ed.apply(edits, editTyping, nil)
}

// closedIndent returns the indentation of line, whose text is text, now
// that t was typed before column col: whether it closes a block.
func (ed *editor) closedIndent(line, at int, text string, col int, t string) (string, bool) {
	l := ed.language()
	r := &ed.view.text
	word := text[at-r.LineStart(line) : col]
	for _, p := range l.Brackets {
		if p.Open == "<" || word != p.Close || t != p.Close {
			continue
		}
		if open, close, ok := ed.bracketsAround(at, at+len(t)); ok && close.Start == at {
			return indentOf(r.Line(r.LineOf(open.Start))), true
		}
		return "", false
	}
	if l.DecreaseIndent == nil || !l.DecreaseIndent.MatchString(text) ||
		l.DecreaseIndent.MatchString(text[:col-len(t)]+text[col:]) {
		return "", false
	}
	// One level less than the line above, and the one it opens.
	above := line - 1
	for above >= 0 && strings.TrimSpace(r.Line(above)) == "" {
		above--
	}
	if above < 0 {
		return "", false
	}
	prev := r.Line(above)
	indent := indentOf(prev)
	if ed.increasesIndent(prev) {
		return indent, true
	}
	return outdentOnce(indent, ed.tabSize()), true
}

// autoPair closes a bracket or quote typed, wraps the selections in it,
// or steps over the closing one it typed before. It reports whether it
// did.
func (ed *editor) autoPair(t string) bool {
	l := ed.language()
	if l == nil {
		return false
	}
	r := &ed.view.text
	// Typing the closer just typed steps over it, at every caret.
	closers := map[int]bool{}
	for _, a := range ed.autoclosed {
		if a.close == t {
			closers[a.at] = true
		}
	}
	allOver := len(closers) > 0
	for _, s := range ed.sels {
		if !s.empty() || !closers[s.head] || r.Slice(s.head, s.head+len(t)) != t {
			allOver = false
			break
		}
	}
	if allOver {
		sels := slices.Clone(ed.sels)
		for i := range sels {
			sels[i].collapse(sels[i].head + len(t))
		}
		ed.autoclosed = slices.DeleteFunc(ed.autoclosed, func(a autoclose) bool { return a.close == t && closers[a.at] })
		ed.sels = sels
		ed.reveal = max(ed.reveal, revealFit)
		return true
	}
	var pair *BracketPair
	for i := range l.AutoClose {
		if l.AutoClose[i].Open == t {
			pair = &l.AutoClose[i]
			break
		}
	}
	if pair == nil {
		return false
	}
	// Selections are wrapped in the pair, when there are only selections.
	if slices.ContainsFunc(ed.sels, func(s sel) bool { return !s.empty() }) {
		if slices.ContainsFunc(ed.sels, func(s sel) bool { return s.empty() }) {
			return false
		}
		var edits []Edit
		for _, s := range ed.sels {
			edits = append(edits, Edit{Start: s.start(), End: s.start(), Text: pair.Open}, Edit{Start: s.end(), End: s.end(), Text: pair.Close})
		}
		sels := slices.Clone(ed.sels)
		return ed.apply(edits, editOther, func(move func(int, bias) int) []sel {
			for i := range sels {
				s := &sels[i]
				a, z := move(s.start(), biasRight), move(s.end(), biasLeft)
				if s.head < s.anchor {
					s.anchor, s.head = z, a
				} else {
					s.anchor, s.head = a, z
				}
			}
			return sels
		})
	}
	// Close it, where what follows allows: the end of the line,
	// whitespace or a closing bracket, and not inside a string or a
	// comment for quotes.
	for _, s := range ed.sels {
		next, _ := r.RuneAt(s.head)
		if s.head < r.Len() && next != '\n' && !unicode.IsSpace(next) && !strings.ContainsRune(")]}>,;:", next) {
			return false
		}
		if pair.Open == pair.Close {
			// A quote after a word character starts nothing.
			prev, _ := r.RuneBefore(s.head)
			if isWordRune(prev) {
				return false
			}
		}
		if len(pair.NotIn) > 0 && slices.Contains(pair.NotIn, ed.tokenAt(s.head)) {
			return false
		}
	}
	ok := ed.replaceEach(editTyping, false, func(int, sel) string { return pair.Open + pair.Close })
	if !ok {
		return false
	}
	for i := range ed.sels {
		s := &ed.sels[i]
		s.collapse(s.head - len(pair.Close))
		ed.autoclosed = append(ed.autoclosed, autoclose{sel: s.id, at: s.head, close: pair.Close})
	}
	ed.afterTyping(t)
	return true
}

// tokenAt returns the kind of the token holding the byte before off, or
// at it at the start of a line.
func (ed *editor) tokenAt(off int) TokenKind {
	v := &ed.view
	l := v.line(v.text.LineOf(off))
	return kindAt(l.toks, max(off-l.start-1, 0))
}

// afterTyping asks for completions as a word is typed.
func (ed *editor) afterTyping(t string) {
	if ed.o.onCompletion == nil {
		return
	}
	r, _ := utf8.DecodeRuneInString(t)
	if isWordRune(r) || strings.ContainsAny(t, ".:>") {
		ed.requestCompletions(false)
	} else {
		ed.complete.close()
	}
}

// indentUnit is the text a level of indentation inserts.
func (ed *editor) indentUnit() string {
	if ed.hardTabs() {
		return "\t"
	}
	return spaces(ed.tabSize())
}

// undo takes back (or makes again) the buffer's last step, and puts this
// editor's selections where they were, when it made the step.
func (ed *editor) undo(redo bool) bool {
	b := ed.buf()
	if b == nil || ed.o.readOnly {
		return false
	}
	b.mu.Lock()
	var tx transaction
	var ok bool
	if redo {
		tx, ok = b.redoStep()
	} else {
		tx, ok = b.undoStep()
	}
	b.mu.Unlock()
	if !ok {
		return false
	}
	ed.sync()
	saved := tx.selBefore
	if redo {
		saved = tx.selAfter
	}
	if s, ok := saved.([]sel); ok && tx.owner == ed && s != nil {
		ed.sels = mergeSels(ed.clampSels(s))
	}
	ed.reveal = revealFit
	ed.needsBuild = true
	ed.complete.close()
	return true
}

// copy writes the selections to the clipboard: a caret alone copies its
// whole line, which pastes above the line of a caret alone, as Zed and
// VS Code do.
func (ed *editor) copy(cut bool) {
	r := &ed.view.text
	var pieces []string
	var whole []bool
	var edits []Edit
	for _, s := range ed.sels {
		if s.empty() {
			line := r.LineOf(s.head)
			a, z := r.LineStart(line), min(r.Len(), r.LineEnd(line)+1)
			t := r.Slice(a, z)
			if !strings.HasSuffix(t, "\n") {
				t += "\n"
			}
			pieces, whole = append(pieces, t), append(whole, true)
			edits = append(edits, Edit{Start: a, End: z})
			continue
		}
		pieces, whole = append(pieces, r.Slice(s.start(), s.end())), append(whole, false)
		edits = append(edits, Edit{Start: s.start(), End: s.end()})
	}
	var text strings.Builder
	for i, p := range pieces {
		if i > 0 && !whole[i-1] {
			text.WriteByte('\n')
		}
		text.WriteString(p)
	}
	lastClip = clipboard{text: text.String(), pieces: pieces, lines: whole}
	ed.services.WriteClipboard(lastClip.text)
	if cut && !ed.o.readOnly {
		ed.apply(edits, editOther, nil)
	}
}

// paste inserts text at the selections: a piece each when the clipboard
// holds what this editor copied from as many selections, and whole lines
// above the lines of carets.
func (ed *editor) paste(text string) {
	if ed.o.readOnly || text == "" {
		return
	}
	text = strings.ReplaceAll(text, "\r\n", "\n")
	r := &ed.view.text
	pieces := []string{text}
	whole := []bool{false}
	clip := lastClip
	switch {
	case clip.text != text:
	case len(clip.pieces) == len(ed.sels):
		pieces, whole = clip.pieces, clip.lines
	case !slices.Contains(clip.lines, false):
		// Whole lines, from one caret or several: pasted as lines above
		// each caret's, as Zed's.
		whole = []bool{true}
	}
	var edits []Edit
	for i, s := range ed.sels {
		p, w := pieces[0], whole[0]
		if len(pieces) == len(ed.sels) {
			p, w = pieces[i], whole[i]
		}
		if w && s.empty() {
			a := r.LineStart(r.LineOf(s.head))
			edits = append(edits, Edit{Start: a, End: a, Text: p})
			continue
		}
		edits = append(edits, Edit{Start: s.start(), End: s.end(), Text: p})
	}
	sels := slices.Clone(ed.sels)
	ed.apply(edits, editOther, func(move func(int, bias) int) []sel {
		for i := range sels {
			sels[i].collapse(move(sels[i].end(), biasRight))
		}
		return sels
	})
}

// deleteMove deletes each selection, or from each caret to where fn
// takes it.
func (ed *editor) deleteMove(fn func(sel) int) bool {
	var edits []Edit
	for _, s := range ed.sels {
		if s.empty() {
			to := fn(s)
			edits = append(edits, Edit{Start: min(to, s.head), End: max(to, s.head)})
			continue
		}
		edits = append(edits, Edit{Start: s.start(), End: s.end()})
	}
	return ed.apply(mergeEdits(edits), editDeleting, nil)
}

// mergeEdits sorts edits and merges those that overlap.
func mergeEdits(edits []Edit) []Edit {
	slices.SortFunc(edits, func(a, b Edit) int { return a.Start - b.Start })
	out := edits[:0]
	for _, e := range edits {
		if n := len(out); n > 0 && e.Start <= out[n-1].End && e.Text == "" && out[n-1].Text == "" {
			out[n-1].End = max(out[n-1].End, e.End)
			continue
		}
		out = append(out, e)
	}
	return out
}

// backspace deletes before the carets: back to the previous tab stop in
// indentation, a pair of brackets just opened, else a character.
func (ed *editor) backspace() bool {
	r := &ed.view.text
	tab := ed.tabSize()
	var edits []Edit
	for _, s := range ed.sels {
		if !s.empty() {
			edits = append(edits, Edit{Start: s.start(), End: s.end()})
			continue
		}
		line := r.LineOf(s.head)
		a := r.LineStart(line)
		before := r.Slice(a, s.head)
		// In the indentation, back to the previous tab stop.
		if before != "" && strings.TrimLeft(before, " ") == "" && !ed.hardTabs() {
			col := len(before)
			to := (col - 1) / tab * tab
			edits = append(edits, Edit{Start: a + to, End: s.head})
			continue
		}
		start := graphemeLeft(*r, s.head)
		end := s.head
		// A pair of brackets just opened, its closer closed by the editor,
		// goes together, as Zed's.
		if l := ed.language(); l != nil && ed.o.autoClose {
			prev := r.Slice(start, s.head)
			for _, p := range l.AutoClose {
				if p.Open == prev && r.Slice(s.head, s.head+len(p.Close)) == p.Close && ed.autoclosedAt(s.head, p.Close) {
					end = s.head + len(p.Close)
					break
				}
			}
		}
		edits = append(edits, Edit{Start: start, End: end})
	}
	return ed.apply(mergeEdits(edits), editDeleting, nil)
}

// newline breaks the line at each caret, indenting the new line as the
// language says: one more level after an opening bracket, with the
// closing bracket on a line of its own, and continuing line comments.
func (ed *editor) newline() bool {
	r := &ed.view.text
	texts := make([]string, len(ed.sels))
	carets := make([]int, len(ed.sels))
	for i, s := range ed.sels {
		line := r.LineOf(s.start())
		a := r.LineStart(line)
		before := r.Slice(a, s.start())
		after := r.Slice(s.end(), r.LineEnd(r.LineOf(s.end())))
		indent := indentOf(r.Line(line))
		if len(indent) > len(before) {
			indent = before
		}
		next := indent
		levels := ed.indentAfter(s.start(), before)
		for ; levels > 0; levels-- {
			next += ed.indentUnit()
		}
		for ; levels < 0 && next != ""; levels++ {
			next = outdentOnce(next, ed.tabSize())
		}
		// Return in a line comment goes on with it, as Zed's
		// extend_comment_on_newline.
		if l := ed.language(); l != nil && l.LineComment != "" && strings.HasPrefix(strings.TrimLeft(before, " \t"), l.LineComment) {
			next = indent + l.LineComment + " "
		}
		t := "\n" + next
		caret := len(t)
		// Between brackets, the closer goes on a line of its own.
		if ed.closesBefore(before, after) {
			t += "\n" + indent
		}
		texts[i], carets[i] = t, caret
	}
	var edits []Edit
	for i, s := range ed.sels {
		start := s.start()
		line := r.LineOf(start)
		a := r.LineStart(line)
		if strings.TrimSpace(r.Slice(a, start)) == "" && r.Slice(start, r.LineEnd(line)) == "" && start > a {
			// Trailing whitespace of a blank line goes.
			edits = append(edits, Edit{Start: a, End: s.end(), Text: texts[i]})
			continue
		}
		edits = append(edits, Edit{Start: start, End: s.end(), Text: texts[i]})
	}
	sels := slices.Clone(ed.sels)
	return ed.apply(edits, editOther, func(move func(int, bias) int) []sel {
		delta := 0
		for i := range sels {
			e := edits[i]
			at := e.Start + delta + carets[i]
			delta += len(e.Text) - (e.End - e.Start)
			sels[i].collapse(at)
		}
		return sels
	})
}

// indentAfter returns how many levels more than its line a line Return
// opens at off is indented: as the highlighter says, when it knows, else
// one after a line ending in an opening bracket, or matching the
// language's IncreaseIndent.
func (ed *editor) indentAfter(off int, before string) int {
	if in, ok := ed.highlighter().(Indenter); ok {
		if levels, ok := in.IndentAfter(off); ok {
			return levels
		}
	}
	if ed.increasesIndent(before) {
		return 1
	}
	return 0
}

// increasesIndent reports whether the line before the caret, before,
// indents the next.
func (ed *editor) increasesIndent(before string) bool {
	l := ed.language()
	if l == nil {
		return false
	}
	if l.IncreaseIndent != nil {
		return l.IncreaseIndent.MatchString(before)
	}
	t := strings.TrimRight(before, " \t")
	if l.LineComment != "" {
		if i := strings.Index(t, l.LineComment); i >= 0 && !strings.ContainsAny(t[:i], "\"'`") {
			t = strings.TrimRight(t[:i], " \t")
		}
	}
	for _, p := range l.Brackets {
		if strings.HasSuffix(t, p.Open) && p.Open != "<" {
			return true
		}
	}
	return false
}

// closesBefore reports whether the caret is between an opening bracket
// and its closer.
func (ed *editor) closesBefore(before, after string) bool {
	l := ed.language()
	if l == nil {
		return false
	}
	b := strings.TrimRight(before, " \t")
	a := strings.TrimLeft(after, " \t")
	for _, p := range l.Brackets {
		if p.Open != "<" && strings.HasSuffix(b, p.Open) && strings.HasPrefix(a, p.Close) {
			return true
		}
	}
	return false
}

// newlineAround starts a line below (or above) the lines of the
// selections, indented as theirs.
func (ed *editor) newlineAround(above bool) bool {
	r := &ed.view.text
	var edits []Edit
	var carets []int
	for _, s := range ed.sels {
		a, z := ed.lineSpanOf(s)
		if above {
			indent := indentOf(r.Line(a))
			at := r.LineStart(a)
			edits = append(edits, Edit{Start: at, End: at, Text: indent + "\n"})
			carets = append(carets, len(indent))
			continue
		}
		line := r.Line(z)
		indent := indentOf(line)
		if ed.increasesIndent(line) {
			indent += ed.indentUnit()
		}
		at := r.LineEnd(z)
		edits = append(edits, Edit{Start: at, End: at, Text: "\n" + indent})
		carets = append(carets, 1+len(indent))
	}
	sels := slices.Clone(ed.sels)
	return ed.apply(edits, editOther, func(move func(int, bias) int) []sel {
		delta := 0
		for i := range sels {
			e := edits[i]
			sels[i].collapse(e.Start + delta + carets[i])
			delta += len(e.Text)
		}
		return sels
	})
}

// outdentOnce takes a level of indentation off indent.
func outdentOnce(indent string, tab int) string {
	if strings.HasSuffix(indent, "\t") {
		return indent[:len(indent)-1]
	}
	n := len(indent) - len(strings.TrimRight(indent, " "))
	return indent[:len(indent)-min(n, max(1, (n-1)%tab+1))]
}

// autoclosedAt reports whether the editor closed the bracket close at
// off, as it was opened.
func (ed *editor) autoclosedAt(off int, close string) bool {
	return slices.ContainsFunc(ed.autoclosed, func(a autoclose) bool { return a.at == off && a.close == close })
}
