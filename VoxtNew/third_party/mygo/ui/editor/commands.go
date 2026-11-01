package editor

import (
	"slices"
	"strings"
)

// run runs a command, and reports whether it did something.
func (ed *editor) run(cmd Command) bool {
	r := &ed.view.text
	extra := ed.wordChars()
	switch cmd {
	case MoveLeft, SelectLeft:
		return ed.move(cmd == SelectLeft, func(s sel) int {
			if s.empty() || cmd == SelectLeft {
				return graphemeLeft(*r, s.head)
			}
			return s.start()
		})
	case MoveRight, SelectRight:
		return ed.move(cmd == SelectRight, func(s sel) int {
			if s.empty() || cmd == SelectRight {
				return graphemeRight(*r, s.head)
			}
			return s.end()
		})
	case MoveWordLeft, SelectWordLeft:
		return ed.move(cmd == SelectWordLeft, func(s sel) int { return wordLeft(*r, s.head, extra, false) })
	case MoveWordRight, SelectWordRight:
		return ed.move(cmd == SelectWordRight, func(s sel) int { return wordRight(*r, s.head, extra, false) })
	case MoveSubwordLeft, SelectSubwordLeft:
		return ed.move(cmd == SelectSubwordLeft, func(s sel) int { return wordLeft(*r, s.head, extra, true) })
	case MoveSubwordRight, SelectSubwordRight:
		return ed.move(cmd == SelectSubwordRight, func(s sel) int { return wordRight(*r, s.head, extra, true) })
	case MoveUp, SelectUp, MoveDown, SelectDown, MovePageUp, SelectPageUp, MovePageDown, SelectPageDown:
		if ed.complete.open() && (cmd == MoveUp || cmd == MoveDown) {
			ed.complete.step(direction(cmd == MoveUp))
			return true
		}
		rows := 1
		page := max(1, int(ed.viewH/ed.view.lineHeight)-2)
		switch cmd {
		case MoveUp, SelectUp:
			rows = -1
		case MovePageUp, SelectPageUp:
			rows = -page
		case MovePageDown, SelectPageDown:
			rows = page
		}
		extend := cmd == SelectUp || cmd == SelectDown || cmd == SelectPageUp || cmd == SelectPageDown
		sels := slices.Clone(ed.sels)
		for i := range sels {
			s := &sels[i]
			from := s.head
			if !extend && !s.empty() {
				if rows < 0 {
					from = s.start()
				} else {
					from = s.end()
				}
			}
			to, goal := ed.verticalMove(from, s.goal, rows)
			if extend {
				s.head = to
			} else {
				s.anchor, s.head = to, to
			}
			s.goal = goal
		}
		if rows != 1 && rows != -1 {
			ed.scrollBy(float64(rows) * float64(ed.view.lineHeight))
		}
		ed.setSels(sels, revealFit)
		ed.endGroup()
		return true
	case MoveLineStart, SelectLineStart:
		return ed.move(cmd == SelectLineStart, func(s sel) int { return ed.smartHome(s.head) })
	case MoveLineEnd, SelectLineEnd:
		return ed.move(cmd == SelectLineEnd, func(s sel) int { return ed.rowEnd(s.head) })
	case MoveDocStart, SelectDocStart:
		return ed.move(cmd == SelectDocStart, func(sel) int { return 0 })
	case MoveDocEnd, SelectDocEnd:
		return ed.move(cmd == SelectDocEnd, func(sel) int { return r.Len() })
	case MoveParagraphUp, SelectParagraphUp:
		return ed.move(cmd == SelectParagraphUp, func(s sel) int { return ed.paragraph(s.head, -1) })
	case MoveParagraphDown, SelectParagraphDown:
		return ed.move(cmd == SelectParagraphDown, func(s sel) int { return ed.paragraph(s.head, 1) })
	case MoveToBracket, SelectToBracket:
		// As Zed's: from either side of a bracket to the same side of
		// the other, inside or outside the pair.
		return ed.move(cmd == SelectToBracket, func(s sel) int {
			a, b, ok := ed.enclosingBrackets(s.head)
			if !ok {
				return s.head
			}
			inside := s.head >= a.End && s.head <= b.Start
			switch {
			case s.head >= b.Start && s.head <= b.End && inside:
				return a.End
			case s.head >= b.Start && s.head <= b.End:
				return a.Start
			case inside:
				return b.Start
			}
			return b.End
		})
	case SelectAll:
		ed.setSels([]sel{ed.newSel(0, r.Len())}, revealNone)
		return true
	case SelectLine:
		// The lines of the selections, and the line a selection of whole
		// lines ends at the start of, as pressing it again extends.
		sels := slices.Clone(ed.sels)
		for i := range sels {
			s := &sels[i]
			a, z := r.LineOf(s.start()), r.LineOf(s.end())
			s.anchor = r.LineStart(a)
			s.head = min(r.LineStart(z+1), r.Len())
			if z+1 >= r.LineCount() {
				s.head = r.Len()
			}
		}
		ed.setSels(sels, revealFit)
		return true
	case SelectNext, SelectPrevious:
		return ed.selectNext(cmd == SelectPrevious)
	case SelectAllMatches:
		return ed.selectAllMatches()
	case AddCursorAbove, AddCursorBelow:
		return ed.addCursor(direction(cmd == AddCursorAbove))
	case SelectLarger:
		return ed.selectLarger()
	case SelectSmaller:
		return ed.selectSmaller()
	case UndoSelection:
		if n := len(ed.selUndo); n > 0 {
			ed.selRedo = append(ed.selRedo, slices.Clone(ed.sels))
			ed.sels = mergeSels(ed.clampSels(ed.selUndo[n-1]))
			ed.selUndo = ed.selUndo[:n-1]
			ed.reveal = revealFit
			return true
		}
		return false
	case RedoSelection:
		if n := len(ed.selRedo); n > 0 {
			ed.selUndo = append(ed.selUndo, slices.Clone(ed.sels))
			ed.sels = mergeSels(ed.clampSels(ed.selRedo[n-1]))
			ed.selRedo = ed.selRedo[:n-1]
			ed.reveal = revealFit
			return true
		}
		return false
	case SplitSelectionIntoLines:
		// A selection of each line's part of each selection, as
		// Sublime's and Zed's.
		var sels []sel
		for _, s := range ed.sels {
			a, z := ed.lineSpanOf(s)
			for line := a; line <= z; line++ {
				sels = append(sels, ed.newSel(max(s.start(), r.LineStart(line)), min(s.end(), r.LineEnd(line))))
			}
		}
		ed.setSels(sels, revealNone)
		return true
	case Cancel:
		switch {
		case ed.complete.open():
			ed.complete.close()
		case ed.hover.open():
			ed.hover.close()
		case ed.goTo.open:
			ed.goTo.close(ed)
		case ed.find.open:
			ed.find.open = false
			ed.needsBuild = true
			ed.refocus = true
		case len(ed.sels) > 1:
			p := ed.primary()
			ed.setSels([]sel{p}, revealFit)
		case !ed.sels[0].empty():
			s := ed.sels[0]
			s.collapse(s.head)
			ed.setSels([]sel{s}, revealNone)
		default:
			return false
		}
		return true
	case Undo, Redo:
		return ed.undo(cmd == Redo)
	case Copy, Cut:
		ed.copy(cmd == Cut)
		return true
	case Paste:
		ed.paste(ed.services.ReadClipboard())
		return true
	case Backspace:
		return ed.backspace()
	case Delete:
		return ed.deleteMove(func(s sel) int { return graphemeRight(*r, s.head) })
	case DeleteWordLeft:
		return ed.deleteMove(func(s sel) int { return wordLeft(*r, s.head, extra, false) })
	case DeleteWordRight:
		return ed.deleteMove(func(s sel) int { return wordRight(*r, s.head, extra, false) })
	case DeleteSubwordLeft:
		return ed.deleteMove(func(s sel) int { return wordLeft(*r, s.head, extra, true) })
	case DeleteSubwordRight:
		return ed.deleteMove(func(s sel) int { return wordRight(*r, s.head, extra, true) })
	case DeleteToLineStart:
		return ed.deleteMove(func(s sel) int {
			if s.head == r.LineStart(r.LineOf(s.head)) {
				return graphemeLeft(*r, s.head)
			}
			return ed.smartHome(s.head)
		})
	case DeleteToLineEnd:
		return ed.deleteMove(func(s sel) int {
			if s.head == r.LineEnd(r.LineOf(s.head)) {
				return graphemeRight(*r, s.head)
			}
			return r.LineEnd(r.LineOf(s.head))
		})
	case CutToLineEnd:
		// As Emacs: the rest of the line, or its newline.
		var cut []string
		for _, s := range ed.sels {
			end := r.LineEnd(r.LineOf(s.head))
			if end == s.head {
				end = min(r.Len(), end+1)
			}
			cut = append(cut, r.Slice(s.head, end))
		}
		ed.services.WriteClipboard(strings.Join(cut, "\n"))
		return ed.deleteMove(func(s sel) int {
			end := r.LineEnd(r.LineOf(s.head))
			if end == s.head {
				return min(r.Len(), end+1)
			}
			return end
		})
	case DeleteLine:
		return ed.deleteLines()
	case Newline:
		if ed.complete.open() {
			return ed.acceptCompletion()
		}
		return ed.newline()
	case NewlineAbove, NewlineBelow:
		return ed.newlineAround(cmd == NewlineAbove)
	case MoveLinesUp, MoveLinesDown:
		return ed.moveLines(direction(cmd == MoveLinesUp))
	case DuplicateLinesUp, DuplicateLinesDown:
		return ed.duplicateLines(cmd == DuplicateLinesUp)
	case DuplicateSelection:
		if slices.ContainsFunc(ed.sels, func(s sel) bool { return s.empty() }) {
			return ed.duplicateLines(false)
		}
		return ed.replaceEach(editOther, true, func(_ int, s sel) string {
			t := r.Slice(s.start(), s.end())
			return t + t
		})
	case JoinLines:
		return ed.joinLines()
	case Transpose:
		return ed.transpose()
	case Indent:
		return ed.indentLines(1)
	case Outdent, Backtab:
		return ed.indentLines(-1)
	case Tab:
		if ed.complete.open() {
			return ed.acceptCompletion()
		}
		if ed.acceptSuggestion() {
			return true
		}
		return ed.tab()
	case ToggleComment:
		return ed.toggleComment()
	case ToggleBlockComment:
		return ed.toggleBlockComment()
	case SortLines, SortLinesCaseInsensitive, ReverseLines, UniqueLines:
		return ed.manipulateLines(cmd)
	case Uppercase, Lowercase, TitleCase, SnakeCase, KebabCase, CamelCase, PascalCase:
		return ed.convertCase(cmd)
	case TrimTrailingWhitespace:
		var edits []Edit
		for line := 0; line < r.LineCount(); line++ {
			s := r.Line(line)
			t := strings.TrimRight(s, " \t")
			if len(t) < len(s) {
				a := r.LineStart(line)
				edits = append(edits, Edit{Start: a + len(t), End: a + len(s)})
			}
		}
		return ed.apply(edits, editOther, nil)
	case Fold:
		return ed.foldAt(ed.primary().head)
	case Unfold:
		return ed.unfoldAt(ed.primary().head, false)
	case ToggleFold:
		if !ed.unfoldAt(ed.primary().head, false) {
			return ed.foldAt(ed.primary().head)
		}
		return true
	case FoldAll:
		return ed.foldLevel(0)
	case UnfoldAll:
		ed.folds = nil
		ed.needsBuild = true
		return true
	case FoldRecursive:
		return ed.foldRecursive(ed.primary().head)
	case UnfoldRecursive:
		return ed.unfoldAt(ed.primary().head, true)
	case FoldSelection:
		for _, s := range ed.sels {
			a, z := ed.lineSpanOf(s)
			if z > a {
				ed.folds = append(ed.folds, fold{a, z})
			}
		}
		ed.moveOutOfFolds()
		ed.needsBuild = true
		return true
	case FoldLevel1, FoldLevel2, FoldLevel3, FoldLevel4, FoldLevel5, FoldLevel6, FoldLevel7, FoldLevel8, FoldLevel9:
		return ed.foldLevel(int(cmd-FoldLevel1) + 1)
	case Find, FindReplace:
		ed.find.show(ed, cmd == FindReplace)
		return true
	case FindNext, FindPrevious:
		return ed.find.step(ed, direction(cmd == FindPrevious))
	case UseSelectionForFind:
		if t := ed.selectedText(); t != "" {
			ed.find.query = t
			ed.find.stale = true
			ed.needsBuild = true
		}
		return true
	case FindInSelection:
		ed.find.show(ed, false)
		ed.find.inSelection = !ed.find.inSelection
		ed.find.scope = nil
		ed.find.stale = true
		return true
	case GoToLine:
		ed.goTo.show(ed)
		return true
	case ScrollCursorCenter:
		if cr, ok := ed.caretRect(ed.primary().head); ok {
			ed.setScroll(float64(cr.Y) - float64(ed.viewH-ed.view.lineHeight)/2)
		}
		return true
	case ScrollLineUp, ScrollLineDown:
		ed.scrollBy(float64(direction(cmd == ScrollLineUp)) * float64(ed.view.lineHeight))
		return true
	case GoToDefinition:
		if ed.o.onDefinition != nil {
			ed.o.onDefinition(ed.primary().head)
			return true
		}
		return false
	case ShowCompletions:
		return ed.requestCompletions(true)
	case ShowHover:
		ed.hover.byPointer = false
		return ed.requestHover(ed.primary().head)
	case NextHunk, PreviousHunk:
		return ed.gotoHunk(direction(cmd == PreviousHunk))
	case RevertHunk:
		return ed.revertHunks()
	case ToggleSoftWrap:
		ed.toggledWrap = !ed.toggledWrap
		ed.needsBuild = true
		return true
	case ToggleLineNumbers:
		ed.toggledNumbers = !ed.toggledNumbers
		ed.needsBuild = true
		return true
	}
	return false
}

// move moves each caret to where fn takes it, extending the selections
// when extend is set.
func (ed *editor) move(extend bool, fn func(sel) int) bool {
	sels := slices.Clone(ed.sels)
	for i := range sels {
		s := &sels[i]
		to := ed.overFolds(s.head, fn(*s))
		if extend {
			s.head = to
		} else {
			s.anchor, s.head = to, to
		}
		s.goal = -1
	}
	ed.setSels(sels, revealFit)
	ed.endGroup()
	return true
}

// endGroup ends the group of typing that undo takes back as one step,
// as moving the carets does, and closes the completions.
func (ed *editor) endGroup() {
	if b := ed.buf(); b != nil {
		b.endGroup()
	}
	ed.complete.close()
}

// clampSels returns sels kept within the text.
func (ed *editor) clampSels(sels []sel) []sel {
	n := ed.view.text.Len()
	out := slices.Clone(sels)
	for i := range out {
		out[i].anchor, out[i].head = min(out[i].anchor, n), min(out[i].head, n)
	}
	return out
}

// smartHome goes to the first non-blank character of the row, or to its
// start from there, as Zed's beginning of line stops at the indentation.
func (ed *editor) smartHome(off int) int {
	v := &ed.view
	r := &v.text
	line := r.LineOf(off)
	l := v.line(line)
	d := l.toDisp(off-l.start, true)
	ri := l.rowOf(d, false)
	if ri > 0 {
		start := l.start + l.fromDisp(l.rows[ri].from)
		if off != start {
			return start
		}
	}
	a := r.LineStart(line)
	first := a + len(indentOf(l.text))
	if off == first {
		return a
	}
	return first
}

// rowEnd returns the end of the row holding off: the end of the line, or
// of its row when it wraps.
func (ed *editor) rowEnd(off int) int {
	v := &ed.view
	line := v.text.LineOf(off)
	l := v.line(line)
	d := l.toDisp(off-l.start, true)
	ri := l.rowOf(d, false)
	if ri < len(l.rows)-1 {
		end := l.start + l.fromDisp(l.rows[ri].to)
		// Before the space a row ends with.
		if end > off && end > l.start && l.text[end-l.start-1] == ' ' {
			return end - 1
		}
		return end
	}
	return v.text.LineEnd(line)
}

// paragraph returns the start of the next blank line after off going
// down, or before it going up.
func (ed *editor) paragraph(off, dir int) int {
	r := &ed.view.text
	line := r.LineOf(off)
	n := r.LineCount()
	blank := func(i int) bool { return strings.TrimSpace(r.Line(i)) == "" }
	for {
		line += dir
		if line <= 0 {
			return 0
		}
		if line >= n {
			return r.Len()
		}
		if blank(line) && !blank(line-dir) {
			return r.LineStart(line)
		}
	}
}

// scrollBy scrolls by dy DIPs, down for positive.
func (ed *editor) scrollBy(dy float64) {
	ed.setScroll(ed.scrollY + dy)
}

// direction is -1 going back, up or left, and 1 the other way.
func direction(back bool) int {
	if back {
		return -1
	}
	return 1
}

// overFolds moves a caret going from off to to over the lines a fold
// hides rather than into them, which would unfold them: going forward to
// the first line shown after them, back to the end of the line folding
// them, as Zed and VS Code skip a fold.
func (ed *editor) overFolds(off, to int) int {
	r := &ed.view.text
	h := heights{hidden: ed.hiddenSpans()}
	line := r.LineOf(to)
	if !h.isHidden(line) {
		return to
	}
	for _, sp := range h.hidden {
		if line < sp.from || line > sp.to {
			continue
		}
		if to > off && sp.to+1 < r.LineCount() {
			return r.LineStart(sp.to + 1)
		}
		return r.LineEnd(sp.from - 1)
	}
	return to
}
