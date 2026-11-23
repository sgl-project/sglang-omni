package editor

import (
	"slices"
	"strings"
)

// Folding, as Zed falls back to without a syntax tree and VS Code does:
// a line ending with an opening bracket folds the lines up to the one
// with its closing bracket, which stays; another folds the lines
// indented more than it after it, but blank lines at the end.

// foldEnd returns the last line folding line hides, and whether line
// starts a fold.
func (ed *editor) foldEnd(line int) (int, bool) {
	if f, ok := ed.highlighter().(Folder); ok {
		return f.FoldEnd(line)
	}
	r := &ed.view.text
	n := r.LineCount()
	if line >= n-1 {
		return 0, false
	}
	s := r.Line(line)
	if strings.TrimSpace(s) == "" {
		return 0, false
	}
	// An opening bracket at the end of the line.
	if open, ok := ed.trailingOpener(line, s); ok {
		if _, close, ok := ed.matchBracket(r.LineStart(line) + open); ok {
			cl := r.LineOf(close)
			if cl > line+1 {
				return cl - 1, true
			}
			if cl == line+1 {
				return 0, false
			}
		}
	}
	tab := ed.tabSize()
	in := indentWidth(s, tab)
	end := line
	for i := line + 1; i < n; i++ {
		t := r.Line(i)
		if strings.TrimSpace(t) == "" {
			continue
		}
		if indentWidth(t, tab) <= in {
			break
		}
		end = i
	}
	if end > line {
		return end, true
	}
	return 0, false
}

// foldable reports whether line starts a fold, cheaply, for the gutter:
// by indentation, or an opening bracket at its end.
func (ed *editor) foldable(line int) bool {
	if f, ok := ed.view.hl.(Folder); ok {
		_, ok := f.FoldEnd(line)
		return ok
	}
	r := &ed.view.text
	n := r.LineCount()
	if line >= n-1 {
		return false
	}
	s := r.Line(line)
	if strings.TrimSpace(s) == "" {
		return false
	}
	tab := ed.tabSize()
	in := indentWidth(s, tab)
	for i := line + 1; i < min(n, line+50); i++ {
		t := r.Line(i)
		if strings.TrimSpace(t) == "" {
			continue
		}
		return indentWidth(t, tab) > in
	}
	return false
}

// trailingOpener returns the offset in s of an opening bracket ending it,
// before spaces and a comment.
func (ed *editor) trailingOpener(line int, s string) (int, bool) {
	l := ed.language()
	if l == nil {
		return 0, false
	}
	t := strings.TrimRight(s, " \t")
	if l.LineComment != "" {
		if i := strings.LastIndex(t, l.LineComment); i >= 0 {
			toks := ed.view.lineTokens(line, s)
			if kindAt(toks, i) == TokenComment {
				t = strings.TrimRight(t[:i], " \t")
			}
		}
	}
	if t == "" {
		return 0, false
	}
	c := t[len(t)-1]
	for _, p := range l.Brackets {
		if len(p.Open) == 1 && p.Open[0] == c && c != '<' {
			return len(t) - 1, true
		}
	}
	return 0, false
}

// kindAt returns the kind of the token holding byte i of a line.
func kindAt(toks []Token, i int) TokenKind {
	for _, t := range toks {
		if i < t.End {
			return t.Kind
		}
	}
	return TokenText
}

// isFolded reports whether a fold starts at line.
func (ed *editor) isFolded(line int) bool {
	return slices.ContainsFunc(ed.folds, func(f fold) bool { return f.start == line })
}

// foldAt folds the innermost fold holding the line of off.
func (ed *editor) foldAt(off int) bool {
	r := &ed.view.text
	line := r.LineOf(off)
	for l := line; l >= 0 && line-l < 10000; l-- {
		if ed.isFolded(l) && l != line {
			continue
		}
		if end, ok := ed.foldEnd(l); ok && end >= line && !ed.isFolded(l) {
			ed.folds = append(ed.folds, fold{l, end})
			ed.afterFold(l)
			return true
		}
	}
	return false
}

// afterFold moves carets out of what a fold hides.
func (ed *editor) afterFold(line int) {
	r := &ed.view.text
	ed.needsBuild = true
	hidden := ed.hiddenSpans()
	h := heights{hidden: hidden}
	sels := slices.Clone(ed.sels)
	moved := false
	for i := range sels {
		s := &sels[i]
		if l := r.LineOf(s.head); h.isHidden(l) {
			s.collapse(r.LineEnd(line))
			moved = true
		}
	}
	if moved {
		ed.sels = mergeSels(sels)
	}
}

// unfoldAt unfolds the folds starting at the line of off or holding it,
// and those inside them when recursive.
func (ed *editor) unfoldAt(off int, recursive bool) bool {
	line := ed.view.text.LineOf(off)
	before := len(ed.folds)
	var outer fold
	for _, f := range ed.folds {
		if f.start == line || line > f.start && line <= f.end {
			if f.end-f.start > outer.end-outer.start {
				outer = f
			}
		}
	}
	ed.folds = slices.DeleteFunc(ed.folds, func(f fold) bool {
		if f.start == line || line > f.start && line <= f.end {
			return true
		}
		return recursive && outer.end > outer.start && f.start >= outer.start && f.end <= outer.end
	})
	ed.needsBuild = true
	return len(ed.folds) != before
}

// foldRanges returns every fold of the text, with how deep each is.
func (ed *editor) foldRanges() (folds []fold, depths []int) {
	r := &ed.view.text
	var stack []fold
	for line := 0; line < r.LineCount()-1; line++ {
		for len(stack) > 0 && stack[len(stack)-1].end < line {
			stack = stack[:len(stack)-1]
		}
		if !ed.foldable(line) {
			if _, ok := ed.trailingOpener(line, r.Line(line)); !ok {
				continue
			}
		}
		if end, ok := ed.foldEnd(line); ok {
			f := fold{line, end}
			folds = append(folds, f)
			depths = append(depths, len(stack)+1)
			stack = append(stack, f)
		}
	}
	return folds, depths
}

// foldLevel folds the folds at a depth, all of them for 0.
func (ed *editor) foldLevel(level int) bool {
	folds, depths := ed.foldRanges()
	ed.folds = ed.folds[:0]
	for i, f := range folds {
		if level == 0 || depths[i] == level {
			ed.folds = append(ed.folds, f)
		}
	}
	ed.needsBuild = true
	ed.moveOutOfFolds()
	return true
}

// foldRecursive folds the fold at off and those inside it.
func (ed *editor) foldRecursive(off int) bool {
	if !ed.foldAt(off) {
		return false
	}
	outer := ed.folds[len(ed.folds)-1]
	folds, _ := ed.foldRanges()
	for _, f := range folds {
		if f.start > outer.start && f.end <= outer.end && !ed.isFolded(f.start) {
			ed.folds = append(ed.folds, f)
		}
	}
	return true
}

// moveOutOfFolds puts carets in folded lines at the end of the line
// folding them.
func (ed *editor) moveOutOfFolds() {
	r := &ed.view.text
	h := heights{hidden: ed.hiddenSpans()}
	sels := slices.Clone(ed.sels)
	for i := range sels {
		s := &sels[i]
		l := r.LineOf(s.head)
		for l > 0 && h.isHidden(l) {
			l--
		}
		if l != r.LineOf(s.head) {
			s.collapse(r.LineEnd(l))
		}
	}
	ed.sels = mergeSels(sels)
}

// toggleFoldLine folds or unfolds the fold starting at line, as the
// gutter's marker does.
func (ed *editor) toggleFoldLine(line int) {
	if ed.isFolded(line) {
		ed.folds = slices.DeleteFunc(ed.folds, func(f fold) bool { return f.start == line })
		ed.needsBuild = true
		return
	}
	if end, ok := ed.foldEnd(line); ok {
		ed.folds = append(ed.folds, fold{line, end})
		ed.moveOutOfFolds()
		ed.needsBuild = true
	}
}
