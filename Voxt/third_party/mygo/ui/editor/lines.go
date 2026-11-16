package editor

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// lineSpanOf returns the lines a selection covers: a selection ending at
// the start of a line leaves that line out, as Zed's.
func (ed *editor) lineSpanOf(s sel) (int, int) {
	r := &ed.view.text
	a, z := r.LineOf(s.start()), r.LineOf(s.end())
	if z > a && s.end() == r.LineStart(z) {
		z--
	}
	return a, z
}

// selLines returns the line spans of the selections, merged.
func (ed *editor) selLines() []lineSpan {
	var spans []lineSpan
	for _, s := range ed.sels {
		a, z := ed.lineSpanOf(s)
		if n := len(spans); n > 0 && a <= spans[n-1].to+1 {
			spans[n-1].to = max(spans[n-1].to, z)
			continue
		}
		spans = append(spans, lineSpan{a, z})
	}
	return spans
}

// deleteLines deletes the lines of the selections.
func (ed *editor) deleteLines() bool {
	r := &ed.view.text
	var edits []Edit
	for _, sp := range ed.selLines() {
		a := r.LineStart(sp.from)
		z := r.LineStart(sp.to + 1)
		if sp.to+1 >= r.LineCount() {
			z = r.Len()
			if sp.from > 0 {
				a = r.LineEnd(sp.from - 1)
			}
		}
		edits = append(edits, Edit{Start: a, End: z})
	}
	// The carets keep their columns on the lines taking the place of
	// those deleted, as Zed's.
	sels := slices.Clone(ed.sels)
	cols := make([]int, len(sels))
	for i, s := range sels {
		cols[i] = utf8.RuneCountInString(r.Slice(r.LineStart(r.LineOf(s.head)), s.head))
	}
	return ed.apply(edits, editOther, func(move func(int, bias) int) []sel {
		t, _ := ed.buf().snapshot()
		for i := range sels {
			at := move(sels[i].head, biasLeft)
			line := t.LineOf(at)
			a, z := t.LineStart(line), t.LineEnd(line)
			sels[i].collapse(a + byteOfRuneIn(t.Slice(a, z), cols[i]))
		}
		return sels
	})
}

// byteOfRuneIn returns the byte of rune n of s, its end past its runes.
func byteOfRuneIn(s string, n int) int {
	for i := range s {
		if n == 0 {
			return i
		}
		n--
	}
	return len(s)
}

// moveLines moves the lines of the selections up or down a line, with
// the selections.
func (ed *editor) moveLines(dir int) bool {
	r := &ed.view.text
	spans := ed.selLines()
	n := r.LineCount()
	if dir < 0 && spans[0].from == 0 || dir > 0 && spans[len(spans)-1].to >= n-1 {
		return false
	}
	var edits []Edit
	type shift struct{ from, to, by int }
	var shifts []shift
	for _, sp := range spans {
		block := r.Slice(r.LineStart(sp.from), r.LineEnd(sp.to))
		if dir < 0 {
			prev := r.Line(sp.from - 1)
			a := r.LineStart(sp.from - 1)
			edits = append(edits, Edit{Start: a, End: r.LineEnd(sp.to), Text: block + "\n" + prev})
			shifts = append(shifts, shift{r.LineStart(sp.from), r.LineEnd(sp.to), -(len(prev) + 1)})
		} else {
			next := r.Line(sp.to + 1)
			a := r.LineStart(sp.from)
			edits = append(edits, Edit{Start: a, End: r.LineEnd(sp.to + 1), Text: next + "\n" + block})
			shifts = append(shifts, shift{r.LineStart(sp.from), r.LineEnd(sp.to), len(next) + 1})
		}
	}
	sels := slices.Clone(ed.sels)
	folds := slices.Clone(ed.folds)
	ok := ed.apply(edits, editOther, func(move func(int, bias) int) []sel {
		for i := range sels {
			s := &sels[i]
			for _, sh := range shifts {
				if s.start() >= sh.from && s.end() <= sh.to+1 {
					s.anchor += sh.by
					s.head += sh.by
					break
				}
			}
		}
		return sels
	})
	if ok {
		// Folds move with their lines.
		for i := range folds {
			for _, sp := range spans {
				if folds[i].start >= sp.from && folds[i].end <= sp.to {
					folds[i].start += dir
					folds[i].end += dir
				}
			}
		}
		ed.folds = folds
	}
	return ok
}

// duplicateLines copies the lines of the selections below them, the
// selections going with the copy, or staying on the lines above it when
// up, as Zed's duplicate line down and up.
func (ed *editor) duplicateLines(up bool) bool {
	r := &ed.view.text
	spans := ed.selLines()
	var edits []Edit
	for _, sp := range spans {
		at := r.LineEnd(sp.to)
		edits = append(edits, Edit{Start: at, End: at, Text: "\n" + r.Slice(r.LineStart(sp.from), at)})
	}
	sels := slices.Clone(ed.sels)
	return ed.apply(edits, editOther, func(move func(int, bias) int) []sel {
		for i := range sels {
			s := &sels[i]
			// A selection of the lines copied goes with the copy below
			// them, or stays above it; the others move with the text.
			a, _ := ed.lineSpanOf(*s)
			in := -1
			for k, sp := range spans {
				if a >= sp.from && a <= sp.to {
					in = k
				}
			}
			if in < 0 {
				s.anchor, s.head = move(s.anchor, biasLeft), move(s.head, biasLeft)
				continue
			}
			if !up {
				sp := spans[in]
				by := r.LineEnd(sp.to) - r.LineStart(sp.from) + 1
				// Moved by the copies of the spans above, and its own.
				before := move(r.LineStart(sp.from), biasLeft) - r.LineStart(sp.from)
				s.anchor, s.head = s.anchor+before+by, s.head+before+by
				continue
			}
			before := move(r.LineStart(spans[in].from), biasLeft) - r.LineStart(spans[in].from)
			s.anchor, s.head = s.anchor+before, s.head+before
		}
		return sels
	})
}

// joinLines joins each selection's lines, or a caret's line with the
// next, with a space, dropping the indentation and comment markers.
func (ed *editor) joinLines() bool {
	r := &ed.view.text
	var edits []Edit
	comment := ""
	if l := ed.language(); l != nil {
		comment = l.LineComment
	}
	for _, sp := range ed.selLines() {
		to := sp.to
		if to == sp.from {
			to++
		}
		if to >= r.LineCount() {
			continue
		}
		for line := sp.from; line < to; line++ {
			end := r.LineEnd(line)
			next := r.Line(line + 1)
			rest := strings.TrimLeft(next, " \t")
			text := r.Line(line)
			cur := strings.TrimRight(text, " \t")
			if comment != "" && strings.HasPrefix(strings.TrimLeft(cur, " \t"), comment) && strings.HasPrefix(rest, comment) {
				rest = strings.TrimLeft(rest[len(comment):], " \t")
			}
			trail := len(text) - len(cur)
			// No space before closing brackets, nor after opening ones.
			sep := " "
			if rest == "" || strings.HasPrefix(rest, ")") || strings.HasPrefix(rest, "]") || strings.HasSuffix(cur, "(") || strings.HasSuffix(cur, "[") {
				sep = ""
			}
			edits = append(edits, Edit{Start: end - trail, End: end + 1 + len(next) - len(rest), Text: sep})
		}
	}
	return ed.apply(edits, editOther, nil)
}

// transpose swaps the characters around each caret, as Emacs's C-t.
func (ed *editor) transpose() bool {
	r := &ed.view.text
	var edits []Edit
	sels := slices.Clone(ed.sels)
	for i, s := range ed.sels {
		if !s.empty() {
			continue
		}
		at := s.head
		line := r.LineOf(at)
		if at == r.LineEnd(line) {
			at = graphemeLeft(*r, at)
		}
		a := graphemeLeft(*r, at)
		z := graphemeRight(*r, at)
		if a >= at || z <= at || a < r.LineStart(line) {
			continue
		}
		if n := len(edits); n > 0 && a < edits[n-1].End {
			// Next to the caret before, whose swap took these characters.
			continue
		}
		edits = append(edits, Edit{Start: a, End: z, Text: r.Slice(at, z) + r.Slice(a, at)})
		sels[i].collapse(z)
	}
	return ed.apply(edits, editOther, func(func(int, bias) int) []sel { return sels })
}

// indentLines indents the lines of the selections a level, or outdents
// them.
func (ed *editor) indentLines(dir int) bool {
	r := &ed.view.text
	unit := ed.indentUnit()
	tab := ed.tabSize()
	var edits []Edit
	for _, sp := range ed.selLines() {
		for line := sp.from; line <= sp.to; line++ {
			s := r.Line(line)
			a := r.LineStart(line)
			if dir > 0 {
				if strings.TrimSpace(s) == "" && sp.from != sp.to {
					continue
				}
				edits = append(edits, Edit{Start: a, End: a, Text: unit})
				continue
			}
			switch {
			case strings.HasPrefix(s, "\t"):
				edits = append(edits, Edit{Start: a, End: a + 1})
			case strings.HasPrefix(s, " "):
				n := len(s) - len(strings.TrimLeft(s, " "))
				n = min(n, (n-1)%tab+1)
				edits = append(edits, Edit{Start: a, End: a + n})
			}
		}
	}
	sels := slices.Clone(ed.sels)
	return ed.apply(edits, editOther, func(move func(int, bias) int) []sel {
		for i := range sels {
			s := &sels[i]
			if s.empty() {
				s.collapse(move(s.head, biasRight))
				continue
			}
			// A selection from a line's start keeps the indentation in it.
			mv := func(off int) int {
				if off == s.start() && off == r.LineStart(r.LineOf(off)) {
					return move(off, biasLeft)
				}
				return move(off, biasRight)
			}
			s.anchor, s.head = mv(s.anchor), mv(s.head)
		}
		return sels
	})
}

// tab inserts a level of indentation, or indents the lines of selections
// across lines.
func (ed *editor) tab() bool {
	r := &ed.view.text
	// A selection indents its lines, as Zed's.
	for _, s := range ed.sels {
		if !s.empty() {
			return ed.indentLines(1)
		}
	}
	tab := ed.tabSize()
	return ed.replaceEach(editTyping, false, func(_ int, s sel) string {
		if ed.hardTabs() {
			return "\t"
		}
		col := columnOf(r.Slice(r.LineStart(r.LineOf(s.start())), s.start()), tab)
		return spaces(tab - col%tab)
	})
}

// columnOf returns the column after s, tabs going to tab stops.
func columnOf(s string, tab int) int {
	col := 0
	for _, c := range s {
		if c == '\t' {
			col += tab - col%tab
		} else {
			col++
		}
	}
	return col
}

// toggleComment comments the lines of the selections out with the
// language's line comment, at the least indentation among them, or
// uncomments them when all are comments, as Zed does.
func (ed *editor) toggleComment() bool {
	l := ed.language()
	if l == nil {
		return false
	}
	if l.LineComment == "" {
		if l.BlockComment[0] != "" {
			return ed.toggleBlockComment()
		}
		return false
	}
	r := &ed.view.text
	prefix := l.LineComment
	var edits []Edit
	spans := ed.selLines()
	for _, sp := range spans {
		minIndent := -1
		all := true
		for line := sp.from; line <= sp.to; line++ {
			s := r.Line(line)
			t := strings.TrimLeft(s, " \t")
			if t == "" {
				continue
			}
			if !strings.HasPrefix(t, prefix) {
				all = false
			}
			if n := len(s) - len(t); minIndent < 0 || n < minIndent {
				minIndent = n
			}
		}
		if minIndent < 0 {
			minIndent = len(indentOf(r.Line(sp.from)))
		}
		for line := sp.from; line <= sp.to; line++ {
			s := r.Line(line)
			a := r.LineStart(line)
			t := strings.TrimLeft(s, " \t")
			if all {
				if t == "" {
					continue
				}
				at := a + len(s) - len(t)
				n := len(prefix)
				if strings.HasPrefix(t[n:], " ") {
					n++
				}
				edits = append(edits, Edit{Start: at, End: at + n})
				continue
			}
			if t == "" && sp.from != sp.to {
				continue
			}
			at := a + min(minIndent, len(s))
			edits = append(edits, Edit{Start: at, End: at, Text: prefix + " "})
		}
	}
	sels := slices.Clone(ed.sels)
	return ed.apply(edits, editOther, func(move func(int, bias) int) []sel {
		for i := range sels {
			s := &sels[i]
			s.anchor, s.head = move(s.anchor, biasRight), move(s.head, biasRight)
		}
		return sels
	})
}

// toggleBlockComment wraps each selection in the language's block
// comment, or unwraps it.
func (ed *editor) toggleBlockComment() bool {
	l := ed.language()
	if l == nil || l.BlockComment[0] == "" {
		return false
	}
	open, close := l.BlockComment[0], l.BlockComment[1]
	r := &ed.view.text
	var edits []Edit
	for _, s := range ed.sels {
		a, z := s.start(), s.end()
		if s.empty() {
			line := r.LineOf(a)
			a, z = r.LineStart(line)+len(indentOf(r.Line(line))), r.LineEnd(line)
		}
		t := r.Slice(a, z)
		if strings.HasPrefix(t, open) && strings.HasSuffix(t, close) && len(t) >= len(open)+len(close) {
			ol, cl := len(open), len(close)
			if strings.HasPrefix(t[ol:], " ") && strings.HasSuffix(t[:len(t)-cl], " ") && len(t) >= ol+cl+2 {
				ol++
				cl++
			}
			edits = append(edits, Edit{Start: a, End: a + ol}, Edit{Start: z - cl, End: z})
			continue
		}
		edits = append(edits, Edit{Start: a, End: a, Text: open + " "}, Edit{Start: z, End: z, Text: " " + close})
	}
	return ed.apply(edits, editOther, nil)
}

// manipulateLines sorts, reverses or dedupes the lines of the
// selections; a caret alone takes all lines.
func (ed *editor) manipulateLines(cmd Command) bool {
	r := &ed.view.text
	spans := ed.selLines()
	if len(spans) == 1 && spans[0].from == spans[0].to {
		spans = []lineSpan{{0, r.LineCount() - 1}}
	}
	var edits []Edit
	for _, sp := range spans {
		a, z := r.LineStart(sp.from), r.LineEnd(sp.to)
		lines := strings.Split(r.Slice(a, z), "\n")
		switch cmd {
		case SortLines:
			slices.Sort(lines)
		case SortLinesCaseInsensitive:
			slices.SortStableFunc(lines, func(x, y string) int { return strings.Compare(strings.ToLower(x), strings.ToLower(y)) })
		case ReverseLines:
			slices.Reverse(lines)
		case UniqueLines:
			seen := map[string]bool{}
			lines = slices.DeleteFunc(lines, func(s string) bool {
				if seen[s] {
					return true
				}
				seen[s] = true
				return false
			})
		}
		edits = append(edits, Edit{Start: a, End: z, Text: strings.Join(lines, "\n")})
	}
	// Selections select their lines as they come out; carets stay.
	sels := slices.Clone(ed.sels)
	return ed.apply(edits, editOther, func(move func(int, bias) int) []sel {
		t, _ := ed.buf().snapshot()
		for i := range sels {
			s := &sels[i]
			if s.empty() {
				s.collapse(min(s.head, t.Len()))
				continue
			}
			for _, e := range edits {
				if s.start() >= e.Start && s.end() <= e.End+1 {
					a := move(e.Start, biasLeft)
					s.anchor, s.head = a, a+len(e.Text)
					if s.end() > e.End {
						s.head = min(t.Len(), s.head+1)
					}
					break
				}
			}
		}
		return sels
	})
}

// convertCase converts the case of each selection, or of the word at a
// caret.
func (ed *editor) convertCase(cmd Command) bool {
	r := &ed.view.text
	extra := ed.wordChars()
	sels := slices.Clone(ed.sels)
	for i := range sels {
		if sels[i].empty() {
			a, z := wordAt(*r, sels[i].head, extra)
			sels[i].anchor, sels[i].head = a, z
		}
	}
	ed.sels = sels
	return ed.replaceEach(editOther, true, func(_ int, s sel) string {
		return convertCase(r.Slice(s.start(), s.end()), cmd)
	})
}

// convertCase converts s: to upper or lower case, title case, or the
// naming conventions of identifiers.
func convertCase(s string, cmd Command) string {
	switch cmd {
	case Uppercase:
		return strings.ToUpper(s)
	case Lowercase:
		return strings.ToLower(s)
	case TitleCase:
		prev := ' '
		return strings.Map(func(r rune) rune {
			defer func() { prev = r }()
			if !isWordRune(prev) {
				return unicode.ToTitle(r)
			}
			return unicode.ToLower(r)
		}, s)
	}
	// Split into words at underscores, hyphens, spaces and case changes.
	var words []string
	var cur []rune
	rs := []rune(s)
	for i, c := range rs {
		switch {
		case c == '_' || c == '-' || unicode.IsSpace(c):
			if len(cur) > 0 {
				words, cur = append(words, string(cur)), nil
			}
			continue
		case unicode.IsUpper(c) && len(cur) > 0 && (unicode.IsLower(cur[len(cur)-1]) || i+1 < len(rs) && unicode.IsLower(rs[i+1]) && unicode.IsUpper(cur[len(cur)-1])):
			words, cur = append(words, string(cur)), nil
		}
		cur = append(cur, c)
	}
	if len(cur) > 0 {
		words = append(words, string(cur))
	}
	for i, w := range words {
		words[i] = strings.ToLower(w)
	}
	title := func(w string) string {
		r, n := utf8.DecodeRuneInString(w)
		return string(unicode.ToTitle(r)) + w[n:]
	}
	switch cmd {
	case SnakeCase:
		return strings.Join(words, "_")
	case KebabCase:
		return strings.Join(words, "-")
	case CamelCase:
		for i := 1; i < len(words); i++ {
			words[i] = title(words[i])
		}
		return strings.Join(words, "")
	case PascalCase:
		for i := range words {
			words[i] = title(words[i])
		}
		return strings.Join(words, "")
	}
	return s
}
