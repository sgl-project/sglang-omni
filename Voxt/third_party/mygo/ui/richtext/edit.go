package richtext

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/egoist/mygo/ui"
)

// Every edit of the editor goes through its document, as one of its steps
// of undo, so that undoing it brings its attributes back with its text.

// lockedPublish gives the document the editor's selection; the caller
// holds its lock. A selection moved forgets the typing attributes set at
// the caret, as in NSTextView.
func (v *view) lockedPublish() {
	d := v.doc
	if d.anchor != v.selAnchor || d.caret != v.caret {
		d.anchor, d.caret = v.selAnchor, v.caret
		d.typing = nil
		d.coalesce = false
	}
}

// publish gives the document the editor's selection.
func (v *view) publish() {
	v.doc.lock()
	defer v.doc.mu.Unlock()
	v.lockedPublish()
}

// record starts an action of the editor: typing joins the step typing
// made last, anything else starts a step of its own.
func (v *view) record(typing bool) {
	if typing && v.coalesce {
		v.mode = typingStep
	} else {
		v.mode = newStep
	}
	v.coalesce = typing
}

// replace replaces the runes from start to end with s in the typing
// attributes, as part of the action's step of undo, and puts the caret
// after it.
func (v *view) replace(start, end int, s string) {
	d := v.doc
	d.lock()
	defer d.mu.Unlock()
	if c, ok := v.lockedSync(); ok {
		start = mapPosition(start, c.start, c.oldEnd-c.start, c.newEnd-c.start, false)
		end = mapPosition(end, c.start, c.oldEnd-c.start, c.newEnd-c.start, true)
	}
	v.lockedPublish()
	attrs := typingAttributes(d.text, start, end)
	if d.typing != nil {
		attrs = *d.typing
	}
	v.lockedEdit(func(e *Edit) { e.ReplaceString(start, end, s, attrs) })
	caret := start + utf8.RuneCountInString(s)
	v.selAnchor, v.caret = caret, caret
	d.anchor, d.caret = caret, caret
	if s == "" && start < end {
		// Typing again where text was deleted gets its attributes, until
		// the caret moves, as in NSTextView.
		keep := attrs
		d.typing = &keep
	}
	v.affinity = ui.TextDownstream
	v.hasDesired = false
	v.reveal = true
}

// lockedEdit runs fn as an edit of the document in the action's step, and
// takes what it changed; the caller holds the lock.
func (v *view) lockedEdit(fn func(e *Edit)) {
	d := v.doc
	before := d.version
	switch v.mode {
	case sameStep:
		if len(d.undo) > 0 {
			s := &d.undo[len(d.undo)-1]
			fn(&Edit{d: d, step: s})
			s.selAfter = [2]int{d.anchor, d.caret}
			clear(d.redo)
			d.redo = d.redo[:0]
			break
		}
		d.edit(false, fn)
	default:
		d.edit(v.mode == typingStep, fn)
		// Typing that starts a step lets the typing after it join it.
		d.coalesce = v.coalesce
	}
	v.mode = sameStep
	if d.version != before {
		v.changed = true
		v.lockedSync()
	}
}

// edit runs fn as a step of undo of its own, as a shortcut's conversion
// does.
func (v *view) edit(fn func(e *Edit)) {
	d := v.doc
	d.lock()
	defer d.mu.Unlock()
	v.lockedSync()
	v.lockedPublish()
	v.mode = newStep
	v.lockedEdit(fn)
	v.mode = newStep
	v.coalesce = false
	v.selAnchor, v.caret = d.anchor, d.caret
	v.selVersion = d.selVersion
	v.reveal = true
}

// command runs a command of the document on the editor's selection.
func (v *view) command(fn func(d *Document)) {
	v.publish()
	before := v.doc.Version()
	fn(v.doc)
	if v.doc.Version() != before {
		v.changed = true
	}
	v.sync()
	v.mode, v.coalesce = newStep, false
}

// takeBack undoes the document's last step, or redoes the last undone.
func (v *view) takeBack(redo bool) {
	v.command(func(d *Document) {
		if redo {
			d.Redo()
		} else {
			d.Undo()
		}
	})
	v.reveal = true
}

// insert replaces the selection with s, as typing does.
func (v *view) insert(s string) {
	if v.opts.ReadOnly {
		return
	}
	s = normalizeNewlines(s)
	a, z := v.selection()
	v.record(a == z && utf8.RuneCountInString(s) == 1 && s != "\n")
	v.replace(a, z, s)
}

// deleteRange deletes the runes from a to z.
func (v *view) deleteRange(a, z int, typing bool) {
	if a == z || v.opts.ReadOnly {
		return
	}
	v.record(typing)
	v.replace(min(a, z), max(a, z), "")
}

// joinStep makes the next edit join the step of the edit before it, unless
// the action started a step no edit made yet: a composition at a caret
// starts one that its commit makes.
func (v *view) joinStep() {
	if v.mode != newStep {
		v.mode = sameStep
	}
}

// commitCompose puts the composition into the text, in the typing
// attributes, as part of the step its start made.
func (v *view) commitCompose() {
	if v.compose != "" {
		s := v.compose
		v.compose = ""
		v.joinStep()
		v.replace(v.caret, v.caret, s)
	}
	v.composeCaret, v.composeSelected = 0, ui.TextInputRange{}
	v.composing = false
}

// special reports whether a paragraph of style st is more than a body
// paragraph, which Backspace at its start makes it before joining.
func special(st ParagraphStyle) bool {
	return st.List != NoList || st.Heading > 0 || st.Quote || st.CodeBlock || st.Indent > 0
}

// unstyle takes away what Backspace at the start of a paragraph of style
// st takes first: a level of indentation of a list's item, the list, a
// heading, a quotation, a block of code.
func unstyle(s *ParagraphStyle) {
	switch {
	case s.List != NoList && s.Indent > 0:
		s.Indent--
	case s.List != NoList:
		s.List, s.Checked = NoList, false
	case s.Heading > 0:
		s.Heading = 0
	case s.Quote:
		s.Quote = false
	case s.CodeBlock:
		s.CodeBlock = false
	case s.Indent > 0:
		s.Indent--
	}
}

// enter breaks the paragraph at the selection: a list's item continues the
// list and an empty one leaves it, a heading ends at its end, and "```"
// starts a block of code.
func (v *view) enter() {
	a, z := v.selection()
	p := v.shown.paraAt(v.caret)
	st := v.para(p).style
	ps := v.shown.paraStart(p)
	pe := ps + v.para(p).n
	if a == z && ps == pe {
		exit := func(s *ParagraphStyle) bool {
			switch {
			case s.List != NoList && s.Indent > 0:
				s.Indent--
			case s.List != NoList:
				s.List, s.Checked = NoList, false
			case s.Quote:
				s.Quote = false
			case s.CodeBlock && p > 0 && v.para(p-1).style.CodeBlock:
				s.CodeBlock = false
			default:
				return false
			}
			return true
		}
		probe := st
		if exit(&probe) {
			v.command(func(d *Document) { d.FormatParagraphs(func(s *ParagraphStyle) { exit(s) }) })
			return
		}
	}
	if !v.opts.NoMarkdown && a == z && v.caret == pe && !st.CodeBlock {
		if line := v.lineOf(p); strings.HasPrefix(line, "```") && !strings.Contains(line[3:], "`") {
			v.edit(func(e *Edit) {
				e.ReplaceString(ps, pe, "", Attributes{})
				e.FormatParagraphs(ps, ps, func(s *ParagraphStyle) { *s = ParagraphStyle{CodeBlock: true, Indent: s.Indent} })
				e.Select(ps, ps)
			})
			return
		}
	}
	atEnd := v.caret == pe && a == z
	atStart := a == z && a == ps && ps < pe
	v.doc.lock()
	v.lockedPublish()
	attrs := v.doc.typingAttributes()
	v.doc.mu.Unlock()
	v.record(false)
	v.replace(a, z, "\n")
	d := v.doc
	d.lock()
	v.lockedEdit(func(e *Edit) {
		// The new paragraph: after the caret's, or before it at its start,
		// as Notes keeps the item it moves down checked.
		at := v.caret
		if atStart {
			at = v.caret - 1
		}
		e.FormatParagraphs(at, at, func(s *ParagraphStyle) {
			if atEnd && s.Heading > 0 {
				s.Heading = 0
			}
			s.Checked = false
		})
	})
	d.anchor, d.caret = v.caret, v.caret
	keep := attrs
	d.typing = &keep
	d.mu.Unlock()
	v.coalesce = false
}

// typed runs the Markdown shortcuts after s was typed at the caret.
func (v *view) typed(s string) {
	if v.opts.NoMarkdown || v.opts.ReadOnly || utf8.RuneCountInString(s) != 1 {
		return
	}
	p := v.shown.paraAt(v.caret)
	ps := v.shown.paraStart(p)
	if v.para(p).style.CodeBlock {
		return
	}
	before := []rune(v.shown.Slice(ps, v.caret).String())
	if len(before) == 0 {
		return
	}
	if s == " " {
		v.blockShortcut(ps, string(before[:len(before)-1]))
		return
	}
	switch s {
	case "*", "_", "`", "~":
		v.inlineShortcut(ps, before)
	}
}

// blockShortcut turns what was typed before a space at the start of a
// paragraph into its style: "# " a heading, "- " a list, "> " a quotation.
func (v *view) blockShortcut(ps int, prefix string) {
	var set func(s *ParagraphStyle)
	switch {
	case prefix != "" && strings.Trim(prefix, "#") == "" && len(prefix) <= 6:
		level := len(prefix)
		set = func(s *ParagraphStyle) { s.Heading, s.List, s.Checked = level, NoList, false }
	case prefix == "-" || prefix == "*" || prefix == "+":
		set = func(s *ParagraphStyle) { s.List, s.Heading = Bullets, 0 }
	case prefix == "[]" || prefix == "[ ]" || prefix == "- [ ]" || prefix == "-[ ]":
		set = func(s *ParagraphStyle) { s.List, s.Heading, s.Checked = Tasks, 0, false }
	case prefix == "[x]" || prefix == "- [x]" || prefix == "[X]":
		set = func(s *ParagraphStyle) { s.List, s.Heading, s.Checked = Tasks, 0, true }
	case prefix == ">":
		set = func(s *ParagraphStyle) { s.Quote = true }
	case len(prefix) >= 2 && (prefix[len(prefix)-1] == '.' || prefix[len(prefix)-1] == ')') && strings.Trim(prefix[:len(prefix)-1], "0123456789") == "":
		set = func(s *ParagraphStyle) { s.List, s.Heading = Numbers, 0 }
	}
	if set == nil {
		return
	}
	n := utf8.RuneCountInString(prefix) + 1
	v.edit(func(e *Edit) {
		e.ReplaceString(ps, ps+n, "", Attributes{})
		e.FormatParagraphs(ps, ps, set)
		e.Select(ps, ps)
	})
}

// inlineShortcut formats what a pair of marks typed around words holds:
// **bold**, *italic*, _italic_, `code` and ~~strikethrough~~.
func (v *view) inlineShortcut(ps int, before []rune) {
	n := len(before)
	last := before[n-1]
	mark := string(last)
	if (last == '*' || last == '~' || last == '_') && n >= 2 && before[n-2] == last {
		mark += mark
	}
	if last == '~' && len(mark) == 1 {
		return
	}
	m := len([]rune(mark))
	closeAt := n - m
	if closeAt <= 0 || unicode.IsSpace(before[closeAt-1]) {
		return
	}
	open := -1
	for i := closeAt - 1; i >= 0; i-- {
		if string(before[i:min(i+m, n)]) == mark && i+m < closeAt {
			// Not the inside of a longer mark, and words right after it.
			if (i > 0 && before[i-1] == last) || before[i+m] == last || unicode.IsSpace(before[i+m]) {
				continue
			}
			// An underscore inside a word, as snake_case, is no mark.
			if last == '_' && i > 0 && (unicode.IsLetter(before[i-1]) || unicode.IsDigit(before[i-1])) {
				continue
			}
			open = i
			break
		}
	}
	if open < 0 {
		return
	}
	set := func(a *Attributes) {
		switch mark {
		case "**", "__":
			a.Weight = 700
		case "*", "_":
			a.Italic = true
		case "`":
			a.Code = true
		case "~~":
			a.Strikethrough = true
		}
	}
	start, end := ps+open, ps+closeAt
	v.edit(func(e *Edit) {
		e.ReplaceString(end, end+m, "", Attributes{})
		e.ReplaceString(start, start+m, "", Attributes{})
		e.Format(start, end-m, set)
		e.Select(end-m, end-m)
	})
	// What is typed next is outside the format, as after a closing mark.
	v.doc.lock()
	plain := Attributes{}
	if start > ps {
		plain = v.doc.text.AttributesAt(start - 1).extending()
	}
	v.doc.typing = &plain
	v.doc.mu.Unlock()
}

// looksLikeURL reports whether s is a URL, as ⌘K links one to itself.
func looksLikeURL(s string) bool {
	if strings.ContainsAny(s, " \n\t") {
		return false
	}
	for _, prefix := range []string{"http://", "https://", "mailto:", "www."} {
		if strings.HasPrefix(s, prefix) {
			return true
		}
	}
	return false
}

// run runs a command of the Format menu or of a shortcut, and reports
// whether there is one of that name.
func (v *view) run(cmd string) bool {
	var fn func(d *Document)
	switch cmd {
	case "bold":
		fn = (*Document).ToggleBold
	case "italic":
		fn = (*Document).ToggleItalic
	case "underline":
		fn = (*Document).ToggleUnderline
	case "strikethrough":
		fn = (*Document).ToggleStrikethrough
	case "code":
		fn = (*Document).ToggleCode
	case "bullets":
		fn = func(d *Document) { d.ToggleList(Bullets) }
	case "numbers":
		fn = func(d *Document) { d.ToggleList(Numbers) }
	case "tasks":
		fn = func(d *Document) { d.ToggleList(Tasks) }
	case "indent":
		fn = func(d *Document) { d.Indent(1) }
	case "outdent":
		fn = func(d *Document) { d.Indent(-1) }
	case "alignLeft":
		fn = func(d *Document) { d.SetAlign(ui.Start) }
	case "alignCenter":
		fn = func(d *Document) { d.SetAlign(ui.Center) }
	case "alignRight":
		fn = func(d *Document) { d.SetAlign(ui.End) }
	case "quote":
		fn = (*Document).ToggleQuote
	case "codeBlock":
		fn = (*Document).ToggleCodeBlock
	case "link":
		if v.opts.ReadOnly {
			return false
		}
		if v.opts.OnLinkCommand != nil {
			v.publish()
			v.opts.OnLinkCommand()
			return true
		}
		a, z := v.selection()
		url := strings.TrimSpace(v.shown.Slice(a, z).String())
		if a == z || !looksLikeURL(url) {
			return true
		}
		fn = func(d *Document) { d.SetLink(url) }
	default:
		if level, ok := strings.CutPrefix(cmd, "heading"); ok && len(level) == 1 && level[0] >= '0' && level[0] <= '6' {
			fn = func(d *Document) { d.SetHeading(int(level[0] - '0')) }
		}
	}
	if fn == nil {
		return false
	}
	if v.opts.ReadOnly {
		return true
	}
	v.command(fn)
	return true
}
