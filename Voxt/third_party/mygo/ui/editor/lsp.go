package editor

import (
	"slices"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/egoist/mygo/ui"
)

// What a language server gives an editor, as Zed shows it: completions,
// hovers, diagnostics and inlay hints, which the app asks its own
// server for, or makes itself.

// CompletionRequest asks for the completions at Offset of the text
// at Version: Prefix is the word before the caret, which the editor
// filters the completions it is given by as typing goes on.
type CompletionRequest struct {
	Offset   int
	Prefix   string
	Version  uint64
	Explicit bool // asked for with ui.Ctrl+Space, rather than by typing
}

// Completion is a completion the editor offers.
type Completion struct {
	// Label shows in the list, and is what the prefix typed matches.
	Label string
	// Detail shows dimmed after it, as a type or a signature.
	Detail string
	Kind   CompletionKind
	// Insert replaces the word before the caret, Label when empty.
	Insert string
	// Documentation shows beside the list for the completion chosen.
	Documentation string
	// FilterText is matched instead of Label when set.
	FilterText string
}

// CompletionKind is what a completion is, which its icon shows.
type CompletionKind uint8

const (
	CompletionText CompletionKind = iota
	CompletionKeyword
	CompletionFunction
	CompletionMethod
	CompletionVariable
	CompletionField
	CompletionType
	CompletionModule
	CompletionConstant
	CompletionSnippet
)

// HoverRequest asks what to show for the text at Offset under the
// pointer, or at the caret.
type HoverRequest struct {
	Offset  int
	Version uint64
}

// Diagnostic is a problem with a range of the text, Start to End,
// which shows underlined, in the hover and the scroll bar.
type Diagnostic struct {
	Start, End int
	Severity   Severity
	Message    string
	Source     string
}

// Severity is how bad a diagnostic is.
type Severity uint8

const (
	SeverityError Severity = iota
	SeverityWarning
	SeverityInfo
	SeverityHint
)

// Inlay is text shown in a line that is not part of it, as an inlay
// hint of a type or a parameter's name: at Offset, muted.
type Inlay struct {
	Offset int
	Text   string
}

// completions is the list of completions the editor shows.
type completions struct {
	shown    bool
	at       int // where the word being completed starts
	items    []Completion
	filtered []int
	scores   []int
	selected int
	prefix   string
	version  uint64
	request  bool // a request is out
	scrollTo bool
}

func (cc *completions) open() bool { return cc.shown && len(cc.filtered) > 0 }

func (cc *completions) close() {
	cc.shown = false
	cc.items, cc.filtered = nil, nil
	cc.request = false
}

func (cc *completions) step(d int) {
	if n := len(cc.filtered); n > 0 {
		cc.selected = (cc.selected + d + n) % n
		cc.scrollTo = true
	}
}

func (cc *completions) moved(cs []change) {
	cc.at = mapOffset(cc.at, cs, biasLeft)
}

// wordStartBefore returns where the word before off starts.
func (ed *editor) wordStartBefore(off int) int {
	r := &ed.view.text
	extra := ed.wordChars()
	a := off
	for a > 0 {
		c, size := r.RuneBefore(a)
		if classOf(c, extra) != classWord {
			break
		}
		a -= size
	}
	return a
}

// requestCompletions asks the app for completions at the newest caret.
func (ed *editor) requestCompletions(explicit bool) bool {
	if ed.o.onCompletion == nil || ed.o.readOnly {
		return false
	}
	p := ed.primary()
	if !p.empty() {
		return false
	}
	start := ed.wordStartBefore(p.head)
	prefix := ed.slice(start, p.head)
	if !explicit && prefix == "" {
		// After a trigger character, as a dot: a new list.
		ed.complete.close()
	}
	if ed.complete.shown && ed.complete.at == start && len(ed.complete.items) > 0 {
		ed.complete.filter(prefix)
		if len(ed.complete.filtered) > 0 || !explicit {
			if len(ed.complete.filtered) == 0 {
				ed.complete.close()
			}
			return true
		}
	}
	ed.complete.at = start
	ed.complete.request = true
	ed.complete.version = ed.view.version
	ed.o.onCompletion(CompletionRequest{Offset: p.head, Prefix: prefix, Version: ed.view.version, Explicit: explicit})
	return true
}

// showCompletions shows items for the word starting at at.
func (ed *editor) showCompletions(at int, items []Completion) {
	cc := &ed.complete
	p := ed.primary()
	if len(items) == 0 || !p.empty() {
		cc.close()
		return
	}
	start := ed.wordStartBefore(p.head)
	if at >= 0 && at <= p.head {
		start = at
	}
	cc.shown, cc.at, cc.items, cc.request = true, start, items, false
	cc.filter(ed.slice(start, p.head))
	if len(cc.filtered) == 0 {
		cc.shown = false
	}
	ed.needsBuild = true
}

// filter keeps the items matching prefix, best first, as Zed's fuzzy
// matching: the prefix's characters in order, each match worth more at
// the start of the label, of a word in it or right after the one
// before, and less the later it comes.
func (cc *completions) filter(prefix string) {
	cc.prefix = prefix
	cc.filtered, cc.scores = cc.filtered[:0], cc.scores[:0]
	for i, it := range cc.items {
		text := it.FilterText
		if text == "" {
			text = it.Label
		}
		if s, ok := fuzzyScore(text, prefix); ok {
			cc.filtered = append(cc.filtered, i)
			cc.scores = append(cc.scores, s)
		}
	}
	idx := make([]int, len(cc.filtered))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return cc.scores[idx[a]] > cc.scores[idx[b]] })
	filtered := make([]int, len(idx))
	scores := make([]int, len(idx))
	for i, j := range idx {
		filtered[i], scores[i] = cc.filtered[j], cc.scores[j]
	}
	cc.filtered, cc.scores = filtered, scores
	cc.selected = 0
	cc.scrollTo = true
}

// fuzzyScore scores how well query matches text, by subsequence.
func fuzzyScore(text, query string) (int, bool) {
	if query == "" {
		return 0, true
	}
	score := 0
	ti := 0
	prevMatch := -2
	tr := []rune(text)
	for _, q := range query {
		ql := unicode.ToLower(q)
		found := false
		for ti < len(tr) {
			c := tr[ti]
			if unicode.ToLower(c) == ql {
				pts := 1
				switch {
				case ti == 0:
					pts += 8
				case ti == prevMatch+1:
					pts += 5
				case !isWordRune(tr[ti-1]) || unicode.IsUpper(c) && unicode.IsLower(tr[ti-1]):
					pts += 4
				}
				if c == q {
					pts++
				}
				score += pts*16 - ti
				prevMatch = ti
				ti++
				found = true
				break
			}
			ti++
		}
		if !found {
			return 0, false
		}
	}
	// Shorter labels win ties.
	score -= len(tr) - utf8.RuneCountInString(query)
	return score, true
}

// acceptCompletion inserts the completion chosen in place of the word
// typed, at every caret.
func (ed *editor) acceptCompletion() bool {
	cc := &ed.complete
	if !cc.open() {
		return false
	}
	it := cc.items[cc.filtered[cc.selected]]
	ins := it.Insert
	if ins == "" {
		ins = it.Label
	}
	p := ed.primary()
	n := p.head - cc.at
	typed := ed.slice(cc.at, p.head)
	cc.close()
	// The word typed goes where the other carets have it before them too,
	// as Zed's; elsewhere the completion goes in at the caret.
	var edits []Edit
	for _, s := range ed.sels {
		a := s.start()
		if s.head-n >= 0 && ed.slice(s.head-n, s.head) == typed {
			a = s.head - n
		}
		edits = append(edits, Edit{Start: a, End: s.end(), Text: ins})
	}
	sels := slices.Clone(ed.sels)
	ed.apply(edits, editOther, func(move func(int, bias) int) []sel {
		for i := range sels {
			at := move(sels[i].end(), biasRight)
			sels[i].collapse(at)
		}
		return sels
	})
	return true
}

// hover is the popover of what the text under the pointer is, and
// the diagnostics there.
type hover struct {
	shown   bool
	start   int
	end     int
	build   func(c *ui.Context)
	text    string
	asked   int // the offset a request was made for
	restAt  time.Time
	restOff int
	pending bool
	diags   []Diagnostic
	over    bool // the pointer is over the popover
	// byPointer marks a hover the resting pointer asked for, which goes
	// as the pointer leaves, rather than the keyboard.
	byPointer bool
	// hideAt is when the hover goes, the pointer having left the text
	// hovered, unless it goes into the hover first; focused reports the
	// keyboard in the hover, as its text is selected and copied, which
	// keeps it.
	hideAt  time.Time
	focused bool
}

// hoverHideDelay is how long a hover stays once the pointer has left the
// text hovered, for the pointer to go into it, as VS Code's
// editor.hover.hidingDelay.
const hoverHideDelay = 300 * time.Millisecond

func (h *hover) open() bool { return h.shown }

func (h *hover) close() {
	h.shown, h.build, h.text, h.diags, h.byPointer = false, nil, "", nil, false
	h.hideAt, h.focused = time.Time{}, false
	h.asked = -1
	h.pending = false
}

// closeHover closes the hover, and gives the keyboard back to the editor
// when the hover had it.
func (ed *editor) closeHover() {
	if ed.hover.focused {
		ed.refocus = true
	}
	ed.hover.close()
}

// requestHover asks the app about the text at off.
func (ed *editor) requestHover(off int) bool {
	ed.hover.diags = ed.diagnosticsAt(off)
	if ed.o.onHover != nil && ed.hover.asked != off {
		ed.hover.asked = off
		ed.o.onHover(HoverRequest{Offset: off, Version: ed.view.version})
	}
	if len(ed.hover.diags) > 0 {
		ed.hover.shown = true
		ed.hover.start, ed.hover.end = ed.hover.diags[0].Start, ed.hover.diags[0].End
		ed.needsBuild = true
	}
	return ed.o.onHover != nil || len(ed.hover.diags) > 0
}

// diagnosticsAt returns the diagnostics holding off.
func (ed *editor) diagnosticsAt(off int) []Diagnostic {
	var out []Diagnostic
	for _, d := range ed.diags {
		if off >= d.Start && off <= max(d.End, d.Start+1) {
			out = append(out, d)
		}
	}
	return out
}

// SeverityColor returns the color of diagnostics of a severity.
func (ct *Theme) SeverityColor(s Severity) ui.Color {
	switch s {
	case SeverityWarning:
		return ct.Warning
	case SeverityInfo:
		return ct.Info
	case SeverityHint:
		return ct.Hint
	}
	return ct.Error
}

// acceptSuggestion inserts the suggestion shown at the caret.
func (ed *editor) acceptSuggestion() bool {
	for i, in := range ed.view.inlays {
		if in.kind == 1 && len(ed.sels) == 1 && ed.sels[0].empty() && ed.sels[0].head == in.at {
			text := ed.suggestion
			ed.view.inlays = slices.Delete(ed.view.inlays, i, i+1)
			ed.suggestion = ""
			ed.replaceEach(editOther, false, func(int, sel) string { return text })
			return true
		}
	}
	return false
}

// dropSuggestion takes the suggestion away, as an edit or a move of the
// caret does.
func (ed *editor) dropSuggestion() {
	if ed.suggestion == "" {
		return
	}
	ed.setSuggestion(0, "")
}

// setSuggestion shows text at off, which Tab inserts.
func (ed *editor) setSuggestion(off int, text string) {
	ed.view.inlays = slices.DeleteFunc(ed.view.inlays, func(in inlay) bool { return in.kind == 1 })
	ed.suggestion = text
	if text != "" {
		first, _, _ := strings.Cut(text, "\n")
		if first != text {
			first += " …"
		}
		ed.view.inlays = append(ed.view.inlays, inlay{at: off, text: first, kind: 1})
		sortInlays(ed.view.inlays)
	}
	ed.view.lines.m = nil
}

func sortInlays(in []inlay) {
	slices.SortStableFunc(in, func(a, b inlay) int { return a.at - b.at })
}

// setInlays shows the inlay hints.
func (ed *editor) setInlays(hints []Inlay) {
	ed.view.inlays = slices.DeleteFunc(ed.view.inlays, func(in inlay) bool { return in.kind == 0 })
	for _, h := range hints {
		ed.view.inlays = append(ed.view.inlays, inlay{at: h.Offset, text: h.Text})
	}
	sortInlays(ed.view.inlays)
	ed.view.lines.m = nil
}
