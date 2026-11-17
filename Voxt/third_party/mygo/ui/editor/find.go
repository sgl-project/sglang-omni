package editor

import (
	"github.com/egoist/mygo/internal/l10n"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// finder is an editor's find bar and its matches, as Zed's buffer
// search: a query, case sensitive or not, whole words, a regular
// expression, in the selections or everywhere, and its replacement.
type finder struct {
	open        bool
	replacing   bool
	query       string
	replace     string
	caseSens    bool
	wholeWord   bool
	regex       bool
	inSelection bool
	scope       []sel
	focusQuery  bool
	typed       string // the query as the matches were last chosen for
	// The matches of the last search, and the text and options it was of;
	// subs are a regular expression's submatches, as found in the text.
	matches  []diffRange
	subs     [][]int
	dirty    []lineSpan // lines edits changed, to search again
	current  int
	stale    bool
	version  uint64
	searched string
	options  [4]bool
	err      error
	lower    string // the text lowered, for searching without case
	lowerVer uint64
	scrollTo bool
}

// maxMatches bounds the matches a search finds.
const maxMatches = 100000

func (f *finder) show(ed *editor, replace bool) {
	f.open = true
	f.replacing = f.replacing || replace
	if replace {
		f.replacing = true
	}
	// The selection, or the word at the caret, seeds the query.
	p := ed.primary()
	if !p.empty() && ed.view.text.LineOf(p.start()) == ed.view.text.LineOf(p.end()) {
		f.query = ed.slice(p.start(), p.end())
	} else if p.empty() {
		if a, z := wordAt(ed.view.text, p.head, ed.wordChars()); z > a && f.query == "" {
			f.query = ed.slice(a, z)
		}
	}
	if f.inSelection {
		f.scope = nil
	}
	f.focusQuery = true
	f.stale = true
	f.scrollTo = true
	ed.needsBuild = true
}

// update searches again when the text, the query or the options
// changed, and selects the match at or after the caret as the query is
// typed, as Zed's search does.
func (f *finder) update(ed *editor) {
	if f.query != f.typed {
		f.typed = f.query
		f.scrollTo = true
	}
	f.search(ed)
	if !f.scrollTo || len(f.matches) == 0 {
		return
	}
	f.scrollTo = false
	f.current = f.nearest(ed.primary().start())
	m := f.matches[f.current]
	// In the middle of the view when out of it, either way, and out of a
	// fold.
	ed.setSels([]sel{ed.newSel(m.start, m.end)}, revealCenter)
	if ed.o.sharedScroll == nil {
		ed.doReveal()
	}
}

// search finds the matches of the query in the text again when either
// changed: after edits, a plain query only in the lines they changed,
// the others' matches having moved with them (moved).
func (f *finder) search(ed *editor) {
	opts := [4]bool{f.caseSens, f.wholeWord, f.regex, f.inSelection}
	if !f.stale && f.version == ed.view.version && f.searched == f.query && f.options == opts {
		if len(f.dirty) > 0 {
			f.searchLines(ed)
		}
		return
	}
	if opts[3] != f.options[3] {
		// Finding in the selection takes the selections there are now.
		f.scope = nil
	}
	f.stale, f.dirty = false, nil
	f.version, f.searched, f.options = ed.view.version, f.query, opts
	f.matches, f.subs = f.matches[:0], f.subs[:0]
	f.err = nil
	if f.query == "" {
		return
	}
	if f.inSelection && f.scope == nil {
		for _, s := range ed.sels {
			if !s.empty() {
				f.scope = append(f.scope, s)
			}
		}
	}
	text := ed.view.text.String()
	if !f.regex && !f.caseSens && isASCII(f.query) {
		if f.lowerVer != ed.view.version || f.lower == "" && text != "" {
			f.lower, f.lowerVer = asciiLower(text), ed.view.version
		}
	}
	f.scan(ed, text, f.lower, 0, func(m diffRange, sub []int) bool {
		f.matches = append(f.matches, m)
		if sub != nil {
			f.subs = append(f.subs, sub)
		}
		return len(f.matches) < maxMatches
	})
	f.current = min(f.current, max(0, len(f.matches)-1))
}

// scan finds the query in text, which starts at byte base of the editor's
// text (lower its lowered ASCII, when known), and calls found with each
// match it keeps, and a regular expression's submatches, until found
// returns false.
func (f *finder) scan(ed *editor, text, lower string, base int, found func(m diffRange, sub []int) bool) {
	extra := ed.wordChars()
	keep := func(a, b int) bool {
		if f.wholeWord && !isWholeWord(text, a, b, extra) {
			return false
		}
		if f.inSelection && len(f.scope) > 0 {
			for _, s := range f.scope {
				if base+a >= s.start() && base+b <= s.end() {
					return true
				}
			}
			return false
		}
		return true
	}
	switch {
	case f.regex:
		pat := f.query
		if !f.caseSens {
			pat = "(?i)" + pat
		}
		re, err := regexp.Compile("(?m)" + pat)
		if err != nil {
			f.err = err
			return
		}
		for _, m := range re.FindAllStringSubmatchIndex(text, maxMatches) {
			if m[1] == m[0] || !keep(m[0], m[1]) {
				continue
			}
			if !found(diffRange{base + m[0], base + m[1]}, m) {
				return
			}
		}
	case f.caseSens || isASCII(f.query):
		q, in := f.query, text
		if !f.caseSens {
			if lower == "" {
				lower = asciiLower(text)
			}
			q, in = asciiLower(f.query), lower
		}
		for at := 0; ; {
			i := strings.Index(in[at:], q)
			if i < 0 {
				return
			}
			a, b := at+i, at+i+len(q)
			if keep(a, b) && !found(diffRange{base + a, base + b}, nil) {
				return
			}
			at = b
		}
	default:
		re := regexp.MustCompile("(?i)" + regexp.QuoteMeta(f.query))
		for _, m := range re.FindAllStringIndex(text, maxMatches) {
			if keep(m[0], m[1]) && !found(diffRange{base + m[0], base + m[1]}, nil) {
				return
			}
		}
	}
}

// moved follows edits of the text the matches were found in: those an
// edit touches go, the others move with the text, and the lines it
// changed are searched again (searchLines). A regular expression, whose
// matches can span lines, or a query of several lines, searches again.
func (f *finder) moved(cs []change) {
	if f.stale || f.regex || f.query == "" || strings.Contains(f.query, "\n") {
		f.stale = true
		return
	}
	out := f.matches[:0]
	for _, m := range f.matches {
		touched := false
		for _, c := range cs {
			if m.start <= c.oldEnd && m.end >= c.oldStart {
				touched = true
				break
			}
		}
		if !touched {
			out = append(out, diffRange{mapOffset(m.start, cs, biasRight), mapOffset(m.end, cs, biasLeft)})
		}
	}
	f.matches = out
	for i := range f.dirty {
		f.dirty[i] = lineSpan{mapLine(f.dirty[i].from, cs), mapLine(f.dirty[i].to, cs)}
	}
	for _, c := range cs {
		f.dirty = append(f.dirty, lineSpan{c.newLine, c.newEndLine})
	}
}

// searchLines searches the lines edits changed again, as moved left them.
func (f *finder) searchLines(ed *editor) {
	r := &ed.view.text
	n := r.LineCount()
	spans := mergeSpans(f.dirty)
	f.dirty = nil
	for _, sp := range spans {
		from, to := min(sp.from, n-1), min(sp.to, n-1)
		a, z := r.LineStart(from), r.LineEnd(to)
		f.matches = slices.DeleteFunc(f.matches, func(m diffRange) bool { return m.start >= a && m.start <= z })
		i, _ := slices.BinarySearchFunc(f.matches, a, func(m diffRange, off int) int { return m.start - off })
		var found []diffRange
		f.scan(ed, r.Slice(a, z), "", a, func(m diffRange, _ []int) bool {
			found = append(found, m)
			return true
		})
		f.matches = slices.Insert(f.matches, i, found...)
	}
	if len(f.matches) > maxMatches {
		f.matches = f.matches[:maxMatches]
	}
	f.current = min(f.current, max(0, len(f.matches)-1))
}

// asciiLower lowers the ASCII letters of s, keeping its length.
func asciiLower(s string) string {
	b := []byte(s)
	for i, c := range b {
		if c >= 'A' && c <= 'Z' {
			b[i] = c + 32
		}
	}
	return string(b)
}

// nearest returns the match at or after off, round to the first.
func (f *finder) nearest(off int) int {
	i := sort.Search(len(f.matches), func(i int) bool { return f.matches[i].start >= off })
	if i == len(f.matches) {
		return 0
	}
	return i
}

// step selects the next match (or previous) after the newest caret.
func (f *finder) step(ed *editor, dir int) bool {
	f.search(ed)
	if len(f.matches) == 0 {
		return false
	}
	p := ed.primary()
	var i int
	if dir > 0 {
		i = f.nearest(p.end())
		if f.matches[i].start == p.start() && f.matches[i].end == p.end() && i+1 <= len(f.matches) {
			i = (i + 1) % len(f.matches)
		}
	} else {
		i = f.nearest(p.start()) - 1
		if i < 0 {
			i = len(f.matches) - 1
		}
	}
	f.current = i
	m := f.matches[i]
	ed.setSels([]sel{ed.newSel(m.start, m.end)}, revealCenter)
	return true
}

// replaceOne replaces the current match when selected, and goes to the
// next.
func (f *finder) replaceOne(ed *editor) {
	f.search(ed)
	p := ed.primary()
	text := f.textForReplacing(ed)
	for i, m := range f.matches {
		if m.start == p.start() && m.end == p.end() {
			with := f.replacement(text, i)
			ed.apply([]Edit{{Start: m.start, End: m.end, Text: with}}, editOther, func(move func(int, bias) int) []sel {
				at := move(m.start, biasLeft) + len(with)
				return []sel{ed.newSel(at, at)}
			})
			f.search(ed)
			f.current = i
			f.step(ed, 1)
			return
		}
	}
	f.step(ed, 1)
}

// replaceAll replaces every match, as one step of undo.
func (f *finder) replaceAll(ed *editor) {
	f.search(ed)
	if len(f.matches) == 0 {
		return
	}
	edits := make([]Edit, len(f.matches))
	text := f.textForReplacing(ed)
	for i, m := range f.matches {
		edits[i] = Edit{Start: m.start, End: m.end, Text: f.replacement(text, i)}
	}
	ed.apply(edits, editOther, nil)
}

// textForReplacing returns the text a regular expression's replacements
// expand their submatches from, nothing else needing it.
func (f *finder) textForReplacing(ed *editor) string {
	if !f.regex || len(f.subs) == 0 {
		return ""
	}
	return ed.view.text.String()
}

// replacement returns the replacement of match i: the replace text, with
// $1 and ${name} expanded for a regular expression, from the submatches
// found in the whole text, as \b and ^ need the text around a match.
func (f *finder) replacement(text string, i int) string {
	if !f.regex || i >= len(f.subs) {
		return f.replace
	}
	pat := f.query
	if !f.caseSens {
		pat = "(?i)" + pat
	}
	re, err := regexp.Compile("(?m)" + pat)
	if err != nil {
		return f.replace
	}
	return string(re.ExpandString(nil, f.replace, text, f.subs[i]))
}

// status is what the bar says of the matches, in the language of tx.
func (f *finder) status(tx l10n.Messages) string {
	switch {
	case f.err != nil:
		return tx.Invalid()
	case f.query == "":
		return ""
	case len(f.matches) == 0:
		return tx.NoResults()
	case len(f.matches) >= maxMatches:
		return tx.MatchesMore(f.current+1, len(f.matches))
	}
	return tx.Matches(f.current+1, len(f.matches))
}

// selectAll selects every match, as alt-enter in Zed's search bar.
func (f *finder) selectAll(ed *editor) {
	f.search(ed)
	if len(f.matches) == 0 {
		return
	}
	sels := make([]sel, 0, len(f.matches))
	for _, m := range f.matches {
		sels = append(sels, ed.newSel(m.start, m.end))
	}
	ed.setSels(sels, revealFit)
	f.open = false
	ed.refocus = true
}

// goToLine is the bar going to a line: line[:column], or +n and -n
// lines from the caret, as Zed's go to line, which shows the line as it
// is typed and goes back where it was on Escape.
type goToLine struct {
	open    bool
	text    string
	focus   bool
	savedY  float64
	savedX  float32
	preview int
}

func (g *goToLine) show(ed *editor) {
	g.open, g.text, g.focus = true, "", true
	g.savedY, g.savedX = ed.scrollY, ed.scrollX
	g.preview = -1
	ed.needsBuild = true
}

func (g *goToLine) close(ed *editor) {
	g.open = false
	ed.setScroll(g.savedY)
	ed.scrollX = g.savedX
	g.preview = -1
	ed.refocus = true
}

// target parses the text into a line and column, from 0.
func (g *goToLine) target(ed *editor) (int, int, bool) {
	s := strings.TrimSpace(g.text)
	if s == "" {
		return 0, 0, false
	}
	cur := ed.view.text.LineOf(ed.primary().head)
	rel := 0
	if s[0] == '+' || s[0] == '-' {
		rel = 1
		if s[0] == '-' {
			rel = -1
		}
		s = s[1:]
	}
	parts := strings.SplitN(s, ":", 2)
	line, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		return 0, 0, false
	}
	col := 0
	if len(parts) == 2 {
		if c, err := strconv.Atoi(strings.TrimSpace(parts[1])); err == nil && c > 0 {
			col = c - 1
		}
	}
	if rel != 0 {
		line = cur + rel*line
	} else {
		line--
	}
	line = max(0, min(line, ed.view.text.LineCount()-1))
	// The column counts characters, the position bytes.
	return line, byteOfRuneIn(ed.view.text.Line(line), col), true
}
