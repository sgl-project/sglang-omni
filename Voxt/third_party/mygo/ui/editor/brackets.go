package editor

import "strings"

// Bracket matching, outside strings and comments by the tokens of the
// lines, within a bounded scan as Zed bounds its queries.

const bracketScanLines = 1000

// bracketCache keeps the brackets around the newest caret, found once
// for a version and a caret.
type bracketCache struct {
	version     uint64
	hlGen       int
	at          int
	open, close Range
	ok          bool
	valid       bool
}

// bracketPairs returns the language's brackets of one byte each.
func (ed *editor) bracketPairs() (opens, closes string) {
	l := ed.language()
	if l == nil {
		return "([{", ")]}"
	}
	for _, p := range l.Brackets {
		if len(p.Open) == 1 && len(p.Close) == 1 && p.Open != "<" {
			opens += p.Open
			closes += p.Close
		}
	}
	return opens, closes
}

func bracketToken(k TokenKind) bool {
	return k != TokenString && k != TokenComment && k != TokenRegexp && k != TokenEscape && k != TokenCode
}

// matchBracket finds the bracket matching the one at off.
func (ed *editor) matchBracket(off int) (int, int, bool) {
	r := &ed.view.text
	opens, closes := ed.bracketPairs()
	c := r.ByteAt(off)
	for i := 0; i < len(opens); i++ {
		if opens[i] == c {
			if z, ok := ed.scanBracket(off+1, 1, opens[i], closes[i]); ok {
				return off, z, true
			}
			return 0, 0, false
		}
		if closes[i] == c {
			if a, ok := ed.scanBracket(off-1, -1, closes[i], opens[i]); ok {
				return a, off, true
			}
			return 0, 0, false
		}
	}
	return 0, 0, false
}

// scanBracket looks from off in dir for the unmatched want, counting
// nested pairs of same and want, skipping strings and comments.
func (ed *editor) scanBracket(off, dir int, same, want byte) (int, bool) {
	r := &ed.view.text
	if off < 0 || off >= r.Len() {
		return 0, false
	}
	line := r.LineOf(off)
	depth := 0
	found := -1
	visit := func(i, a int, s string) bool {
		toks := ed.view.lineTokens(i, s)
		var at int
		if dir > 0 {
			at = 0
			if off >= a && off <= a+len(s) {
				at = off - a
			}
		} else {
			at = len(s) - 1
			if off >= a && off < a+len(s) {
				at = off - a
			}
		}
		for ; at >= 0 && at < len(s); at += dir {
			c := s[at]
			if (c == same || c == want) && bracketToken(kindAt(toks, at)) {
				if c == same {
					depth++
				} else if depth == 0 {
					found = a + at
					return false
				} else {
					depth--
				}
			}
		}
		return true
	}
	if dir > 0 {
		r.EachLine(line, line+bracketScanLines, visit)
	} else {
		r.EachLineBack(line, line-bracketScanLines, visit)
	}
	return found, found >= 0
}

// lineTokens returns the tokens of line i, whose text is s, without
// laying it out: those of the line laid out when it is, else in a
// buffer the next call reuses. Nil means the whole line is code, as a
// line starting outside strings and comments without the characters
// that start them is, without tokenizing it.
func (v *view) lineTokens(i int, s string) []Token {
	if l, ok := v.lines.m[i]; ok {
		return l.toks
	}
	if v.hl == nil || len(s) > maxTokenize {
		return nil
	}
	// The default highlighter's lines starting in code without the
	// characters starting strings and comments are all code.
	if th, ok := v.hl.(*tokenHighlighter); ok {
		if state, _ := th.stateOf(i); state == 0 && !strings.ContainsAny(s, "\"'`/#*-<%;") {
			return nil
		}
	}
	v.tokScratch, _ = v.hl.Tokens(i, s, v.tokScratch[:0])
	return v.tokScratch
}

// enclosingBrackets returns the innermost pair of brackets around off,
// the ranges of the opening and the closing one, as Zed highlights.
func (ed *editor) enclosingBrackets(off int) (Range, Range, bool) {
	c := &ed.brackets
	if c.valid && c.version == ed.view.version && c.hlGen == ed.view.hlGen && c.at == off {
		return c.open, c.close, c.ok
	}
	open, close, ok := ed.bracketsAround(off, off)
	*c = bracketCache{version: ed.view.version, hlGen: ed.view.hlGen, at: off, open: open, close: close, ok: ok, valid: true}
	return open, close, ok
}

// bracketsAround returns the innermost pair of brackets around a to z,
// or touching them, as BracketMatcher says: the highlighter's, when it
// matches brackets, else those the tokens show outside strings and
// comments.
func (ed *editor) bracketsAround(a, z int) (Range, Range, bool) {
	if bm, ok := ed.highlighter().(BracketMatcher); ok {
		return bm.EnclosingBrackets(a, z)
	}
	o, c, ok := ed.enclosingBracketsOf(a, z)
	return Range{o, o + 1}, Range{c, c + 1}, ok
}

// highlighter returns the buffer's highlighter, up to date, or nil.
func (ed *editor) highlighter() Highlighter {
	if b := ed.buf(); b != nil {
		return b.highlighter()
	}
	return nil
}

// enclosingBracketsOf returns the innermost pair of brackets around a to
// z, by the tokens: the pair a bracket just before z closes, or one at a
// opens, when it holds them, else the opening bracket before a and the
// closing one at or after z.
func (ed *editor) enclosingBracketsOf(a, z int) (int, int, bool) {
	r := &ed.view.text
	opens, closes := ed.bracketPairs()
	o, c, ok := 0, 0, false
	if z > 0 && indexByte(closes, r.ByteAt(z-1)) >= 0 && bracketToken(ed.tokenAt(z)) {
		if o2, c2, ok2 := ed.matchBracket(z - 1); ok2 && o2 <= a {
			o, c, ok = o2, c2, true
		}
	}
	if a < r.Len() && indexByte(opens, r.ByteAt(a)) >= 0 && bracketToken(ed.tokenAt(a+1)) {
		if o2, c2, ok2 := ed.matchBracket(a); ok2 && c2 >= z-1 && (!ok || c2-o2 < c-o) {
			o, c, ok = o2, c2, true
		}
	}
	if ok {
		return o, c, true
	}
	from := a - 1
	for tries := 0; tries < 64 && from >= 0; tries++ {
		open, which, ok := ed.unmatchedOpener(from, opens, closes)
		if !ok {
			return 0, 0, false
		}
		close, ok := ed.scanBracket(open+1, 1, opens[which], closes[which])
		if ok && close >= z {
			return open, close, true
		}
		from = open - 1
	}
	return 0, 0, false
}

// unmatchedOpener scans back from off for an opening bracket the closing
// ones after it do not close, in one pass for every kind of bracket.
func (ed *editor) unmatchedOpener(off int, opens, closes string) (int, int, bool) {
	r := &ed.view.text
	if off < 0 {
		return 0, 0, false
	}
	var stack []byte
	at, which := -1, -1
	line := r.LineOf(min(off, r.Len()))
	r.EachLineBack(line, line-bracketScanLines, func(i, a int, s string) bool {
		// Brackets are rare: look at the tokens only for lines with some.
		if !hasAnyByte(s, opens, closes) {
			return true
		}
		toks := ed.view.lineTokens(i, s)
		k := len(s) - 1
		if off >= a && off < a+len(s) {
			k = off - a
		}
		ti := len(toks) - 1
		for ; k >= 0; k-- {
			c := s[k]
			j := indexByte(opens, c)
			cj := indexByte(closes, c)
			if j < 0 && cj < 0 {
				continue
			}
			for ti > 0 && toks[ti-1].End > k {
				ti--
			}
			if ti >= 0 && ti < len(toks) && !bracketToken(toks[ti].Kind) {
				continue
			}
			if cj >= 0 {
				stack = append(stack, c)
				continue
			}
			if n := len(stack); n > 0 {
				if stack[n-1] == closes[j] {
					stack = stack[:n-1]
				}
				continue
			}
			at, which = a+k, j
			return false
		}
		return true
	})
	return at, which, at >= 0
}

func hasAnyByte(s, a, b string) bool {
	for i := 0; i < len(s); i++ {
		if indexByte(a, s[i]) >= 0 || indexByte(b, s[i]) >= 0 {
			return true
		}
	}
	return false
}

func indexByte(s string, c byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			return i
		}
	}
	return -1
}
