package meeting

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// loopRepeats copies of a unit back to back are a speech model's loop, not
// speech: a stutter ("我我我") or "对对对对对" stays (#3981).
const loopRepeats = 8

// maxLoopUnit is the longest unit looked for, in runes: a looped sentence
// in #3981 ran 97.
const maxLoopUnit = 160

// collapseLoops cuts each run of loopRepeats or more copies of a unit with a
// letter in it to one copy, and drops a partial copy that ends the text after
// such a run. Runs without a letter stay: "100000000" is a number. Of the
// run's phases, the copy kept ends at a punctuation mark (spaces aside) when
// one does: "一个，那个，那个，…" keeps "那个，", not "个，那".
func collapseLoops(s string) string {
	rs := []rune(s)
	out := make([]rune, 0, len(rs))
	changed := false
	for i := 0; i < len(rs); {
		u, n := loopAt(rs, i)
		if n == 0 {
			out = append(out, rs[i])
			i++
			continue
		}
		changed = true
		for k := i; k < i+u; k++ {
			if endsAtPunct(rs[k:k+u]) && copiesAt(rs, k, u) >= loopRepeats {
				out = append(out, rs[i:k]...)
				i, n = k, copiesAt(rs, k, u)
				break
			}
		}
		out = append(out, rs[i:i+u]...)
		i += n * u
		if rest := rs[i:]; len(rest) < u && slices.Equal(rest, rs[i-u:i-u+len(rest)]) {
			i = len(rs)
		}
	}
	if !changed {
		return s
	}
	return strings.TrimSpace(string(out))
}

// loopAt is the shortest unit starting at rs[i] that repeats loopRepeats
// times or more, and how many times; 0, 0 when there is none.
func loopAt(rs []rune, i int) (unit, count int) {
	for u := 1; u <= maxLoopUnit && i+u*loopRepeats <= len(rs); u++ {
		if n := copiesAt(rs, i, u); n >= loopRepeats && slices.ContainsFunc(rs[i:i+u], unicode.IsLetter) {
			return u, n
		}
	}
	return 0, 0
}

// copiesAt is how many copies of rs[i:i+u] follow one another from i.
func copiesAt(rs []rune, i, u int) int {
	n := 1
	for i+(n+1)*u <= len(rs) && slices.Equal(rs[i+n*u:i+(n+1)*u], rs[i:i+u]) {
		n++
	}
	return n
}

// endsAtPunct says whether unit's last rune but spaces is punctuation.
func endsAtPunct(unit []rune) bool {
	t := strings.TrimRightFunc(string(unit), unicode.IsSpace)
	r, _ := utf8.DecodeLastRuneInString(t)
	return t != "" && unicode.IsPunct(r)
}
