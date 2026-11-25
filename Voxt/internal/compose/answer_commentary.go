package compose

import (
	"strings"
	"unicode"
)

// This file is the tag-less half of the clean-up's output contract
// (#4086): a model that ignores the <output> tags and thinks out loud —
// its deliberation about the edits, then the final text — still shows
// and inserts only the cleaned text. A clean-up only lightly edits, so
// the final text's sentences share most of their characters with the
// input and the deliberation does not.

// folded is s for comparison: lower case, without spaces, punctuation
// or symbols.
func folded(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range strings.ToLower(s) {
		if !unicode.IsSpace(r) && !unicode.IsPunct(r) && !unicode.IsSymbol(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// inputShare is the fraction of seg's bigrams that occur in in, both
// folded: the cleaned text consists of the input's material.
func inputShare(seg, in string) float64 {
	s, t := []rune(seg), []rune(in)
	if len(s) < 2 {
		return 0
	}
	have := make(map[[2]rune]bool, len(t))
	for i := 0; i+1 < len(t); i++ {
		have[[2]rune{t[i], t[i+1]}] = true
	}
	hit := 0
	for i := 0; i+1 < len(s); i++ {
		if have[[2]rune{s[i], s[i+1]}] {
			hit++
		}
	}
	return float64(hit) / float64(len(s)-1)
}

// sentences splits s after each sentence end (。！？!?, a newline, or "."
// before a space or the end), every piece keeping its ending; the last
// piece may be unfinished.
func sentences(s string) []string {
	rs := []rune(s)
	var out []string
	start := 0
	for i, r := range rs {
		if strings.ContainsRune("。！？!?\n", r) || r == '.' && (i+1 == len(rs) || rs[i+1] == ' ' || rs[i+1] == '\n') {
			out = append(out, string(rs[start:i+1]))
			start = i + 1
		}
	}
	if start < len(rs) {
		out = append(out, string(rs[start:]))
	}
	return out
}

// cleanedSpan is s from its first to its last sentence of the cleaned
// text: a sentence whose folded bigrams mostly occur in the folded
// input. Short connectors (嗯, OK) next to such a sentence are part of
// it. ok is false when no sentence is the cleaned text: a wholesale
// rewrite (a custom prompt's translation, say), an input too short to
// tell its material from the model's words, or a stream that has not
// reached the final text yet — the caller then keeps the text as it is.
// shortcut: deliberation quoting a whole input sentence anchors that
// sentence too, and a short aside below ballooned stays; a model that
// wraps its deliberation around near-verbatim quotes still leaks.
func cleanedSpan(s, input string) (string, bool) {
	in := folded(input)
	if len([]rune(in)) < 8 {
		return "", false
	}
	segs := sentences(s)
	first, last := -1, -1
	for i, seg := range segs {
		if f := folded(seg); len([]rune(f)) >= 4 && inputShare(f, in) >= 0.5 {
			if first < 0 {
				first = i
			}
			last = i
		}
	}
	if first < 0 {
		return "", false
	}
	for first > 0 && len([]rune(folded(segs[first-1]))) < 4 {
		first--
	}
	for last+1 < len(segs) && len([]rune(folded(segs[last+1]))) < 4 {
		last++
	}
	return strings.TrimSpace(strings.Join(segs[first:last+1], "")), true
}

// ballooned reports whether an answer is far longer than a clean-up of
// input can produce: the sign that the model added its deliberation.
func ballooned(s, input string) bool {
	out, in := len([]rune(folded(s))), len([]rune(folded(input)))
	return out > 80 && out > in*3/2+24
}

// deliberating reports whether a streaming tag-less answer has shown
// itself to be the model's deliberation rather than the cleaned text
// (#4086): a sentence of clearly foreign material (eight or more folded
// runes, under a quarter of them the input's) that is complete and
// followed by more text, or still growing past sixteen runes. The bar
// is lower than cleanedSpan's, so prose that merely mentions the
// input's words still shows. Until a sentence shows itself foreign the
// stream shows as it is: a deliberation's first sentence can stream
// into the draft before a second one reveals it, and CleanOutput's
// trimmed result replaces it.
func deliberating(s, in string) bool {
	segs := sentences(s)
	for i, seg := range segs {
		if f := folded(seg); len([]rune(f)) >= 8 && inputShare(f, in) < 0.25 {
			if i+1 < len(segs) || len([]rune(f)) >= 16 && !terminated(seg) {
				return true
			}
		}
	}
	return false
}

// terminated reports whether a sentence piece ends on a sentence end.
func terminated(seg string) bool {
	rs := []rune(strings.TrimRight(seg, " \t\r\n"))
	if len(rs) == 0 {
		return true
	}
	return strings.ContainsRune("。！？!?.", rs[len(rs)-1])
}
