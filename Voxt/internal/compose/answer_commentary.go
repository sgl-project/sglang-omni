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

// cleanedSpan is s from the first sentence of the cleaned text on: a
// sentence whose folded bigrams mostly occur in the folded input, with
// short connectors (嗯, OK) just before it included. The deliberation a
// noncompliant model writes comes before the final text, so only what
// precedes that sentence is dropped; what follows it stays, since a
// custom prompt may legitimately expand the dictation and a foreign
// tail cannot be told from that expansion. ok is false when no sentence
// is the cleaned text: a wholesale rewrite (a custom prompt's
// translation, say), an input too short to tell its material from the
// model's words, or a stream that has not reached the final text yet —
// the caller then keeps the text as it is.
// shortcut: deliberation quoting a whole input sentence anchors that
// sentence too; a tag-less tail after the cleaned text (I removed…)
// stays, as does any aside when the answer did not balloon.
func cleanedSpan(s, input string) (string, bool) {
	in := folded(input)
	if len([]rune(in)) < 8 {
		return "", false
	}
	segs := sentences(s)
	for i, seg := range segs {
		f := folded(seg)
		if len([]rune(f)) < 4 {
			continue
		}
		// An unfinished sentence anchors at a higher share: its share
		// still moves as it grows — a quoted fragment or a coincidental
		// pair reads as a half before the rest of the words dilute it.
		bar := 0.5
		if !terminated(seg) {
			bar = 0.65
		}
		if inputShare(f, in) >= bar {
			for i > 0 && len([]rune(folded(segs[i-1]))) < 4 {
				i--
			}
			return strings.TrimSpace(strings.Join(segs[i:], "")), true
		}
	}
	return "", false
}

// terminated reports whether a sentence piece ends on a sentence end.
func terminated(seg string) bool {
	rs := []rune(strings.TrimRight(seg, " \t\r\n"))
	if len(rs) == 0 {
		return true
	}
	return strings.ContainsRune("。！？!?.", rs[len(rs)-1])
}

// deliberating reports whether a streaming tag-less answer has a
// finished sentence of clearly foreign material: the model's
// deliberation about its edits, which the draft hides (#4086). A
// sentence still being typed passes through as it is — it may be the
// cleaned text's start, and the draft corrects when the sentence
// completes and shows itself foreign.
func deliberating(s, in string) bool {
	for _, seg := range sentences(s) {
		if f := folded(seg); terminated(seg) && len([]rune(f)) >= 8 && inputShare(f, in) < 0.25 {
			return true
		}
	}
	return false
}

// ballooned reports whether an answer is far longer than a clean-up of
// input can produce: the sign that the model added its deliberation.
func ballooned(s, input string) bool {
	out, in := len([]rune(folded(s))), len([]rune(folded(input)))
	return out > 80 && out > in*3/2+24
}
