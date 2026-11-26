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

// cleanedSpan is s from the first to the last sentence of the cleaned
// text: a sentence whose folded bigrams mostly occur in the folded
// input, with short connectors (嗯, OK) next to such a sentence
// included. The clean-up's built-in contract allows only the cleaned
// text, so what a noncompliant model writes around it — its
// deliberation about the edits — is dropped; the sentences between the
// anchors stay, since a clean-up may reword one of them. ok is false
// when no sentence is the cleaned text: a wholesale rewrite (a custom
// prompt's translation, say), an input too short to tell its material
// from the model's words, or a stream that has not reached the final
// text yet — the caller then keeps the text as it is.
// shortcut: deliberation quoting a whole input sentence anchors that
// sentence too, and a heavily reworded first or last sentence can fall
// outside the span; both are rare next to the reported leak.
func cleanedSpan(s, input string) (string, bool) {
	in := folded(input)
	if len([]rune(in)) < 2 {
		// Under two runes the input has no bigrams, so no sentence can
		// be told from the model's words: keep the text as it is.
		return "", false
	}
	segs := sentences(s)
	first, last := -1, -1
	for i, seg := range segs {
		f := folded(seg)
		n := len([]rune(f))
		var anchor bool
		if n >= 4 {
			// An unfinished sentence anchors at a higher share: its
			// share still moves as it grows — a quoted fragment or a
			// coincidental pair reads as a half before the rest of the
			// words dilutes it.
			bar := 0.5
			if !terminated(seg) {
				bar = 0.65
			}
			anchor = inputShare(f, in) >= bar
		} else if n > 0 && terminated(seg) {
			// An exact short answer (Hi., 你好。): the whole input, or
			// contained in it. A connector that is not the input (OK,
			// 嗯 before the cleaned text) does not anchor; it keeps the
			// side of the sentence next to it, below. An empty segment
			// (a blank line) never anchors either: containment holds
			// for it trivially.
			anchor = f == in || strings.Contains(in, f) || strings.Contains(f, in)
		}
		if anchor {
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

// terminated reports whether a sentence piece ends on a sentence end.
func terminated(seg string) bool {
	rs := []rune(strings.TrimRight(seg, " \t\r\n"))
	if len(rs) == 0 {
		return true
	}
	return strings.ContainsRune("。！？!?.", rs[len(rs)-1])
}

// deliberating reports whether a streaming tag-less answer has a
// finished sentence that is not the cleaned text, before any sentence
// of it has anchored: the model's deliberation, which the draft hides
// (#4086). A sentence still being typed passes through — it may be the
// cleaned text's start, and the draft corrects when it completes. An
// input too short for bigrams, or carrying control markup of its own,
// keeps the text as it is.
func deliberating(s, input string) bool {
	in := folded(input)
	if len([]rune(in)) < 2 || markupInput(input) {
		return false
	}
	for _, seg := range sentences(s) {
		if f := folded(seg); terminated(seg) && len([]rune(f)) >= 8 && inputShare(f, in) < 0.5 {
			return true
		}
	}
	return false
}

// markupInput reports whether the input itself carries the control or
// thinking markup whose answers keep their literal form: its draft
// keeps the literal-markup rules, with no deliberation hiding.
func markupInput(input string) bool {
	for _, tag := range thinkingTags {
		if strings.Contains(input, tag[0]) || strings.Contains(input, tag[1]) {
			return true
		}
	}
	for _, token := range controlTokens {
		if strings.Contains(input, token) {
			return true
		}
	}
	return false
}
