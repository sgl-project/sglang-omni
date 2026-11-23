package eval

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Verdict is a clean-up (or a translation) of a clip, checked.
type Verdict struct {
	// Fillers are fillers the answer still has.
	Fillers []string
	// Missing are Must phrases it lost; Kept MustNot phrases it kept (a
	// filler, what a self-correction replaced).
	Missing, Kept []string
	// Invented are its tokens in neither the input nor the expected
	// answer: words the model made up (a clean-up only).
	Invented []string
	// Copied: it has a phrase of the prompt's example answer that the
	// input has not.
	Copied bool
	// Language: a clean-up changed the language: the share of Chinese in
	// it is more than 30 points off the input's (it translated); a
	// translation into English left a fifth of it in Chinese.
	Language bool
	// Score is the answer against the expected one (a clean-up only).
	Score Score
}

// OK is a verdict without a fault (at most one invented token: a
// rephrasing may add a word).
func (v Verdict) OK() bool {
	return len(v.Fillers) == 0 && len(v.Missing) == 0 && len(v.Kept) == 0 && len(v.Invented) <= 1 && !v.Copied && !v.Language
}

// Corrections are the Must and MustNot phrases (other than fillers) it
// got wrong: mostly self-corrections not applied.
func (v Verdict) Corrections() int {
	n := len(v.Missing)
	for _, k := range v.Kept {
		if !isFiller(strings.ToLower(k)) {
			n++
		}
	}
	return n
}

// Faults says what is wrong, "" for nothing.
func (v Verdict) Faults() string {
	var f []string
	add := func(what string, items []string) {
		if len(items) > 0 {
			f = append(f, what+" "+strings.Join(items, ","))
		}
	}
	add("fillers", v.Fillers)
	add("lost", v.Missing)
	add("kept", v.Kept)
	if len(v.Invented) > 1 {
		add("invented", v.Invented)
	}
	if v.Copied {
		f = append(f, "copied the example")
	}
	if v.Language {
		f = append(f, "changed the language")
	}
	return strings.Join(f, "; ")
}

// CheckClean checks out, a clean-up of c's text, against the clip's
// expectations; example is the prompt's example answer ("" for none).
func CheckClean(c Clip, input, out, example string) Verdict {
	v := check(out, input, example, c.Must, c.MustNot)
	v.Score = Compare(c.Clean, out)
	if d := cjkShare(out) - cjkShare(input); d > 0.3 || d < -0.3 {
		v.Language = true
	}
	have := map[string]bool{}
	for _, t := range append(Tokens(input), Tokens(c.Clean)...) {
		have[t] = true
	}
	for _, t := range Tokens(out) {
		if !have[t] && strings.Trim(t, "0123456789") != "" { // numbers may be written anew
			v.Invented = append(v.Invented, t)
		}
	}
	return v
}

// CheckTranslation checks out, a translation of c's text, against the
// clip's expectations of an English translation (MustEn, MustNotEn).
func CheckTranslation(must, mustNot []string, input, out, example string) Verdict {
	v := check(out, input, example, must, mustNot)
	v.Language = cjkShare(out) > 0.2
	return v
}

func check(out, input, example string, must, mustNot []string) Verdict {
	var v Verdict
	low := strings.ToLower(out)
	words := map[string]bool{}
	for _, w := range strings.FieldsFunc(low, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }) {
		words[w] = true
	}
	has := func(p string) bool {
		p = strings.ToLower(p)
		if isFiller(p) && p[0] < 0x80 { // "um": a word, not a part of "umbrella"
			return words[p]
		}
		return strings.Contains(low, p)
	}
	for _, f := range Fillers {
		if has(f) && !strings.Contains(strings.ToLower(input), "不"+f) {
			v.Fillers = append(v.Fillers, f)
		}
	}
	for _, m := range must {
		ok := false
		for _, alt := range strings.Split(m, "|") {
			ok = ok || has(alt)
		}
		if !ok {
			v.Missing = append(v.Missing, m)
		}
	}
	for _, m := range mustNot {
		if has(m) {
			v.Kept = append(v.Kept, m)
		}
	}
	v.Copied = copied(out, input, example)
	return v
}

// copied: out has a run of the example answer that input has not: four
// Chinese characters, or five words of Latin text.
func copied(out, input, example string) bool {
	o, in := Tokens(out), Tokens(input)
	has := func(toks []string, run []string) bool {
		for i := 0; i+len(run) <= len(toks); i++ {
			if slices.Equal(toks[i:i+len(run)], run) {
				return true
			}
		}
		return false
	}
	ex := Tokens(example)
	for i := range ex {
		n := 5
		if r, _ := utf8.DecodeRuneInString(ex[i]); isCJK(r) {
			n = 4
		}
		if i+n > len(ex) {
			continue
		}
		run := ex[i : i+n]
		if has(o, run) && !has(in, run) {
			return true
		}
	}
	return false
}

// cjkShare is the share of text's tokens that are Chinese (Japanese,
// Korean) characters.
func cjkShare(text string) float64 {
	toks := Tokens(text)
	if len(toks) == 0 {
		return 0
	}
	n := 0
	for _, t := range toks {
		if r, _ := utf8.DecodeRuneInString(t); isCJK(r) {
			n++
		}
	}
	return float64(n) / float64(len(toks))
}
