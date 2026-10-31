package dict

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// Replace writes the enabled terms' spelling for their aliases in text,
// a transcript: aliases are, by definition, what speech recognition
// writes for a term ("Or Link" for Auralink). Latin aliases match whole
// words, ignoring case, any run of spaces for a space; Chinese,
// Japanese and Korean ones match anywhere. At each place the longest
// match wins, and a term already written as itself stays (so Replace
// changes its own output no further). An alias two terms share goes to
// the first.
func Replace(text string, terms []store.Term) string {
	type pattern struct {
		words []string // the alias (or term), split at its spaces
		term  string   // "" for a term itself: kept as written
		size  int      // characters, for the longest first
	}
	var patterns []pattern
	add := func(s, term string) {
		words := strings.Fields(s)
		if len(words) > 0 {
			patterns = append(patterns, pattern{words, term, utf8.RuneCountInString(strings.Join(words, " "))})
		}
	}
	for _, t := range terms {
		term := strings.TrimSpace(t.Term)
		if !t.Enabled || term == "" {
			continue
		}
		replaced := false
		for _, a := range t.Aliases {
			if a = strings.TrimSpace(a); a != "" && !strings.EqualFold(a, term) {
				add(a, term)
				replaced = true
			}
		}
		if replaced {
			add(term, "")
		}
	}
	if len(patterns) == 0 {
		return text
	}
	sort.SliceStable(patterns, func(i, j int) bool { return patterns[i].size > patterns[j].size })
	var b strings.Builder
	for i := 0; i < len(text); {
		matched := false
		for _, p := range patterns {
			if n := matchAt(text, i, p.words); n > 0 {
				if p.term == "" {
					b.WriteString(text[i : i+n])
				} else {
					b.WriteString(p.term)
				}
				i += n
				matched = true
				break
			}
		}
		if !matched {
			_, n := utf8.DecodeRuneInString(text[i:])
			b.WriteString(text[i : i+n])
			i += n
		}
	}
	return b.String()
}

// matchAt is the length in bytes of words matched at text[i:] (0 for
// none): the words in order, ignoring case, white space between them;
// a word boundary at either end that is Latin.
func matchAt(text string, i int, words []string) int {
	j := i
	for k, w := range words {
		if k > 0 {
			n := 0
			for j+n < len(text) {
				r, size := utf8.DecodeRuneInString(text[j+n:])
				if !unicode.IsSpace(r) {
					break
				}
				n += size
			}
			if n == 0 {
				return 0
			}
			j += n
		}
		n := prefixFold(text[j:], w)
		if n == 0 {
			return 0
		}
		j += n
	}
	first, _ := utf8.DecodeRuneInString(words[0])
	last, _ := utf8.DecodeLastRuneInString(words[len(words)-1])
	if before, _ := utf8.DecodeLastRuneInString(text[:i]); i > 0 && wordRune(first) && wordRune(before) {
		return 0
	}
	if after, _ := utf8.DecodeRuneInString(text[j:]); j < len(text) && wordRune(last) && wordRune(after) {
		return 0
	}
	return j - i
}

// prefixFold is the length in bytes of the prefix of s equal to w,
// ignoring case (0 when s does not start so).
func prefixFold(s, w string) int {
	j := 0
	for _, wr := range w {
		if j >= len(s) {
			return 0
		}
		r, n := utf8.DecodeRuneInString(s[j:])
		if !foldEqual(r, wr) {
			return 0
		}
		j += n
	}
	return j
}

// foldEqual: a and b are the same letter, ignoring case.
func foldEqual(a, b rune) bool {
	for f := a; ; {
		if f == b {
			return true
		}
		if f = unicode.SimpleFold(f); f == a {
			return false
		}
	}
}

// wordRune: a letter or digit of a script that separates its words with
// spaces (not Chinese, Japanese or Korean, whose aliases match anywhere).
func wordRune(r rune) bool {
	if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
		return false
	}
	return !unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul)
}
