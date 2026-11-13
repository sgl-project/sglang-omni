package dict

import (
	"math/rand/v2"
	"sort"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestReplaceMatchesScan: Replace gives what trying every pattern at
// every place gives (replaceScan, Replace before it looked patterns up by
// their first letter), for random terms and texts of letters with several
// cases (K and the Kelvin sign, s and the long s, the three sigmas),
// Chinese, spaces, punctuation and a byte that is not UTF-8.
func TestReplaceMatchesScan(t *testing.T) {
	letters := []string{"a", "A", "k", "K", "K", "s", "S", "ſ", "σ", "Σ", "ς", "ß", "1", "奥", "拉", "林", " ", "  ", "\t", "-", ",", "\xff"}
	rng := rand.New(rand.NewPCG(3785, 1))
	word := func(n int) string {
		var b strings.Builder
		for range 1 + rng.IntN(n) {
			b.WriteString(letters[rng.IntN(len(letters))])
		}
		return b.String()
	}
	for range 5000 {
		terms := make([]store.Term, 1+rng.IntN(6))
		for i := range terms {
			terms[i] = store.Term{Term: word(4), Enabled: rng.IntN(5) > 0}
			for range rng.IntN(3) {
				terms[i].Aliases = append(terms[i].Aliases, word(4))
			}
		}
		text := word(40)
		if got, want := Replace(text, terms), replaceScan(text, terms); got != want {
			t.Fatalf("Replace(%q, %+v) = %q, trying every pattern gives %q", text, terms, got, want)
		}
	}
}

// replaceScan is Replace as it was before it looked patterns up by their
// first letter: every pattern at every place.
func replaceScan(text string, terms []store.Term) string {
	type pattern struct {
		words []string
		term  string
		size  int
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
