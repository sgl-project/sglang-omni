package dict

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Span is a match's byte range in the original text.
type Span struct {
	Start, End int
}

// Replace writes the enabled terms' spelling for their aliases in text,
// a transcript: aliases are, by definition, what speech recognition
// writes for a term ("Or Link" for Auralink). Latin aliases match whole
// words, ignoring case, any run of spaces for a space; Chinese,
// Japanese and Korean ones match anywhere. At each place the longest
// match wins, and a term already written as itself stays (so Replace
// changes its own output no further). An alias two terms share goes to
// the first.
func Replace(text string, terms []store.Term) string {
	text, _ = matchText(text, terms, false, builtinRules{from: len(terms)})
	return text
}

// HighlightSpans finds enabled terms and their aliases in text with the
// same matching and overlap rules as Replace. Its spans are byte ranges
// in text, with the longest match at each place from left to right.
func HighlightSpans(text string, terms []store.Term) []Span {
	_, spans := matchText(text, terms, true, builtinRules{from: len(terms)})
	return spans
}

// matchText is the dictionary's one matcher. highlight keeps the input
// and collects its matches, including terms without aliases; otherwise it
// performs Replace's alias substitution, and writes a term in spelled as
// spelled where it is found in another case ("Github" for GitHub), except
// one word all in lower or all in upper case: that is how code, files and
// addresses write it (config.json, OPENAI_API_KEY), and it stays. Terms
// from index rules.from on are the built-in words': a match of theirs
// inside a longer token (inToken: X-Api-Key, Github.com, Support/SGLang
// Omni/) stays as written, and so does one where its rule says it is not
// the name (rules.guarded, casing's where).
func matchText(text string, terms []store.Term, highlight bool, rules builtinRules) (string, []Span) {
	type pattern struct {
		words   []string                        // the alias (or term), split at its spaces
		term    string                          // "" for a term itself: kept as written
		size    int                             // characters, for the longest first
		spell   string                          // a term in spelled: written so, unless oneCaseWord
		where   func(before, after string) bool // where it is the name (nil: everywhere)
		builtin bool                            // a built-in word's: not inside a token
	}
	var patterns []pattern
	fromBuiltin := false
	add := func(s, term string) {
		words := strings.Fields(s)
		if len(words) > 0 {
			p := pattern{words: words, term: term, size: utf8.RuneCountInString(strings.Join(words, " ")), builtin: fromBuiltin}
			if fromBuiltin && term != "" {
				p.where = rules.guarded[strings.ToLower(s)]
			}
			patterns = append(patterns, p)
		}
	}
	for k, t := range terms {
		fromBuiltin = k >= rules.from
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
		sp, spell := rules.casing[term]
		if replaced || highlight || spell {
			add(term, "")
			if spell && !highlight {
				patterns[len(patterns)-1].spell, patterns[len(patterns)-1].where = term, sp.where
			}
		}
	}
	if len(patterns) == 0 {
		return text, nil
	}
	sort.SliceStable(patterns, func(i, j int) bool { return patterns[i].size > patterns[j].size })
	// Each place tries only the patterns that start with its letter, in
	// any case, in the same order.
	byFirst := map[rune][]pattern{}
	for _, p := range patterns {
		r, _ := utf8.DecodeRuneInString(p.words[0])
		byFirst[foldKey(r)] = append(byFirst[foldKey(r)], p)
	}
	var b strings.Builder
	var spans []Span
	for i := 0; i < len(text); {
		r, _ := utf8.DecodeRuneInString(text[i:])
		matched := false
		for _, p := range byFirst[foldKey(r)] {
			if n := matchAt(text, i, p.words); n > 0 {
				if highlight {
					spans = append(spans, Span{Start: i, End: i + n})
				} else if p.builtin && inToken(text, i, i+n) {
					b.WriteString(text[i : i+n])
				} else if p.where != nil && !p.where(text[:i], text[i+n:]) {
					b.WriteString(text[i : i+n])
				} else if p.spell != "" && !oneCaseWord(text[i:i+n]) {
					b.WriteString(p.spell)
				} else if p.term == "" {
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
			if !highlight {
				b.WriteString(text[i : i+n])
			}
			i += n
		}
	}
	if highlight {
		return text, spans
	}
	return b.String(), nil
}

// joiners join words into one token in code, paths and addresses.
const joiners = "-_./\\@:#~+="

// inToken: text[start:end] is joined to a word before or after it by one
// of joiners ("X-Api-Key", "Github.com", "Support/SGLang Omni/models").
func inToken(text string, start, end int) bool {
	if r, n := utf8.DecodeLastRuneInString(text[:start]); start > 0 && strings.ContainsRune(joiners, r) {
		if w, _ := utf8.DecodeLastRuneInString(text[:start-n]); start-n > 0 && wordRune(w) {
			return true
		}
	}
	if r, n := utf8.DecodeRuneInString(text[end:]); end < len(text) && strings.ContainsRune(joiners, r) {
		if w, _ := utf8.DecodeRuneInString(text[end+n:]); end+n < len(text) && wordRune(w) {
			return true
		}
	}
	return false
}

// oneCaseWord: s is one word with no capital letter or no small one.
func oneCaseWord(s string) bool {
	return !strings.ContainsFunc(s, unicode.IsSpace) && (strings.ToLower(s) == s || strings.ToUpper(s) == s)
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

// foldKey is the same rune for every case of r (foldEqual): the least
// of them.
func foldKey(r rune) rune {
	k := r
	for f := unicode.SimpleFold(r); f != r; f = unicode.SimpleFold(f) {
		k = min(k, f)
	}
	return k
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
