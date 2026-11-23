package learn

import (
	"regexp"
	"slices"
	"strings"
	"unicode"

	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Voxt 1.x's limits for a correction worth learning.
const (
	maxEditRatio    = 0.8
	maxExactLength  = 2000 // past this the edit ratio is not worth computing
	maxTermRunes    = 48
	maxChineseRunes = 8
	maxTermWords    = 4
)

// Why an edit teaches nothing, for the log: never its text.
const (
	skipEmpty       = "empty text"
	skipIdentical   = "baseline and final text are identical"
	skipMeaningless = "changed fragment has no meaningful terms"
	skipNotFound    = "inserted text not found inside baseline snapshot"
	skipOutside     = "detected edit does not intersect inserted text"
	skipRatio       = "edit ratio exceeded limit"
	skipNotTerm     = "the changed words do not look like a term"
	skipNumbers     = "only numbers changed"
	skipInvalid     = "the correction is not a valid term"
)

// edit is what the user changed in the pasted text: the changed words
// before and after, and the edit's size relative to it; skip says why
// nothing is to be learned from it.
type edit struct {
	before, after, skip string
	ratio               float64
}

// suggestionFromEdit is judgeEdit and the term rules: one suggestion, or
// why none. Text stays in the result, never in the reason.
func suggestionFromEdit(inserted, baseline, final string) (Suggestion, string) {
	e := judgeEdit(inserted, baseline, final)
	if e.skip != "" {
		return Suggestion{}, e.skip
	}
	term, heard := directTerm(e.before, e.after)
	if term == "" {
		return Suggestion{}, skipNotTerm
	}
	beforeBare, termBare := withoutNumbers(heard), withoutNumbers(term)
	if fold(beforeBare) == fold(termBare) || !strings.ContainsFunc(termBare, unicode.IsLetter) {
		return Suggestion{}, skipNumbers
	}
	if dict.Check(store.Term{Term: term}) != nil {
		return Suggestion{}, skipInvalid
	}
	return Suggestion{Term: term, Heard: []string{heard}, Source: FromEdit}, ""
}

// chineseNumerals count as digits when a number or a time changed.
const chineseNumerals = "〇零一二三四五六七八九十百千万亿两半"

func withoutNumbers(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsDigit(r) || strings.ContainsRune(chineseNumerals, r) {
			return -1
		}
		return r
	}, s)
}

// judgeEdit compares the text at the insertion's baseline and at the end
// of the watch, as Voxt 1.x's makeLearningRequest: the inserted text's
// line first, then the whole text and the line it is in.
func judgeEdit(rawInserted, rawBaseline, rawFinal string) edit {
	inserted, baseline, final := strings.TrimSpace(rawInserted), strings.TrimSpace(rawBaseline), strings.TrimSpace(rawFinal)
	if inserted == "" || baseline == "" || final == "" {
		return edit{skip: skipEmpty}
	}
	if baseline == final {
		return edit{skip: skipIdentical}
	}
	if ins, found := findInserted(baseline, inserted); found {
		baselineLine := lineScope(baseline, ins, inserted)
		if finalScoped := finalScope(inserted, baselineLine, final); finalScoped != "" {
			return checkEdit(inserted, baselineLine, finalScoped)
		}
	}
	whole := checkEdit(inserted, baseline, final)
	if whole.skip == "" {
		return whole
	}
	if ins, found := findInserted(baseline, inserted); found {
		baselineRunes := []rune(baseline)
		start, end := lineBounds(baselineRunes, ins)
		baselineLine := strings.TrimSpace(string(baselineRunes[start:end]))
		if finalLine, ok := bestLine(inserted, baselineLine, final); ok && baselineLine != "" {
			if line := checkEdit(inserted, baselineLine, finalLine); line.skip == "" {
				return line
			}
		}
	}
	return whole
}

// checkEdit is the checks of one scope: there is an edit of the inserted
// text, more than writing on, not a rewrite.
func checkEdit(inserted, baseline, final string) edit {
	w := changeWindow(baseline, final)
	if !hasNewWord(w.before, w.after) {
		return edit{skip: skipMeaningless}
	}
	ins, found := findInserted(baseline, inserted)
	if !found {
		return edit{skip: skipNotFound}
	}
	if !touchesInserted(w, ins) || isPureAppend(baseline, final) {
		return edit{skip: skipOutside}
	}
	ratio := editRatio(inserted, baseline, final)
	if ratio > maxEditRatio {
		return edit{skip: skipRatio}
	}
	return edit{before: w.before, after: w.after, ratio: ratio}
}

func touchesInserted(w window, ins [2]int) bool {
	if w.baseStart == w.baseEnd {
		return w.baseStart > ins[0] && w.baseStart < ins[1]
	}
	return max(w.baseStart, ins[0]) < min(w.baseEnd, ins[1])
}

func isPureAppend(baseline, final string) bool {
	if !strings.HasPrefix(final, baseline) || len(final) <= len(baseline) {
		return false
	}
	base := []rune(baseline)
	suffix := []rune(final[len(baseline):])
	return !(wordRune(base[len(base)-1]) && wordRune(suffix[0]))
}

// editRatio is the edit distance between the two texts over the inserted
// text's length.
func editRatio(inserted, baseline, final string) float64 {
	base := []rune(strings.ToLower(strings.TrimSpace(baseline)))
	end := []rune(strings.ToLower(strings.TrimSpace(final)))
	if max(len(base), len(end)) > maxExactLength {
		return 1
	}
	return float64(levenshtein(base, end)) / float64(len([]rune(strings.ToLower(strings.TrimSpace(inserted)))))
}

func levenshtein(a, b []rune) int {
	previous := make([]int, len(b)+1)
	for j := range previous {
		previous[j] = j
	}
	for i := range a {
		current := make([]int, len(b)+1)
		current[0] = i + 1
		for j := range b {
			cost := 1
			if a[i] == b[j] {
				cost = 0
			}
			current[j+1] = min(previous[j+1]+1, current[j]+1, previous[j]+cost)
		}
		previous = current
	}
	return previous[len(b)]
}

// window is the part of the texts that differs, widened to whole words.
type window struct {
	baseStart, baseEnd int // runes of the baseline
	before, after      string
}

// changeWindow compares the texts by their shared prefix and suffix.
func changeWindow(baseline, final string) window {
	base, end := []rune(baseline), []rune(final)
	prefix := 0
	for prefix < min(len(base), len(end)) && base[prefix] == end[prefix] {
		prefix++
	}
	suffix := 0
	for suffix < min(len(base), len(end))-prefix && base[len(base)-1-suffix] == end[len(end)-1-suffix] {
		suffix++
	}
	baseStart, baseEnd := expandToWord(base, prefix, max(prefix, len(base)-suffix))
	finalStart, finalEnd := expandToWord(end, prefix, max(prefix, len(end)-suffix))
	return window{baseStart: baseStart, baseEnd: baseEnd,
		before: strings.TrimSpace(string(base[baseStart:baseEnd])), after: strings.TrimSpace(string(end[finalStart:finalEnd]))}
}

// expandToWord widens [start, end) over the word its first rune is in; an
// empty range takes the word it touches.
func expandToWord(rs []rune, start, end int) (int, int) {
	lower, upper := min(start, len(rs)), end
	anchor := -1
	switch {
	case lower < upper:
		anchor = lower
	case lower < len(rs) && wordRune(rs[lower]):
		anchor, upper = lower, lower+1
	case lower > 0 && wordRune(rs[lower-1]):
		lower--
		anchor, upper = lower, lower+1
	}
	if anchor < 0 || !wordRune(rs[anchor]) {
		return lower, upper
	}
	for lower > 0 && wordRune(rs[lower-1]) {
		lower--
	}
	for upper < len(rs) && wordRune(rs[upper]) {
		upper++
	}
	return lower, upper
}

// wordRune is a letter or digit of any script (Han too) or a character
// that joins a word: 1.x's change window.
func wordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.IsMark(r) || strings.ContainsRune("._+-'", r)
}

var (
	latinWord = regexp.MustCompile(`[A-Za-z0-9]+(?:[._+\-'][A-Za-z0-9]+)*`)
	hanRun    = regexp.MustCompile(`\p{Han}{2,12}`)
)

// hasNewWord: the changed text after the edit has a word the changed
// text before did not; only punctuation or spacing changed otherwise.
func hasNewWord(before, after string) bool {
	old := vocabulary(before)
	return slices.ContainsFunc(vocabulary(after), func(word string) bool { return !slices.Contains(old, word) })
}

// vocabulary is what 1.x calls candidate words: Latin words or numbers of
// 3–32 characters with a letter, Han runs of 2–12.
func vocabulary(text string) []string {
	var out []string
	for _, word := range latinWord.FindAllString(text, -1) {
		if n := len([]rune(word)); n >= 3 && n <= 32 && strings.ContainsFunc(word, unicode.IsLetter) {
			out = append(out, word)
		}
	}
	return append(out, hanRun.FindAllString(text, -1)...)
}

var promptMark = regexp.MustCompile(`^\s*>+\s*`)

// directTerm returns the changed words as a term and what was heard, or
// "" when they do not look like one (1.x's direct candidate).
func directTerm(before, after string) (term, heard string) {
	clean := func(s string) string {
		s = strings.TrimSpace(promptMark.ReplaceAllString(s, ""))
		return strings.Trim(s, ".,;:!?，。；：！？\"'()[]{}<>")
	}
	heard, term = clean(before), clean(after)
	if heard == "" || term == "" || fold(heard) == fold(term) || !termLike(heard) || !termLike(term) {
		return "", ""
	}
	return term, heard
}

// termLike is 1.x's rule: 2–48 characters on one line, no sentence
// punctuation, at most four words, with Latin letters or digits, or a
// single word of at most eight Han characters.
func termLike(text string) bool {
	rs := []rune(text)
	if len(rs) < 2 || len(rs) > maxTermRunes || strings.ContainsAny(text, "\n。！？；") {
		return false
	}
	parts := strings.Fields(text)
	if len(parts) == 0 || len(parts) > maxTermWords {
		return false
	}
	if strings.ContainsFunc(text, isASCIIWord) {
		return true
	}
	return strings.ContainsFunc(text, isIdeograph) && len(parts) == 1 && len(rs) <= maxChineseRunes
}

func isASCIIWord(r rune) bool {
	return r < unicode.MaxASCII && (unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-')
}

func isIdeograph(r rune) bool { return unicode.Is(unicode.Ideographic, r) }

// incompleteEdit says whether the text still looks in the middle of a
// replacement: words deleted and none written in their place, anywhere
// in the text (1.x's shouldContinueObservingForPotentialReplacement).
func incompleteEdit(baseline, final string) bool {
	w := changeWindow(baseline, final)
	meaningful := func(s string) bool {
		return strings.ContainsFunc(s, func(r rune) bool { return unicode.IsLetter(r) || unicode.IsNumber(r) })
	}
	return (meaningful(w.before) && !meaningful(w.after)) || hasDeletionGroup(baseline, final)
}

// hasDeletionGroup finds a stretch of baseline words with nothing in its
// place in the end's, comparing the sequences of words.
func hasDeletionGroup(baseline, final string) bool {
	base, end := semanticWords(baseline), semanticWords(final)
	table := make([][]int, len(base)+1)
	for i := range table {
		table[i] = make([]int, len(end)+1)
	}
	for i := len(base) - 1; i >= 0; i-- {
		for j := len(end) - 1; j >= 0; j-- {
			if base[i] == end[j] {
				table[i][j] = table[i+1][j+1] + 1
			} else {
				table[i][j] = max(table[i+1][j], table[i][j+1])
			}
		}
	}
	var pairs [][2]int
	i, j := 0, 0
	for i < len(base) && j < len(end) {
		switch {
		case base[i] == end[j]:
			pairs = append(pairs, [2]int{i, j})
			i++
			j++
		case table[i+1][j] >= table[i][j+1]:
			i++
		default:
			j++
		}
	}
	lastBase, lastEnd := -1, -1
	for _, p := range append(pairs, [2]int{len(base), len(end)}) {
		if lastBase+1 < p[0] && lastEnd+1 >= p[1] {
			return true
		}
		lastBase, lastEnd = p[0], p[1]
	}
	return false
}

// semanticWords are the text's words: ASCII words, single ideographs,
// and runs of everything else, without spacing or punctuation.
func semanticWords(text string) []string {
	rs := []rune(text)
	var out []string
	isBreak := func(r rune) bool { return unicode.IsSpace(r) || unicode.IsPunct(r) || unicode.IsSymbol(r) }
	for i := 0; i < len(rs); {
		if isBreak(rs[i]) {
			i++
			continue
		}
		start := i
		i++
		switch {
		case isASCIIWord(rs[start]):
			for i < len(rs) && isASCIIWord(rs[i]) {
				i++
			}
		case isIdeograph(rs[start]):
		default:
			for i < len(rs) && !isBreak(rs[i]) && !isASCIIWord(rs[i]) && !isIdeograph(rs[i]) {
				i++
			}
		}
		out = append(out, string(rs[start:i]))
	}
	return out
}
