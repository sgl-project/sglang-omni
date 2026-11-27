// Package snippet is the user's snippets: text saved once (an email
// address, a link, a prompt) that a spoken trigger puts in its place. A
// dictation that is only a trigger is its snippet, not the language
// model's text; a trigger inside a dictation is replaced in the final
// text.
// Triggers match ignoring case, spaces and punctuation, so "my LinkedIn"
// matches "My linked in." as recognised. As the dictionary's aliases
// (dict.Replace), a trigger in a script that puts spaces between words
// (Latin, Cyrillic…) matches whole words only, and one in Chinese,
// Japanese or Korean anywhere, so it needs at least MinWordless
// characters. A trigger does not run over the end of a sentence ("my.
// Email me" holds no "my email"), and Traditional and Simplified Chinese
// match each other (lang.CanonicalChinese), as dictation may be written
// in either. Only a trigger
// the user said is replaced: one the language model wrote is its own
// wording. A snippet keeps its words as saved; only where it meets the
// text is it formatted (place). The snippets live in a document of
// their own (Key).
//
// Imports: lang.
package snippet

import (
	"encoding/json"
	"slices"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
)

// Key is the document the snippets are stored under (store.Documents).
const Key = "snippets:list"

// Limits of a snippet. MinWordless is the fewest characters of a
// trigger only in Chinese, Japanese or Korean: it matches inside longer
// words too, so a shorter one ("地址") would go off in everyday
// speech ("收货地址").
const (
	MaxTrigger  = 80
	MaxText     = 10000
	MinWordless = 3
)

// Snippet is a trigger and the text it stands for.
type Snippet struct {
	ID      string `json:"id"`
	Trigger string `json:"trigger"`
	Text    string `json:"text"`
}

// List is the stored snippets, in the user's order.
type List struct {
	Snippets []Snippet `json:"snippets"`
}

// Marshal and Unmarshal are the stored form.
func (l List) Marshal() ([]byte, error) { return json.Marshal(l) }

func Unmarshal(data []byte) (List, error) {
	var l List
	err := json.Unmarshal(data, &l)
	return l, err
}

// Problems of a snippet (Check).
const (
	ProblemNoTrigger = "no-trigger"
	ProblemNoText    = "no-text"
	ProblemLong      = "too-long"
	ProblemShort     = "too-short" // a Chinese trigger under MinWordless characters
	ProblemTaken     = "taken"     // another snippet has the same trigger
)

// Check says what keeps s from being saved in l ("" for nothing): s may
// be one of l's, by its ID.
func Check(l List, s Snippet) string {
	key := normalize(s.Trigger)
	switch {
	case len(key) == 0:
		return ProblemNoTrigger
	case tooShort(key):
		return ProblemShort
	case strings.TrimSpace(s.Text) == "":
		return ProblemNoText
	case len([]rune(s.Trigger)) > MaxTrigger || len([]rune(s.Text)) > MaxText:
		return ProblemLong
	}
	for _, o := range l.Snippets {
		if o.ID != s.ID && string(normalize(o.Trigger)) == string(key) {
			return ProblemTaken
		}
	}
	return ""
}

// tooShort says whether a trigger (normalized) is fewer than MinWordless
// characters, none of them of a script that spaces its words.
func tooShort(key []rune) bool {
	if len(key) >= MinWordless {
		return false
	}
	for _, r := range key {
		if wordRune(r) {
			return false
		}
	}
	return true
}

// Alone is a snippet's text when it is all of a dictation: without the
// white space at its ends.
func Alone(text string) string {
	if t := strings.TrimSpace(text); t != "" {
		return t
	}
	return text
}

// Whole is the snippet whose trigger is all of text, ignoring case,
// spaces and punctuation ("My email address." is "my email address").
func Whole(text string, snippets []Snippet) (Snippet, bool) {
	key := string(normalize(text))
	if key == "" {
		return Snippet{}, false
	}
	runes := []rune(text)
	if sentenceEnd(runes[slices.IndexFunc(runes, keep) : lastIndex(runes, keep)+1]) {
		return Snippet{}, false // more than one sentence
	}
	for _, s := range snippets {
		if string(normalize(s.Trigger)) == key {
			return s, true
		}
	}
	return Snippet{}, false
}

// Expand replaces each trigger in text with its snippet's text: the
// longest trigger at each place, from left to right, never inside
// another's replacement, each formatted where it meets the text (place).
func Expand(text string, snippets []Snippet) string { return expand("", text, snippets) }

// expand is Expand of text that follows prefix in the final text.
func expand(prefix, text string, snippets []Snippet) string {
	runes, found := find(text, snippets)
	return rebuild(text, runes, found, func(m match, before string) string {
		return place(m.snippet.Text, prefix+before, string(runes[m.end:]))
	})
}

// place is a snippet's text between before and after in the final text:
// without spaces at its ends (line breaks stay), its closing punctuation
// fitted to after's (fit), and a space where it meets Chinese with Latin
// letters or digits ("发到 a@b.com，"; spaced). Its words stay as saved.
func place(text, before, after string) string {
	t := strings.TrimFunc(text, func(r rune) bool { return r != '\n' && unicode.IsSpace(r) })
	if t == "" {
		return text
	}
	return spaced(before, fit(t, after), after)
}

// spaced is text with a space added at each end where it meets Chinese
// with Latin text, as the clean-up writes it ("a space between CJK and
// Latin text"). Japanese is left alone: it puts no spaces there.
func spaced(before, text, after string) string {
	if kana(before) || kana(text) || kana(after) {
		return text
	}
	last, _ := utf8.DecodeLastRuneInString(before)
	first, _ := utf8.DecodeRuneInString(text)
	end, _ := utf8.DecodeLastRuneInString(text)
	next, _ := utf8.DecodeRuneInString(after)
	if gap(last, first) {
		text = " " + text
	}
	if gap(end, next) {
		text += " "
	}
	return text
}

// gap says whether a and b, side by side, are Chinese and Latin text.
func gap(a, b rune) bool { return han(a) && latin(b) || latin(a) && han(b) }

func han(r rune) bool { return unicode.Is(unicode.Han, r) }

// latin: a letter or digit of a spaced script, or a sign that starts or
// ends such a word ("@name", "#tag", "$5", "100%").
func latin(r rune) bool { return wordRune(r) || strings.ContainsRune("@#$%", r) }

func kana(s string) bool {
	return strings.ContainsFunc(s, func(r rune) bool { return unicode.In(r, unicode.Hiragana, unicode.Katakana) })
}

// Punctuation fit looks at: before the sentence's own, a snippet's
// closing Chinese marks and Latin ",;:" give way, and so does a closing
// Latin mark the sentence repeats ("Best," "." is "Best.", "etc." "." is
// "etc.").
const (
	fullMarks = "。！？；，、：．"
	softMarks = ",;:"
	anyMarks  = fullMarks + ".!?;,:"
)

// fit is a snippet's text before rest, its closing punctuation fitted
// to rest's: "…优化。" before "，然后" is not "。，".
func fit(text, rest string) string {
	r, _ := utf8.DecodeRuneInString(rest)
	if rest == "" || !strings.ContainsRune(anyMarks, r) {
		return text
	}
	t := strings.TrimRight(text, fullMarks+softMarks)
	if r < utf8.RuneSelf && strings.HasSuffix(t, string(r)) {
		t = t[:len(t)-1]
	}
	if t == "" {
		return text
	}
	return t
}

// Open and Close mark a trigger in the text a language model cleans up
// (Mark), so that the model keeps it and Restore finds it again.
const (
	Open  = "⟦"
	Close = "⟧"
)

// Mark wraps each trigger in text, as Expand finds them, in Open and
// Close, and returns the snippets marked, in order.
func Mark(text string, snippets []Snippet) (string, []Snippet) {
	runes, found := find(text, snippets)
	marked := make([]Snippet, len(found))
	for i, m := range found {
		marked[i] = m.snippet
	}
	return rebuild(text, runes, found, func(m match, _ string) string {
		return Open + string(runes[m.start:m.end]) + Close
	}), marked
}

// Restore is a cleaned-up text with its snippets in place: a marked
// trigger is its snippet (by what the marks hold, or else by its place
// when the model kept as many marks as Mark made: marked), and triggers
// the model unmarked are found as Expand finds them. snippets are the
// ones the user said (Mark's): a trigger of another one that the model
// wrote stays as written. A mark that names none of them is left as its
// text; stray marks are dropped. Each snippet is formatted where it
// meets the text (place).
func Restore(text string, marked, snippets []Snippet) string {
	var b strings.Builder
	n := strings.Count(text, Open)
	for i := 0; ; i++ {
		open := strings.Index(text, Open)
		if open < 0 {
			break
		}
		closing := strings.Index(text[open:], Close)
		if closing < 0 {
			break
		}
		inner := text[open+len(Open) : open+closing]
		b.WriteString(expand(b.String(), unmark(text[:open]), snippets))
		rest := unmark(text[open+closing+len(Close):])
		switch s, ok := Whole(inner, snippets); {
		case ok:
			b.WriteString(place(s.Text, b.String(), rest))
		case n == len(marked) && i < len(marked):
			b.WriteString(place(marked[i].Text, b.String(), rest))
		default:
			b.WriteString(unmark(inner))
		}
		text = text[open+closing+len(Close):]
	}
	b.WriteString(expand(b.String(), unmark(text), snippets))
	return b.String()
}

func unmark(s string) string {
	return strings.NewReplacer(Open, "", Close, "").Replace(s)
}

// Unmark is text without Mark's marks.
func Unmark(text string) string { return unmark(text) }

// match is a trigger in a text: runes[start:end].
type match struct {
	start, end int
	snippet    Snippet
}

// find is text's triggers: the longest at each place, from left to
// right, not overlapping; a trigger in a script that spaces its words
// does not cut a word.
func find(text string, snippets []Snippet) ([]rune, []match) {
	type pattern struct {
		key  []rune
		word bool // starts or ends with a letter or digit of a spaced script
		s    Snippet
	}
	var patterns []pattern
	for _, s := range snippets {
		key := normalize(s.Trigger)
		if len(key) == 0 || strings.TrimSpace(s.Text) == "" {
			continue
		}
		patterns = append(patterns, pattern{key, wordRune(key[0]) || wordRune(key[len(key)-1]), s})
	}
	if len(patterns) == 0 {
		return nil, nil
	}
	sort.SliceStable(patterns, func(i, j int) bool { return len(patterns[i].key) > len(patterns[j].key) })
	runes := []rune(text)
	folded := foldScript(runes)
	// The kept characters, folded, and where each is in runes.
	var kept []rune
	var at []int
	for i, r := range runes {
		if keep(r) {
			kept, at = append(kept, unicode.ToLower(folded[i])), append(at, i)
		}
	}
	var found []match
	for k := 0; k < len(kept); {
		matched := false
		for _, p := range patterns {
			if k+len(p.key) > len(kept) || !equal(kept[k:k+len(p.key)], p.key) {
				continue
			}
			start, end := at[k], at[k+len(p.key)-1]+1
			if p.word && (start > 0 && wordRune(runes[start-1]) || end < len(runes) && wordRune(runes[end])) {
				continue // inside a longer word
			}
			if sentenceEnd(runes[start:end]) {
				continue // the end of one sentence and the start of the next
			}
			found = append(found, match{start, end, p.s})
			k, matched = k+len(p.key), true
			break
		}
		if !matched {
			k++
		}
	}
	return runes, found
}

// rebuild is text with each match replaced by with(match, what is
// before it in the new text).
func rebuild(text string, runes []rune, found []match, with func(match, string) string) string {
	if len(found) == 0 {
		return text
	}
	var b strings.Builder
	last := 0
	for _, m := range found {
		b.WriteString(string(runes[last:m.start]))
		b.WriteString(with(m, b.String()))
		last = m.end
	}
	b.WriteString(string(runes[last:]))
	return b.String()
}

// normalize is s's letters and digits, lowercase and in Simplified
// Chinese: what a trigger is matched by.
func normalize(s string) []rune {
	var out []rune
	for _, r := range foldScript([]rune(s)) {
		if keep(r) {
			out = append(out, unicode.ToLower(r))
		}
	}
	return out
}

// foldScript is runes with Traditional Chinese characters as Simplified,
// one for one (lang.CanonicalChinese); other runes stay as they are.
func foldScript(runes []rune) []rune {
	if !slices.ContainsFunc(runes, func(r rune) bool { return unicode.Is(unicode.Han, r) }) {
		return runes
	}
	if out := []rune(lang.CanonicalChinese(string(runes))); len(out) == len(runes) {
		return out
	}
	out := slices.Clone(runes) // a phrase changed length: character by character
	for i, r := range runes {
		if f := []rune(lang.CanonicalChinese(string(r))); len(f) == 1 {
			out[i] = f[0]
		}
	}
	return out
}

// sentenceEnd says whether runes hold the end of a sentence: 。！？, a
// line break, or . ! ? before a space ("v1.2" and "a.b" go on).
func sentenceEnd(runes []rune) bool {
	for i, r := range runes {
		switch {
		case strings.ContainsRune("。！？\n", r):
			return true
		case strings.ContainsRune(".!?", r) && i+1 < len(runes) && unicode.IsSpace(runes[i+1]):
			return true
		}
	}
	return false
}

// lastIndex is the index of the last rune of runes that f accepts, or -1.
func lastIndex(runes []rune, f func(rune) bool) int {
	for i := len(runes) - 1; i >= 0; i-- {
		if f(runes[i]) {
			return i
		}
	}
	return -1
}

func keep(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }

// wordRune: a letter or digit of a script that separates its words with
// spaces, where a match must not cut a word; not Chinese, Japanese or
// Korean, which match anywhere (dict's wordRune).
func wordRune(r rune) bool {
	if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
		return false
	}
	return !unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul)
}

func equal(a, b []rune) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
