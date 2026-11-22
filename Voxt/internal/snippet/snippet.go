// Package snippet is the user's snippets: text saved once (an email
// address, a link, a prompt) that a spoken trigger puts in its place. A
// dictation that is only a trigger is its snippet, without the language
// model; a trigger inside a dictation is replaced in the final text.
// Triggers match ignoring case, spaces and punctuation, so "my LinkedIn"
// matches "My linked in." as recognised; a trigger in Latin letters or
// digits matches whole words only. The snippets live in a document of
// their own (Key).
//
// Imports: nothing internal.
package snippet

import (
	"encoding/json"
	"sort"
	"strings"
	"unicode"
)

// Key is the document the snippets are stored under (store.Documents).
const Key = "snippets:list"

// Limits of a snippet.
const (
	MaxTrigger = 80
	MaxText    = 10000
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
	ProblemTaken     = "taken" // another snippet has the same trigger
)

// Check says what keeps s from being saved in l ("" for nothing): s may
// be one of l's, by its ID.
func Check(l List, s Snippet) string {
	key := normalize(s.Trigger)
	switch {
	case len(key) == 0:
		return ProblemNoTrigger
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

// Whole is the snippet whose trigger is all of text, ignoring case,
// spaces and punctuation ("My email address." is "my email address").
func Whole(text string, snippets []Snippet) (Snippet, bool) {
	key := string(normalize(text))
	if key == "" {
		return Snippet{}, false
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
// another's replacement. A snippet ending in punctuation before more
// (fit) loses its own.
func Expand(text string, snippets []Snippet) string {
	runes, found := find(text, snippets)
	return rebuild(text, runes, found, func(m match) string { return fit(m.snippet.Text, string(runes[m.end:])) })
}

// marks are the punctuation fit does not double.
const marks = "。．！？；，、：.!?;,:"

// fit is a snippet's text before rest: without its last punctuation
// marks when rest starts with one, so that "…优化。" before "，然后" is
// not "。，".
func fit(text, rest string) string {
	if rest == "" || !strings.ContainsRune(marks, []rune(rest)[0]) {
		return text
	}
	if t := strings.TrimRight(text, marks); t != "" {
		return t
	}
	return text
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
	return rebuild(text, runes, found, func(m match) string {
		return Open + string(runes[m.start:m.end]) + Close
	}), marked
}

// Restore is a cleaned-up text with its snippets in place: a marked
// trigger is its snippet (by what the marks hold, or else by its place
// when the model kept as many marks as Mark made: marked), and triggers
// the model unmarked are found as Expand finds them. A mark that names
// no snippet is left as its text; stray marks are dropped.
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
		b.WriteString(Expand(unmark(text[:open]), snippets))
		rest := unmark(text[open+closing+len(Close):])
		switch s, ok := Whole(inner, snippets); {
		case ok:
			b.WriteString(fit(s.Text, rest))
		case n == len(marked) && i < len(marked):
			b.WriteString(fit(marked[i].Text, rest))
		default:
			b.WriteString(unmark(inner))
		}
		text = text[open+closing+len(Close):]
	}
	b.WriteString(Expand(unmark(text), snippets))
	return b.String()
}

func unmark(s string) string {
	return strings.NewReplacer(Open, "", Close, "").Replace(s)
}

// match is a trigger in a text: runes[start:end].
type match struct {
	start, end int
	snippet    Snippet
}

// find is text's triggers: the longest at each place, from left to
// right, not overlapping; a trigger in Latin letters or digits does not
// cut a word.
func find(text string, snippets []Snippet) ([]rune, []match) {
	type pattern struct {
		key  []rune
		word bool // starts or ends with a letter or digit of a Latin word
		s    Snippet
	}
	var patterns []pattern
	for _, s := range snippets {
		key := normalize(s.Trigger)
		if len(key) == 0 || strings.TrimSpace(s.Text) == "" {
			continue
		}
		patterns = append(patterns, pattern{key, latin(key[0]) || latin(key[len(key)-1]), s})
	}
	if len(patterns) == 0 {
		return nil, nil
	}
	sort.SliceStable(patterns, func(i, j int) bool { return len(patterns[i].key) > len(patterns[j].key) })
	runes := []rune(text)
	// The kept characters, folded, and where each is in runes.
	var kept []rune
	var at []int
	for i, r := range runes {
		if keep(r) {
			kept, at = append(kept, unicode.ToLower(r)), append(at, i)
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
			if p.word && (start > 0 && latin(runes[start-1]) || end < len(runes) && latin(runes[end])) {
				continue // inside a longer word
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

// rebuild is text with each match replaced by with(match).
func rebuild(text string, runes []rune, found []match, with func(match) string) string {
	if len(found) == 0 {
		return text
	}
	var b strings.Builder
	last := 0
	for _, m := range found {
		b.WriteString(string(runes[last:m.start]))
		b.WriteString(with(m))
		last = m.end
	}
	b.WriteString(string(runes[last:]))
	return b.String()
}

// normalize is s's letters and digits, lowercase: what a trigger is
// matched by.
func normalize(s string) []rune {
	var out []rune
	for _, r := range s {
		if keep(r) {
			out = append(out, unicode.ToLower(r))
		}
	}
	return out
}

func keep(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }

// latin reports whether r is part of a word written with spaces between
// words: a Latin letter or a digit, where a match must not cut a word.
func latin(r rune) bool {
	return unicode.IsDigit(r) || unicode.Is(unicode.Latin, r)
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
