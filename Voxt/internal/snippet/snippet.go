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
// another's replacement.
func Expand(text string, snippets []Snippet) string {
	type pattern struct {
		key  []rune
		word bool // starts or ends with a letter or digit of a Latin word
		text string
	}
	var patterns []pattern
	for _, s := range snippets {
		key := normalize(s.Trigger)
		if len(key) == 0 || strings.TrimSpace(s.Text) == "" {
			continue
		}
		patterns = append(patterns, pattern{key, latin(key[0]) || latin(key[len(key)-1]), s.Text})
	}
	if len(patterns) == 0 {
		return text
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
	var b strings.Builder
	last := 0 // runes written so far
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
			b.WriteString(string(runes[last:start]))
			b.WriteString(p.text)
			last, k, matched = end, k+len(p.key), true
			break
		}
		if !matched {
			k++
		}
	}
	if last == 0 {
		return text
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
