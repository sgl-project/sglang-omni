package meeting

import (
	"strings"
	"time"
	"unicode"
)

// A meeting that ends with hardly anything heard was probably started by
// mistake: Finish asks whether to keep it instead of filing it (Snapshot's
// Mistake). It asks only about a short one, so that a long recording whose
// live recognition failed is never one click from being deleted.
const (
	// MistakeWords is the least a meeting hears to end without the question.
	MistakeWords = 5
	// MistakeWithin is how long a meeting may have run for the question.
	MistakeWithin = 2 * time.Minute
)

// heardWords counts the words the meeting has heard so far, on every
// track: the sentences on the timeline and the live text after them.
func heardWords(segments []Segment, live map[Source]string) int {
	n := 0
	for _, s := range segments {
		n += countWords(s.Text)
	}
	for _, text := range live {
		n += countWords(text)
	}
	return n
}

// countWords counts text's words; a Chinese, Japanese or Korean character
// is a word of its own, as those languages put no space between words.
func countWords(text string) int {
	n := 0
	for _, field := range strings.FieldsFunc(text, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }) {
		run := false // in a run of letters that spaces delimit
		for _, r := range field {
			if unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul) {
				n++
				run = false
			} else if !run {
				n++
				run = true
			}
		}
	}
	return n
}
