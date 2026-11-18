package meeting

import (
	"strings"
	"time"
	"unicode"
)

const (
	// echoSlack widens a mic segment's time span when looking for the system
	// speech it may have picked up: the two tracks cut their segments at
	// different moments (#3982).
	echoSlack = 2 * time.Second
	// echoRun is how many words in a row a mic segment must share with the
	// system speech for them to count as heard from the speakers.
	echoRun = 3
	// echoShare is the part of a mic segment's words that must be in such
	// runs. In the 20-minute call of #3982, 180 of 274 mic segments reached
	// it against the system speech of their time, and 0–8 against the system
	// speech moved one to three minutes away.
	echoShare = 0.5
)

// isEcho reports whether a mic segment's text is mostly the system speech
// near it, picked up from the speakers, rather than the user's own words.
func isEcho(mic, system string) bool {
	m := echoWords(mic)
	s := echoWords(system)
	if len(m) < echoRun || len(s) < echoRun {
		return false
	}
	runs := map[string]bool{}
	for i := 0; i+echoRun <= len(s); i++ {
		runs[strings.Join(s[i:i+echoRun], " ")] = true
	}
	shared := make([]bool, len(m))
	for i := 0; i+echoRun <= len(m); i++ {
		if runs[strings.Join(m[i:i+echoRun], " ")] {
			for j := i; j < i+echoRun; j++ {
				shared[j] = true
			}
		}
	}
	n := 0
	for _, ok := range shared {
		if ok {
			n++
		}
	}
	return float64(n) >= echoShare*float64(len(m))
}

// echoWords splits text into lower-case words: each Han, kana or Hangul
// character is a word of its own; other letters and digits form words
// between spaces and punctuation.
func echoWords(text string) []string {
	var out []string
	var word []rune
	flush := func() {
		if len(word) > 0 {
			out = append(out, string(word))
			word = word[:0]
		}
	}
	for _, r := range strings.ToLower(text) {
		switch {
		case unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul):
			flush()
			out = append(out, string(r))
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			word = append(word, r)
		default:
			flush()
		}
	}
	flush()
	return out
}

// systemTextNear joins the text of the system segments within echoSlack of
// [start, end].
func systemTextNear(segs []Segment, start, end time.Duration) string {
	var b strings.Builder
	for _, s := range segs {
		if s.Source == SourceSystem && s.End >= start-echoSlack && s.Start <= end+echoSlack {
			b.WriteString(s.Text)
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// dropEchoes is mic without the segments that are echoes of system.
func dropEchoes(mic, system []Segment) []Segment {
	out := mic[:0:0]
	for _, s := range mic {
		if !isEcho(s.Text, systemTextNear(system, s.Start, s.End)) {
			out = append(out, s)
		}
	}
	return out
}
