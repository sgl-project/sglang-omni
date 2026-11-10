package aliyun

import (
	"crypto/rand"
	"encoding/hex"
	"strings"
	"unicode"
	"unicode/utf8"
)

// transcript accumulates a stream's text: the sentences (inference) or
// items (realtime) in the order they began, each either final or still
// changing. Its String is the whole text so far, what EventPartial
// carries.
type transcript struct {
	segs []segment
}

type segment struct {
	id    string
	text  string
	final bool
}

// set records the text of segment id: a new one is appended, a known one
// replaced. id "" means the segment in progress: the last one unless it
// is final.
func (t *transcript) set(id, text string, final bool) {
	text = strings.TrimSpace(text)
	if id == "" {
		if n := len(t.segs); n > 0 && !t.segs[n-1].final {
			t.segs[n-1].text, t.segs[n-1].final = text, final
			return
		}
		t.segs = append(t.segs, segment{text: text, final: final})
		return
	}
	for i := range t.segs {
		if t.segs[i].id == id {
			t.segs[i].text, t.segs[i].final = text, final
			return
		}
	}
	t.segs = append(t.segs, segment{id: id, text: text, final: final})
}

func (t *transcript) String() string {
	parts := make([]string, 0, len(t.segs))
	for _, s := range t.segs {
		if s.text != "" {
			parts = append(parts, s.text)
		}
	}
	return joinText(parts)
}

// joinText joins sentences: with a space between two that meet in a
// spaced script (Latin, Cyrillic…), with nothing where either side is
// Chinese, Japanese or Korean, or full-width punctuation. (Voxt joins
// with a space everywhere, which puts spaces between Chinese sentences.)
func joinText(parts []string) string {
	var b strings.Builder
	for i, p := range parts {
		if i > 0 {
			last, _ := utf8.DecodeLastRuneInString(b.String())
			first, _ := utf8.DecodeRuneInString(p)
			if !wide(last) && !wide(first) && !unicode.IsSpace(last) {
				b.WriteByte(' ')
			}
		}
		b.WriteString(p)
	}
	return b.String()
}

// wide reports a rune of a script written without spaces, or full-width
// punctuation.
func wide(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul, unicode.Thai, unicode.Lao, unicode.Khmer, unicode.Myanmar) ||
		(r >= 0x3000 && r <= 0x303F) || (r >= 0xFF00 && r <= 0xFFEF)
}

// newID is a random 32-hex-digit id (a UUID without dashes, as
// DashScope's SDKs make task ids).
func newID() string {
	var b [16]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// clipRunes cuts s to at most n characters (the service counts
// characters, not bytes).
func clipRunes(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	r := []rune(s)
	return strings.TrimSpace(string(r[:n]))
}
