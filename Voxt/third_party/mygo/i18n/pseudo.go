package i18n

import (
	"strings"
	"unicode"
)

// The pseudo locales, which test an app's layouts without translations, as
// Android's: their messages are the source language's, transformed.
const (
	// PseudoAccents is en-XA: the letters accented, the text a third
	// longer and in brackets, "[Šéļéçţ ƒîļéš one two]", to find strings
	// that are not in the catalog, are cut, or are put together from
	// pieces.
	PseudoAccents = "en-XA"
	// PseudoBidi is ar-XB: the source's words shown right to left, as
	// Arabic or Hebrew text lays out, with the layout of the app right to
	// left too (Direction is RightToLeft), to find what does not mirror.
	PseudoBidi = "ar-XB"
)

// IsPseudo reports whether tag is a pseudo locale: en-XA or ar-XB.
func IsPseudo(tag string) bool {
	switch stripExtensions(Normalize(tag)) {
	case PseudoAccents, PseudoBidi:
		return true
	}
	return false
}

// pseudoMethod transforms the text of a message, and what its arguments
// write.
type pseudoMethod interface {
	text(s string) string
	placeholder(s string) string
	// wrap transforms the formatted message.
	wrap(s string) string
}

// pseudoFor returns the method of the pseudo locale tag, or nil.
func pseudoFor(tag string) pseudoMethod {
	switch stripExtensions(Normalize(tag)) {
	case PseudoAccents:
		return &accents{}
	case PseudoBidi:
		return bidi{}
	}
	return nil
}

// accents is en-XA.
type accents struct{ length int }

// accented are the letters of en-XA, from Android's pseudolocalizer.
var accented = map[rune]string{
	'a': "å", 'b': "ƀ", 'c': "ç", 'd': "ð", 'e': "é", 'f': "ƒ", 'g': "ĝ", 'h': "ĥ", 'i': "î",
	'j': "ĵ", 'k': "ķ", 'l': "ļ", 'm': "ɱ", 'n': "ñ", 'o': "ö", 'p': "þ", 'q': "ǫ", 'r': "ŕ",
	's': "š", 't': "ţ", 'u': "û", 'v': "ṽ", 'w': "ŵ", 'x': "ẋ", 'y': "ý", 'z': "ž",
	'A': "Å", 'B': "Ɓ", 'C': "Ç", 'D': "Ð", 'E': "É", 'F': "Ƒ", 'G': "Ĝ", 'H': "Ĥ", 'I': "Î",
	'J': "Ĵ", 'K': "Ķ", 'L': "Ļ", 'M': "Ṁ", 'N': "Ñ", 'O': "Ö", 'P': "Þ", 'Q': "Ǫ", 'R': "Ŕ",
	'S': "Š", 'T': "Ţ", 'U': "Û", 'V': "Ṽ", 'W': "Ŵ", 'X': "Ẋ", 'Y': "Ý", 'Z': "Ž",
}

func (a *accents) text(s string) string {
	var b strings.Builder
	for _, r := range s {
		if x, ok := accented[r]; ok {
			b.WriteString(x)
		} else {
			b.WriteRune(r)
		}
		a.length++
	}
	return b.String()
}

func (a *accents) placeholder(s string) string { return s }

// padWords lengthen en-XA's messages, as Android's do.
var padWords = strings.Fields("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty")

// wrap brackets the message and pads it by 40% of its text, rounded up,
// with words, as translations run longer than English.
func (a *accents) wrap(s string) string {
	pad := (a.length*2 + 4) / 5
	var b strings.Builder
	b.WriteString("[")
	b.WriteString(s)
	for i, n := 0, 0; n < pad; i++ {
		w := padWords[i%len(padWords)]
		b.WriteString(" " + w)
		n += 1 + len(w)
	}
	b.WriteString("]")
	return b.String()
}

// bidi is ar-XB.
type bidi struct{}

const (
	rlm = "\u200f" // right-to-left mark
	rlo = "\u202e" // right-to-left override
	pdf = "\u202c" // pop directional formatting
)

// text shows each word of s, each run of characters other than spaces,
// right to left, as Android's ar-XB does: an override reverses its
// letters, and marks around it set it in a right-to-left run.
func (bidi) text(s string) string {
	var b strings.Builder
	for len(s) > 0 {
		end := strings.IndexFunc(s, func(r rune) bool { return !unicode.IsSpace(r) })
		if end < 0 {
			end = len(s)
		}
		b.WriteString(s[:end])
		s = s[end:]
		if s == "" {
			break
		}
		end = strings.IndexFunc(s, unicode.IsSpace)
		if end < 0 {
			end = len(s)
		}
		b.WriteString(rlm + rlo + s[:end] + pdf + rlm)
		s = s[end:]
	}
	return b.String()
}

// placeholder keeps what an argument wrote as it is, in a right-to-left
// run.
func (bidi) placeholder(s string) string { return rlm + s + rlm }

func (bidi) wrap(s string) string { return s }
