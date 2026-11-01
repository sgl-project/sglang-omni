package compose

import (
	"strings"
	"unicode"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/lang"
)

// TargetLanguages are the languages Translate offers, in the menu's
// order (Voxt's list, with Simplified and Traditional Chinese apart).
var TargetLanguages = []string{
	"en", "zh-Hans", "zh-Hant", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
	"nl", "tr", "pl", "uk", "ar", "he", "hi", "th", "vi", "id", "ms",
}

// LanguageName names a language in a template set's language: the lang
// table's name, with "zh-Hans" and "zh-Hant" as Simplified and
// Traditional Chinese.
func LanguageName(code, set string) string {
	zh := set == Chinese || strings.HasPrefix(strings.ToLower(set), "zh")
	switch strings.ToLower(strings.TrimSpace(code)) {
	case "zh-hans", "zh-cn", "zh-sg":
		if zh {
			return "简体中文"
		}
		return "Simplified Chinese"
	case "zh-hant", "zh-tw", "zh-hk", "zh-mo":
		if zh {
			return "繁体中文"
		}
		return "Traditional Chinese"
	}
	if l, ok := lang.Lookup(code); ok {
		return l.NameIn(set)
	}
	return code
}

// Untranslated reports whether a translation into target looks like it
// was not made, and is worth asking again (Input.Strict):
//
//   - it is the source again, ignoring case, spaces and punctuation;
//   - or its letters are mostly not in the target's script while the
//     source's are not either (Chinese asked, Latin text back; English
//     asked, Chinese back). Scripts cannot tell English from French, so
//     Latin into Latin is judged by the first rule only.
//
// A source already in the target's script (Chinese into Chinese) is
// never judged untranslated: the model may rightly return it.
func Untranslated(source, result, target string) bool {
	src, res := fold(source), fold(result)
	if src == "" || res == "" {
		return false
	}
	is := scriptOf(target)
	if is == nil {
		return src == res
	}
	srcShare, srcLetters := share(source, is)
	resShare, resLetters := share(result, is)
	if srcLetters > 0 && srcShare >= 0.5 && !isLatin(target) {
		return false // the source is in the target's script already
	}
	if src == res {
		return true
	}
	return resLetters > 0 && resShare < 0.5 && srcShare < 0.5
}

// fold is s without case, spaces and punctuation.
func fold(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) || unicode.IsPunct(r) || unicode.IsSymbol(r) {
			return -1
		}
		return unicode.ToLower(r)
	}, s)
}

// scriptOf is a test of the letters a target language is written in; nil
// when unknown.
func scriptOf(target string) func(rune) bool {
	code := strings.ToLower(target)
	script := ""
	switch {
	case strings.HasPrefix(code, "zh"), code == "yue":
		script = "Hani"
	default:
		if l, ok := lang.Lookup(target); ok {
			script = l.Script
		}
	}
	switch script {
	case "Hans", "Hant", "Hani":
		return func(r rune) bool { return unicode.Is(unicode.Han, r) }
	case "Jpan":
		return func(r rune) bool {
			return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana)
		}
	case "Kore":
		return func(r rune) bool { return unicode.In(r, unicode.Hangul, unicode.Han) }
	case "":
		return nil
	}
	if t, ok := unicode.Scripts[scriptNames[script]]; ok {
		return func(r rune) bool { return unicode.Is(t, r) }
	}
	return nil
}

func isLatin(target string) bool {
	l, ok := lang.Lookup(target)
	return ok && l.Script == "Latn" && !strings.HasPrefix(strings.ToLower(target), "zh")
}

// scriptNames maps ISO 15924 codes to package unicode's script names.
var scriptNames = map[string]string{
	"Latn": "Latin", "Cyrl": "Cyrillic", "Arab": "Arabic", "Hebr": "Hebrew", "Deva": "Devanagari",
	"Thai": "Thai", "Grek": "Greek", "Hang": "Hangul", "Beng": "Bengali", "Taml": "Tamil",
	"Telu": "Telugu", "Gujr": "Gujarati", "Guru": "Gurmukhi", "Knda": "Kannada", "Mlym": "Malayalam",
	"Khmr": "Khmer", "Laoo": "Lao", "Mymr": "Myanmar", "Geor": "Georgian", "Armn": "Armenian",
	"Ethi": "Ethiopic", "Sinh": "Sinhala",
}

// share is the part of s's words that is in the script, and how many
// words s has. A word is a run of letters of an alphabet, or one
// ideograph, kana or Hangul syllable (a Chinese name in Latin text weighs
// what an English one in Chinese text does).
func share(s string, in func(rune) bool) (float64, int) {
	n, k := 0, 0
	inWord := false
	for _, r := range s {
		switch {
		case !unicode.IsLetter(r):
			inWord = false
		case unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul):
			inWord = false
			n++
			if in(r) {
				k++
			}
		case !inWord:
			inWord = true
			n++
			if in(r) {
				k++
			}
		}
	}
	if n == 0 {
		return 0, 0
	}
	return float64(k) / float64(n), n
}
