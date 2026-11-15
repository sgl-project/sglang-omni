package text

import (
	"strings"
	"unicode"
)

// The language of a text decides the glyphs of the scripts it writes,
// which some fonts draw apart by language, as the Han characters Chinese
// and Japanese share, and the fonts the system falls back to for them.
// Han characters in a text of a language that does not write them, as a
// Chinese name in an English sentence, keep the glyphs the system picks
// for them by the user's settings, as they had without a language: the
// language says nothing of them, and an English app shown to a reader of
// Japanese should not draw a Japanese name with Chinese glyphs. Browsers
// do the same (Chromium's LayoutLocale::LocaleForHan). Core Text does it
// by itself; DirectWrite and Pango are told (cjkRuns).

// writesHan reports whether a language writes Han characters: Chinese,
// with its languages of other codes, Japanese and Korean.
func writesHan(lang string) bool {
	base, _, _ := strings.Cut(strings.ToLower(lang), "-")
	base, _, _ = strings.Cut(base, "_")
	switch base {
	case "zh", "ja", "ko", "yue", "cmn", "wuu", "hak", "nan", "gan", "hsn", "lzh", "cjy", "cpx", "czh", "czo", "mnp", "cdo":
		return true
	}
	return false
}

// isCJK reports whether r is a Han character, or a mark of punctuation
// or a full-width form that the languages writing Han draw apart.
func isCJK(r rune) bool {
	switch {
	case r < 0x2E80:
		return false
	case r >= 0x3000 && r <= 0x303F, r >= 0xFF00 && r <= 0xFFEF:
		return true
	}
	return unicode.Is(unicode.Han, r)
}

// langRun is a run of runes of a text, from start to end, in one language
// as the text engine is to be told it.
type langRun struct {
	start, end int
	lang       string
}

// langRuns returns the runs of a text whose language the engine must be
// told apart from base, the language of its style, given the language of
// the spans. A run of "" is in the engine's own language, as Han
// characters are in a language that does not write them. It returns nil
// when every rune is in base.
func langRuns(text []rune, base string, spans []Span) []langRun {
	if (base == "" || writesHan(base)) && !spansHaveLang(spans) {
		return nil
	}
	var out []langRun
	add := func(i int, lang string) {
		if n := len(out); n > 0 && out[n-1].end == i && out[n-1].lang == lang {
			out[n-1].end = i + 1
			return
		}
		out = append(out, langRun{i, i + 1, lang})
	}
	span, from := 0, 0
	for i, r := range text {
		lang := base
		for span < len(spans) && max(spans[span].End, from) <= i {
			from = max(spans[span].End, from)
			span++
		}
		if span < len(spans) && spans[span].Lang != "" {
			lang = spans[span].Lang
		}
		if lang != "" && isCJK(r) && !writesHan(lang) {
			lang = ""
		}
		if lang != base {
			add(i, lang)
		}
	}
	return out
}

func spansHaveLang(spans []Span) bool {
	for _, sp := range spans {
		if sp.Lang != "" {
			return true
		}
	}
	return false
}
