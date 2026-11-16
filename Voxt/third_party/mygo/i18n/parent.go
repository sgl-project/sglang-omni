package i18n

import (
	"strings"

	"golang.org/x/text/language"
)

// Parent returns the locale tag inherits what it lacks from, by the parent
// locales of CLDR, or "" when it is the root: "en-001" for "en-AU", "en"
// for "en-001", "zh-Hant-HK" for "zh-Hant-MO", "no" for "nb", and the
// tag without its last subtag otherwise ("de" for "de-CH"). A language
// written in a script other than its usual one inherits from nothing
// (CLDR's nonlikelyScript rule): the parent of "zh-Hant" or "sr-Latn" is
// the root, not "zh" or "sr", whose texts are in another script.
// Extensions, as "-u-ca-buddhist", are dropped first.
func Parent(tag string) string {
	t := stripExtensions(Normalize(tag))
	if t == "" {
		return ""
	}
	if i, ok := search(parentChildren[:], t); ok {
		return parentLocales[i]
	}
	lang, rest, ok := strings.Cut(t, "-")
	if !ok {
		return ""
	}
	if len(rest) == 4 && isLetters(rest) {
		// language-Script: the root unless the script is the language's
		// usual one.
		if likely, conf := language.Make(lang).Script(); conf != language.No && likely.String() != rest {
			return ""
		}
	}
	return t[:strings.LastIndexByte(t, '-')]
}

func isLetters(s string) bool {
	for i := 0; i < len(s); i++ {
		if c := s[i] | 0x20; c < 'a' || c > 'z' {
			return false
		}
	}
	return true
}

// chain returns tag and its parents, nearest first, without the root.
func chain(tag string) []string {
	var out []string
	for t := stripExtensions(Normalize(tag)); t != "" && len(out) < 8; t = Parent(t) {
		out = append(out, t)
	}
	return out
}
