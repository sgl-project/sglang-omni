// Package i18n reads the languages people read and picks, of those an app
// is written in, the one that suits them best, and says an app's messages
// in it.
//
// A language is a BCP 47 tag, as "en", "en-GB", "zh-Hant" or "sr-Latn-RS".
// Normalize reads the other forms desktops report, as the POSIX locales of
// Linux ("zh_CN.UTF-8", "sr_RS@latin"). Match compares tags by the
// distances of the Unicode CLDR, through golang.org/x/text/language, so
// that zh-TW gets zh-Hant and not zh-Hans, es-MX gets es-419 before es, and
// nb gets no.
//
// An app's messages are in catalogs (Catalog), a JSON file per language,
// of messages of ICU MessageFormat (Message), whose plural forms follow
// the plural rules of CLDR (PluralCategory). `mygo generate` writes typed
// Go and TypeScript functions for them, which say them in the language
// the app shows (Locale). The data of CLDR is generated from a pinned
// release (CLDRVersion) by `go generate`.
//
// FormatNumber and FormatDate write numbers, dates and times in the way of
// a locale, by CLDR's data of the locales of its modern coverage, as ICU
// and Intl write them; Months, Weekdays and DayPeriods give their names,
// FirstWeekday and HourCycle a region's week and clock, and Autonym the
// name of a language in itself. The data is linked into an app only when
// it formats something.
package i18n

import (
	"strings"

	"golang.org/x/text/language"
)

// Normalize returns locale as a BCP 47 tag in its canonical form, or ""
// when it names no language, as "C", "POSIX" or "". It reads BCP 47 tags in
// any case ("en-us" is "en-US"), POSIX locales ("zh_CN.UTF-8" is "zh-CN",
// "sr_RS@latin" is "sr-Latn-RS") and the old codes of a language ("iw" is
// "he", "in" is "id").
func Normalize(locale string) string {
	t, ok := parse(locale)
	if !ok {
		return ""
	}
	return t.String()
}

// parse reads locale as Normalize does.
func parse(locale string) (language.Tag, bool) {
	locale = strings.TrimSpace(locale)
	// A POSIX locale: language[_territory][.codeset][@modifier].
	var modifier string
	if i := strings.IndexByte(locale, '@'); i >= 0 {
		locale, modifier = locale[:i], locale[i+1:]
	}
	if i := strings.IndexByte(locale, '.'); i >= 0 {
		locale = locale[:i]
	}
	switch locale {
	case "", "C", "POSIX":
		return language.Und, false
	}
	t, err := language.Parse(strings.ReplaceAll(locale, "_", "-"))
	if err != nil || t == language.Und {
		return language.Und, false
	}
	if script, ok := modifierScripts[strings.ToLower(modifier)]; ok {
		if s, err := language.ParseScript(script); err == nil {
			if b, conf := t.Base(); conf != language.No {
				r, _ := t.Region()
				if with, err := language.Compose(b, s, r); err == nil {
					t = with
				}
			}
		}
	}
	return t, true
}

// modifierScripts are the scripts the modifiers of POSIX locales name.
var modifierScripts = map[string]string{
	"latin":      "Latn",
	"cyrillic":   "Cyrl",
	"arabic":     "Arab",
	"devanagari": "Deva",
	"hebrew":     "Hebr",
}

// Match returns the language of supported that best suits a reader of
// preferred, the languages they read, most preferred first, or "" when
// none suits them. It returns the tag as supported spells it.
//
// A supported language suits a reader when CLDR finds it close to one they
// read: another region of the same language and script, as en-GB for
// en-AU, pt-BR for pt-PT or es-ES for es-MX, or the language without a
// region. A language written in another script does not suit them: zh-Hans
// is no match for zh-TW, nor sr-Latn for sr-Cyrl. A match of a language
// they prefer more wins over a closer match of one they prefer less: fr
// for fr-CA wins over de for de.
func Match(preferred, supported []string) string {
	var tags []language.Tag
	var kept []string
	for _, s := range supported {
		if t, ok := parse(s); ok {
			tags = append(tags, t)
			kept = append(kept, s)
		}
	}
	if len(tags) == 0 {
		return ""
	}
	var want []language.Tag
	for _, p := range preferred {
		if t, ok := parse(p); ok {
			want = append(want, t)
		}
	}
	if len(want) == 0 {
		return ""
	}
	_, i, conf := language.NewMatcher(tags).Match(want...)
	if conf < language.High {
		return ""
	}
	return kept[i]
}

// Dir is the direction a language is written in.
type Dir uint8

const (
	LeftToRight Dir = iota
	RightToLeft
)

func (d Dir) String() string {
	if d == RightToLeft {
		return "rtl"
	}
	return "ltr"
}

// Direction returns the direction tag is written in, that of its script:
// right to left for Arabic, Hebrew and the other scripts written so, where
// a tag names no script, the one its language is most likely written in
// (ar is Arab, ar-XB, the pseudo locale, too). A tag that names no language
// is left to right.
func Direction(tag string) Dir {
	t, ok := parse(tag)
	if !ok {
		return LeftToRight
	}
	s, _ := t.Script()
	if rtlScripts[s.String()] {
		return RightToLeft
	}
	return LeftToRight
}

// rtlScripts are the scripts of Unicode written right to left.
var rtlScripts = map[string]bool{
	"Adlm": true, "Arab": true, "Aran": true, "Armi": true, "Avst": true,
	"Chrs": true, "Cprt": true, "Elym": true, "Hatr": true, "Hebr": true,
	"Hung": true, "Khar": true, "Lydi": true, "Mand": true, "Mani": true,
	"Mend": true, "Merc": true, "Mero": true, "Narb": true, "Nbat": true,
	"Nkoo": true, "Orkh": true, "Ougr": true, "Palm": true, "Phli": true,
	"Phlp": true, "Phnx": true, "Prti": true, "Rohg": true, "Samr": true,
	"Sarb": true, "Sogd": true, "Sogo": true, "Syrc": true, "Thaa": true,
	"Yezi": true,
}

// Base returns the language of tag, without its script and region: "zh"
// for "zh-Hant-TW", "he" for "iw". It returns "" when tag names no language.
func Base(tag string) string {
	t, ok := parse(tag)
	if !ok {
		return ""
	}
	b, _ := t.Base()
	return b.String()
}
