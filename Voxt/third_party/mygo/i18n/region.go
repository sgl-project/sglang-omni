package i18n

import (
	"strings"
	"time"

	"golang.org/x/text/language"
)

// Region is how a reader writes dates, times and numbers: the regional
// formats of their system, with the changes they made to them, which may
// be those of another language than the one they read, as an English
// reader in Germany who wants weeks to start on Monday and the time in 24
// hours. MyGo's App.Region reports it, and native UI reads it with
// Context.Region.
type Region struct {
	// Locale is the locale of the formats, a BCP 47 tag such as "en-GB"
	// or "de-DE", or "" when the system names none.
	Locale string
	// FirstWeekday is the day calendars start their weeks on.
	FirstWeekday time.Weekday
	// HourCycle is 12 for times with AM and PM, 24 for times without, or
	// 0 when it is not known, which the zero Region is: its first weekday,
	// Sunday, means nothing then.
	HourCycle int
}

// FormatLocale returns the locale a reader of lang with these regional
// formats writes numbers and dates in, as Apple's Locale.current puts the
// two together: the language with the region of the formats, and their
// first weekday and hour cycle as the keys of its Unicode extension, as
// "en-GB-u-fw-mon-hc-h23" for English read with British formats, or
// "en-DE-u-fw-mon-hc-h23" with German ones. Without a locale of formats
// it is lang's, with its first weekday and hour cycle when set. The
// pseudo locales (IsPseudo) write them in English, as the formatting of
// package i18n reads them.
func (r Region) FormatLocale(lang string) string {
	t, ok := parse(lang)
	if !ok || IsPseudo(lang) {
		t, _ = parse("en")
	}
	if rt, ok := parse(r.Locale); ok {
		if reg, conf := rt.Region(); conf == language.Exact {
			b, _ := t.Base()
			s, sconf := t.Script()
			parts := []any{b}
			// The script the reader reads, named or implied (zh-HK reads
			// Hant), unless the new region implies it too: zh-HK with
			// German formats is zh-Hant-DE, not zh-DE, which is Hans.
			if sconf != language.No {
				implied, _ := language.Make(b.String() + "-" + reg.String()).Script()
				if sconf == language.Exact || implied != s {
					parts = append(parts, s)
				}
			}
			parts = append(parts, reg)
			if c, err := language.Compose(parts...); err == nil {
				t = c
			}
		}
	}
	tag := stripExtensions(t.String())
	var ext []string
	if r.HourCycle == 12 || r.HourCycle == 24 {
		ext = append(ext, "fw", [...]string{"sun", "mon", "tue", "wed", "thu", "fri", "sat"}[r.FirstWeekday%7])
		if r.HourCycle == 12 {
			ext = append(ext, "hc", "h12")
		} else {
			ext = append(ext, "hc", "h23")
		}
	}
	if len(ext) == 0 {
		return tag
	}
	return tag + "-u-" + strings.Join(ext, "-")
}
