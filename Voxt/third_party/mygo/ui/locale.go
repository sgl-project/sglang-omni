package ui

import (
	"time"

	"github.com/egoist/mygo/i18n"
	"github.com/egoist/mygo/internal/l10n"
)

// Locale returns the language the app shows (mygo.App.Locale), a BCP 47
// tag such as "en" or "zh-Hans", for what the view says in it. Text is in
// it unless an element says otherwise (Element.Lang). A frame follows its
// changes, as App.SetLocale makes them.
func (c *context) Locale() string { return c.rt.lang }

// Region returns the user's regional formats (mygo.App.Region): the
// locale of the formats, the first day of the week and the hour cycle,
// which dates and times follow. A frame follows their changes.
func (c *context) Region() i18n.Region { return c.rt.host.region() }

// texts returns MyGo's own strings (internal/l10n) in the language the
// window shows, which an app changes with keys "mygo.<key>" of its
// catalogs. Saying one again allocates nothing.
func (c *context) texts() l10n.Messages { return l10n.In(c.rt.lang) }

// formats returns the regional formats the controls write dates and times
// in, and the locale that puts them together with the window's language
// (Region.FormatLocale). Where the system says nothing of them, as for
// the zero Region of a Tester, they are those CLDR gives the language.
func (c *context) formats() (i18n.Region, string) {
	rt := c.rt
	sys := c.Region()
	if rt.fmtKnown && rt.fmtLang == rt.lang && rt.fmtSys == sys {
		return rt.fmtRegion, rt.fmtLocale
	}
	lang := rt.lang
	if lang == "" {
		lang = "en"
	}
	r := sys
	if r.HourCycle != 12 && r.HourCycle != 24 {
		r = i18n.RegionFor(lang)
		r.Locale = ""
	}
	rt.fmtKnown, rt.fmtLang, rt.fmtSys = true, rt.lang, sys
	rt.fmtRegion, rt.fmtLocale = r, r.FormatLocale(lang)
	return rt.fmtRegion, rt.fmtLocale
}

// fmtKey is a string the controls format, by locale, kind and value.
type fmtKey struct {
	locale string
	kind   uint8
	v      int64
}

// The kinds of fmtKey.
const (
	fmtMonthYear  uint8 = iota // a calendar's title: the month v of year*12
	fmtDay                     // a calendar's day: v days after 1970-01-01
	fmtDayNumber               // the number of a day of the month
	fmtShortDate               // a date field's date: as fmtDay
	fmtTwoDigits               // a number of two digits at least
	fmtNumber                  // a number
	fmtColorValue              // a color picker's saturation*1000+brightness, in percents
	fmtColorName               // the name of a color picker's swatch v
)

// formatted returns the string of key, formatting it with make once: the
// controls write the same dates and numbers frame after frame.
func (c *context) formatted(key fmtKey, make func() string) string {
	rt := c.rt
	if s, ok := rt.fmtCache[key]; ok {
		return s
	}
	if rt.fmtCache == nil || len(rt.fmtCache) >= 1024 {
		rt.fmtCache = map[fmtKey]string{}
	}
	s := make()
	rt.fmtCache[key] = s
	return s
}

// dayNumber returns the days from 1970-01-01 to the date of t.
func dayNumber(t time.Time) int64 {
	y, m, d := t.Date()
	return time.Date(y, m, d, 0, 0, 0, 0, time.UTC).Unix() / 86400
}

// formatNumber returns n in the locale's digits, with min digits at
// least and without grouping.
func (c *context) formatNumber(locale string, n, min int) string {
	kind := fmtNumber
	if min == 2 {
		kind = fmtTwoDigits
	}
	return c.formatted(fmtKey{locale, kind, int64(n)}, func() string {
		return i18n.FormatNumber(locale, n, i18n.NumberOptions{NoGrouping: true, MinIntegerDigits: min})
	})
}
