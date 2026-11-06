package mygo

import (
	"log"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/egoist/mygo/i18n"
	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/internal/platform"
)

// The locale model: the one language the app shows, which its own texts,
// MyGo's controls and menus, the text engine and the pages all follow, and
// the regional formats, which are a setting of their own.
//
// The app's language is the user's preferred languages, or the language
// the app chose for them (SetLocale), matched against the languages the
// app is written in (SetLanguages), and the first of those, the app's own,
// where none matches: an app written in English alone shows English to a
// reader of Chinese, as AppKit does. Everything that shows a language
// reads it from here, so that no part of the app resolves its own.

// Region is the user's regional formats (App.Region).
type Region = i18n.Region

// localeState is what the app knows of its languages. Guarded by mu but
// shown, which the main thread sets and every goroutine reads.
type localeState struct {
	mu sync.Mutex
	// languages are those SetLanguages set, or nil for the catalogs'.
	languages []string
	// override is the language the app chose (SetLocale, the one saved,
	// or MYGO_LOCALE), "" to follow the user's; fromEnv tells that
	// MYGO_LOCALE set it, which makes SetLocale save nothing.
	override string
	fromEnv  bool
	// loaded tells that the saved language and MYGO_LOCALE were read.
	loaded bool
	// chosen tells that SetLocale chose the language, which reading the
	// saved one again does not undo.
	chosen bool
	// generation counts the SetLocale calls; the work they queue runs for
	// the last one only.
	generation uint64
	// shown is what the app shows, as the windows and listeners last heard
	// of it, nil before anything asked.
	shown atomic.Pointer[shownLocale]

	onLocaleChanged listeners[func(string)]
	onRegionChanged listeners[func(Region)]
}

// shownLocale is the language and the formats the app shows.
type shownLocale struct {
	locale    string
	preferred []string
	region    Region
}

var locales localeState

// PreferredLanguages returns the languages the user reads, most preferred
// first, as normalized BCP 47 tags, e.g. ["zh-Hans-CN", "en-GB"]: on
// macOS the languages of System Settings, on Windows the display
// languages, on Linux LANGUAGE, then the locale of messages (LC_ALL,
// LC_MESSAGES, LANG), in a browser navigator.languages. The language the
// app chose (SetLocale) never changes them.
func (a *Application) PreferredLanguages() []string {
	return slices.Clone(a.shownLocale().preferred)
}

// SetLanguages sets the languages the app is written in, its own
// language first, which the app falls back to: those the app's texts are
// translated to, e.g. "en", "de", "zh-Hans". Locale picks among them. By
// default they are the languages of the catalogs of the app (package
// i18n), and English alone without any. It panics on a tag that names no
// language.
func (a *Application) SetLanguages(tags ...string) {
	for _, t := range tags {
		if i18n.Normalize(t) == "" {
			panic("mygo: SetLanguages: " + strings.TrimSpace(t) + " is not a language tag")
		}
	}
	locales.mu.Lock()
	locales.languages = slices.Clone(tags)
	if len(tags) == 0 {
		locales.languages = nil
	}
	locales.mu.Unlock()
	postMain(func() { a.updateLocale(true) })
}

// Languages returns the languages the app is written in, its own first
// (SetLanguages).
func (a *Application) Languages() []string {
	locales.mu.Lock()
	defer locales.mu.Unlock()
	return appLanguages(locales.languages)
}

// appLanguages returns set, else the languages of the catalogs, else
// English.
func appLanguages(set []string) []string {
	if len(set) > 0 {
		return slices.Clone(set)
	}
	if langs := i18n.Languages(); len(langs) > 0 {
		return langs
	}
	return []string{"en"}
}

// Locale returns the language the app shows, as a tag of the languages
// it is written in (SetLanguages), e.g. "zh-Hans": the one the user reads
// best of them, or the one SetLocale chose, else the app's own language.
// The app's own texts, MyGo's controls and menus, the text engine and the
// pages (mygo.locale) all show it.
//
// Before MyGo 0.5 it was the user's preferred language whatever the app
// is written in; PreferredLanguages reports those now.
func (a *Application) Locale() string {
	return a.shownLocale().locale
}

// SetLocale makes the app show the language of tag, a BCP 47 tag such as
// "de" or "zh-Hant-TW", of those it is written in, or "" to follow the
// user's languages again, which is the default. The app shows it at once:
// windows of native UI draw anew, menus say their items in it, pages hear
// the mygo:locale event and those loaded later start with it, and
// OnLocaleChanged listeners are called. It is kept for the next launches:
// on macOS as the app's language in its defaults, which System Settings
// shows and AppKit and WebKit say their own texts in from the next launch,
// in browsers in localStorage, elsewhere in a file in PathUserData. The
// MYGO_LOCALE environment variable (`mygo dev -lang`) chooses the language
// for one run and keeps nothing.
//
// The pseudo locales en-XA and ar-XB are always shown, to test what
// translations would do. SetLocale panics on a tag that names no language.
// It is safe from any goroutine; calls made before the main thread gets to
// them show the last tag only. While MYGO_LOCALE is set, calls made before
// the app is ready change nothing.
func (a *Application) SetLocale(tag string) {
	normalized := ""
	if strings.TrimSpace(tag) != "" {
		if normalized = i18n.Normalize(tag); normalized == "" {
			panic("mygo: SetLocale: " + strings.TrimSpace(tag) + " is not a language tag")
		}
	}
	locales.mu.Lock()
	locales.generation++
	generation := locales.generation
	locales.mu.Unlock()
	apply := func() {
		loadLocale()
		locales.mu.Lock()
		if locales.generation != generation || locales.fromEnv && !a.IsReady() {
			// A later call shows its own; MYGO_LOCALE wins over the
			// language the app starts with.
			locales.mu.Unlock()
			return
		}
		locales.override, locales.chosen = normalized, true
		save := !locales.fromEnv
		locales.mu.Unlock()
		if save {
			saveLocale(normalized)
		}
		a.updateLocale(true)
	}
	if isMainThread() {
		apply()
	} else {
		postMain(apply)
	}
}

// OnLocaleChanged calls fn on the main thread with the language the app
// shows when it changes: SetLocale chose another, the languages the app
// is written in changed, or the user's languages changed while the app
// follows them. It returns a function that stops the listener.
func (a *Application) OnLocaleChanged(fn func(locale string)) (off func()) {
	return locales.onLocaleChanged.add(fn, false)
}

// Region returns the user's regional formats, as the system's settings
// have them, with the changes the user made: the locale of the formats,
// which may be another language's than the app's (an English app reading
// German dates for a reader in Germany), the first day of the week and
// the hour cycle. macOS, Windows and browsers report changes while the
// app runs (OnRegionChanged); Linux reads the environment (LC_TIME,
// LC_ALL, LANG) once.
func (a *Application) Region() Region {
	return a.shownLocale().region
}

// OnRegionChanged calls fn on the main thread with the user's regional
// formats when they change. It returns a function that stops the
// listener.
func (a *Application) OnRegionChanged(fn func(Region)) (off func()) {
	return locales.onRegionChanged.add(fn, false)
}

// The accessors of catalogs (package i18n), the app's and MyGo's own, say
// their messages in the language the app shows.
func init() { i18n.SetLocaleFunc(currentLocale) }

// currentLocale returns the language the app shows, from any goroutine,
// for the accessors of catalogs: as App.Locale does once the app works it
// out on the main thread, or the language MYGO_LOCALE chooses, else the
// app's own, before, without waiting for the main thread.
func currentLocale() string {
	if s := locales.shown.Load(); s != nil {
		return s.locale
	}
	App.mu.Lock()
	running := App.running
	App.mu.Unlock()
	if running || isMainThread() {
		return App.Locale()
	}
	locales.mu.Lock()
	langs := appLanguages(locales.languages)
	locales.mu.Unlock()
	return resolveLocale(i18n.Normalize(os.Getenv("MYGO_LOCALE")), nil, langs)
}

// shownLocale returns what the app shows, working it out on the main
// thread the first time.
func (a *Application) shownLocale() *shownLocale {
	if s := locales.shown.Load(); s != nil {
		return s
	}
	return onMainValue(func() *shownLocale {
		if s := locales.shown.Load(); s != nil {
			return s
		}
		a.updateLocale(false)
		return locales.shown.Load()
	})
}

// startLocale tells the backend the app's languages, once it runs. Main
// thread only.
func (a *Application) startLocale() {
	locales.mu.Lock()
	if !locales.chosen {
		// Read before Run, the saved language may have been looked for
		// under the name and paths the app had then: read it again where
		// they put it now.
		locales.loaded = false
	}
	locales.mu.Unlock()
	a.updateLocale(false)
	backend().App().SetLanguages(webLanguages(locales.shown.Load()), appRightToLeft())
	// Menus made before the app knew its language took their roles'
	// labels in its own language.
	relabelMenus()
	locales.mu.Lock()
	langs := appLanguages(locales.languages)
	locales.mu.Unlock()
	for _, missing := range missingTranslations(langs, l10n.Languages, l10n.Catalog.Languages()) {
		log.Printf("mygo: the app is written in %s, but MyGo's own strings are in English there: import _ \"github.com/egoist/mygo/locales/%s\", as the code mygo generate writes for an app's catalogs does, or translate them with keys \"mygo.<key>\" in the app's catalog", missing[0], missing[1])
	}
}

// missingTranslations returns the languages of langs that MyGo's own
// strings are translated to, each in a package, but not linked, with the
// translation that suits each, as {"zh-TW", "zh-Hant"}.
func missingTranslations(langs, translations, linked []string) [][2]string {
	var out [][2]string
	for _, lang := range langs {
		if m := i18n.Match([]string{lang}, translations); m != "" && !slices.Contains(linked, m) {
			out = append(out, [2]string{lang, m})
		}
	}
	return out
}

// localeChangedFromSystem tells the app that the user's languages or
// formats may have changed.
func (a *Application) localeChangedFromSystem() {
	postMain(func() { a.updateLocale(true) })
}

// updateLocale works out what the app shows and, when tell is set and it
// changed, tells the windows, the menus, the backend and the listeners.
// Main thread only.
func (a *Application) updateLocale(tell bool) {
	loadLocale()
	ctl := backend().App()
	next := &shownLocale{preferred: normalizeTags(ctl.PreferredLanguages())}
	locales.mu.Lock()
	next.locale = resolveLocale(locales.override, next.preferred, appLanguages(locales.languages))
	locales.mu.Unlock()
	next.region = regionOf(ctl.Region(), next.preferred)
	before := locales.shown.Swap(next)
	if before == nil || before.locale != next.locale {
		platform.SetTexts(platformTexts(next.locale))
	}
	if before == nil {
		return // nothing heard of a language before
	}
	localeChanged := before.locale != next.locale
	if (localeChanged || !slices.Equal(before.preferred, next.preferred)) && a.initialized {
		ctl.SetLanguages(webLanguages(next), appRightToLeft())
	}
	regionChanged := before.region != next.region
	if !tell || !localeChanged && !regionChanged {
		return
	}
	contentLocaleChanged()
	if localeChanged {
		pagesLocaleChanged(next.locale)
		relabelMenus()
		fire1(&locales.onLocaleChanged, next.locale)
	}
	if regionChanged {
		fire1(&locales.onRegionChanged, next.region)
	}
}

// appRightToLeft reports whether the app is laid out right to left: as the
// language it shows is written, unless MYGO_DIRECTION ("rtl" or "ltr")
// lays it out otherwise, to try a layout without translating the app.
// Native UI, the pages (mygo.dir) and the backend's widgets agree.
func appRightToLeft() bool {
	switch os.Getenv("MYGO_DIRECTION") {
	case "rtl":
		return true
	case "ltr":
		return false
	}
	return i18n.Direction(App.Locale()) == i18n.RightToLeft
}

// appDir returns the direction of appRightToLeft as pages write it.
func appDir() string {
	if appRightToLeft() {
		return "rtl"
	}
	return "ltr"
}

// resolveLocale returns the language of supported to show a reader of
// preferred, or of override and then preferred when the app chose one:
// the pseudo locales as they are, the best match, else the app's own.
func resolveLocale(override string, preferred, supported []string) string {
	want := preferred
	if override != "" {
		if i18n.IsPseudo(override) {
			return override
		}
		want = append([]string{override}, preferred...)
	}
	if m := i18n.Match(want, supported); m != "" {
		return m
	}
	return supported[0]
}

// webLanguages returns the languages for what the backend shows: the
// app's, then the user's others.
func webLanguages(s *shownLocale) []string {
	out := []string{s.locale}
	for _, p := range s.preferred {
		if !slices.ContainsFunc(out, func(t string) bool { return strings.EqualFold(t, p) }) {
			out = append(out, p)
		}
	}
	return out
}

// normalizeTags normalizes tags, leaving out those that name no language
// and those named before.
func normalizeTags(tags []string) []string {
	var out []string
	for _, t := range tags {
		if n := i18n.Normalize(t); n != "" && !slices.Contains(out, n) {
			out = append(out, n)
		}
	}
	return out
}

// loadLocale reads the language the app chose in an earlier run, or
// MYGO_LOCALE, which wins and is never saved, once. Main thread only.
func loadLocale() {
	locales.mu.Lock()
	if locales.loaded {
		locales.mu.Unlock()
		return
	}
	locales.loaded = true
	locales.mu.Unlock()
	tag, fromEnv := os.Getenv("MYGO_LOCALE"), true
	if strings.TrimSpace(tag) == "" {
		tag, fromEnv = savedLocale(), false
	}
	tag = i18n.Normalize(tag)
	locales.mu.Lock()
	locales.override, locales.fromEnv = tag, fromEnv && tag != ""
	locales.mu.Unlock()
}

// savedLocale returns the language the app saved, where the system keeps
// it or in the file. Main thread only.
func savedLocale() string {
	if tag, kept := backend().App().SavedLocale(); kept {
		return tag
	}
	return readLocaleFile()
}

// saveLocale keeps the language the app chose, "" for none. Main thread
// only.
func saveLocale(tag string) {
	if !backend().App().SaveLocale(tag) {
		writeLocaleFile(tag)
	}
}

// regionOf returns the formats of the system, with what it does not say
// filled in from the locale of the formats or, without one, the language
// the user reads.
func regionOf(r platform.Region, preferred []string) Region {
	out := Region{Locale: regionLocale(r.Locale), FirstWeekday: time.Monday, HourCycle: r.HourCycle}
	if out.Locale == "" && len(preferred) > 0 {
		out.Locale = preferred[0]
	}
	// What the system does not say comes from CLDR's week and time data
	// for the locale's region (or its language's likely one), as for a
	// libc without them (musl), a locale the system has not generated,
	// or browsers that do not tell.
	if r.FirstWeekday >= 0 && r.FirstWeekday <= 6 {
		out.FirstWeekday = time.Weekday(r.FirstWeekday)
	} else if out.Locale != "" {
		out.FirstWeekday = i18n.FirstWeekday(out.Locale)
	}
	if out.HourCycle != 12 && out.HourCycle != 24 {
		out.HourCycle = 24
		if out.Locale != "" {
			out.HourCycle = i18n.HourCycle(out.Locale)
		}
	}
	return out
}

// regionLocale normalizes the locale of regional formats, whose region
// the "rg" key of macOS's locales changes: en_US@rg=gbzzzz has the formats
// of en-GB.
func regionLocale(locale string) string {
	if i := strings.IndexByte(locale, '@'); i >= 0 {
		// Keywords, as "@calendar=japanese;rg=gbzzzz".
		var rg string
		for _, kw := range strings.Split(locale[i+1:], ";") {
			if k, v, ok := strings.Cut(kw, "="); ok && strings.EqualFold(k, "rg") {
				rg = v
			}
		}
		base := i18n.Normalize(locale[:i])
		if len(rg) >= 2 && base != "" {
			lang := base
			if r := regionCode(base); r != "" {
				lang = strings.TrimSuffix(base, "-"+r)
			}
			if n := i18n.Normalize(lang + "-" + strings.ToUpper(rg[:2])); n != "" {
				return n
			}
		}
		locale = locale[:i]
	}
	return i18n.Normalize(locale)
}

// regionCode returns the region of a normalized tag, as "GB" of "en-GB",
// or "" without one.
func regionCode(tag string) string {
	parts := strings.Split(tag, "-")
	for _, p := range parts[1:] {
		if len(p) == 2 && p[0] >= 'A' && p[0] <= 'Z' || len(p) == 3 && p[0] >= '0' && p[0] <= '9' {
			return p
		}
	}
	return ""
}
