//go:build darwin

package darwin

import (
	"strings"
	"sync"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/platform"
)

// The languages and regional formats of the user, and the language the app
// chose for itself.
//
// The languages the user reads are the global AppleLanguages, which System
// Settings orders: NSLocale's preferredLanguages are the app's own when its
// defaults have an AppleLanguages, as System Settings' language of an app
// writes, and as SaveLocale does. That one is what AppKit and WebKit show
// their own texts in from the next launch.

var (
	prefsOnce                sync.Once
	cfPreferencesCopyValue   func(key, app, user, host uintptr) uintptr
	cfPreferencesSetValue    func(key, value, app, user, host uintptr)
	cfPreferencesSynchronize func(app, user, host uintptr) bool
	// The domains of CFPreferences.
	cfAnyApplication, cfCurrentApplication, cfCurrentUser, cfAnyHost uintptr
)

func loadPreferences() {
	prefsOnce.Do(func() {
		purego.RegisterLibFunc(&cfPreferencesCopyValue, libCF, "CFPreferencesCopyValue")
		purego.RegisterLibFunc(&cfPreferencesSetValue, libCF, "CFPreferencesSetValue")
		purego.RegisterLibFunc(&cfPreferencesSynchronize, libCF, "CFPreferencesSynchronize")
		cfAnyApplication = uintptr(constString(libCF, "kCFPreferencesAnyApplication"))
		cfCurrentApplication = uintptr(constString(libCF, "kCFPreferencesCurrentApplication"))
		cfCurrentUser = uintptr(constString(libCF, "kCFPreferencesCurrentUser"))
		cfAnyHost = uintptr(constString(libCF, "kCFPreferencesAnyHost"))
	})
}

// appleLanguages reads AppleLanguages of the current user, of the app's
// own domain alone when own, else of the global domain alone.
func appleLanguages(own bool) []string {
	loadPreferences()
	app := cfAnyApplication
	if own {
		app = cfCurrentApplication
	}
	var out []string
	withPool(func() {
		cfPreferencesSynchronize(app, cfCurrentUser, cfAnyHost)
		v := id(cfPreferencesCopyValue(uintptr(nsString("AppleLanguages")), app, cfCurrentUser, cfAnyHost))
		if v == 0 {
			return
		}
		defer cfRelease(uintptr(v))
		if !sendBool(v, "isKindOfClass:", uintptr(class("NSArray"))) {
			return
		}
		for _, s := range arrayItems(v) {
			if sendBool(s, "isKindOfClass:", uintptr(class("NSString"))) {
				out = append(out, goString(s))
			}
		}
	})
	return out
}

func (a appController) PreferredLanguages() []string {
	if langs := appleLanguages(false); len(langs) > 0 {
		return langs
	}
	// A user who never ordered languages: the system's.
	var out []string
	withPool(func() {
		for _, s := range arrayItems(send(class("NSLocale"), "preferredLanguages")) {
			out = append(out, goString(s))
		}
	})
	return out
}

func (a appController) SavedLocale() (string, bool) {
	if langs := appleLanguages(true); len(langs) > 0 {
		return langs[0], true
	}
	return "", true
}

func (a appController) SaveLocale(tag string) bool {
	loadPreferences()
	withPool(func() {
		var value uintptr
		if tag != "" {
			value = uintptr(nsArray(nsString(tag)))
		}
		cfPreferencesSetValue(uintptr(nsString("AppleLanguages")), value, cfCurrentApplication, cfCurrentUser, cfAnyHost)
		cfPreferencesSynchronize(cfCurrentApplication, cfCurrentUser, cfAnyHost)
	})
	return true
}

// SetLanguages does nothing: AppKit and WKWebView take the app's language,
// and its direction, from its AppleLanguages, at launch.
func (a appController) SetLanguages([]string, bool) {}

// Region reads the current locale, which macOS makes of the region and the
// formats the user chose, its calendar's first day of the week, and its
// hour cycle: the hour of the skeleton "j" in its date formats, which the
// 24-hour time setting changes.
func (a appController) Region() platform.Region {
	r := platform.Region{FirstWeekday: -1}
	withPool(func() {
		locale := send(class("NSLocale"), "autoupdatingCurrentLocale")
		r.Locale = goString(send(locale, "localeIdentifier"))
		calendar := send(class("NSCalendar"), "autoupdatingCurrentCalendar")
		if d := sendInt(calendar, "firstWeekday"); d >= 1 && d <= 7 {
			r.FirstWeekday = d - 1
		}
		format := goString(send(class("NSDateFormatter"), "dateFormatFromTemplate:options:locale:",
			uintptr(nsString("j")), 0, uintptr(locale)))
		r.HourCycle = hourCycle(format)
	})
	return r
}

// hourCycle returns the hour cycle of a date format pattern: 12 for h and
// K, 24 for H and k, 0 for none. Quoted text is not a field.
func hourCycle(pattern string) int {
	quoted := false
	for _, c := range pattern {
		switch {
		case c == '\'':
			quoted = !quoted
		case quoted:
		case strings.ContainsRune("hK", c):
			return 12
		case strings.ContainsRune("Hk", c):
			return 24
		}
	}
	return 0
}
