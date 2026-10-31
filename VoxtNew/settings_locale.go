package main

// The language VoxtNew shows (shell_settings.go): applying it, the
// system's, the language MyGo shows, and the lists of languages the settings
// offer.

import (
	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// applyLocale switches the UI to the settings' language, or the system's,
// and rebuilds the native menu (and the menu bar icon's) in it.
func (a *app) applyLocale() {
	tag := a.prefs.General.Language
	if tag == "" {
		tag = a.systemLocale
	}
	i18n.SetLocale(tag)
	if a.hasMenu { // the app runs: MyGo's locale waits for its main loop
		showLanguage(i18n.Locale())
	}
	if i18n.Locale() != a.menuLocale {
		a.buildMenu()
		a.buildTrayMenu() // tray.go
	}
	a.loadSettingsForm() // its labels are translated
}

// systemLanguage is the user's first preferred language, as the system
// lists them, "" when it lists none: the language VoxtNew follows unless
// the settings name one. (Before MyGo 0.5, mygo.App.Locale; it is now the
// language the app shows, of those it is written in.)
func systemLanguage() string {
	if langs := mygo.App.PreferredLanguages(); len(langs) > 0 {
		return langs[0]
	}
	return ""
}

// showLanguage makes tag, one of i18n.Supported, the language MyGo shows
// (mygo.App.SetLocale; the languages are main's SetLanguages): its own
// strings say it, and the text engine lays the windows' text out in it.
// Without it, MyGo 0.5 lays out every text as English, which on macOS
// gives Chinese punctuation at the end of a line its full width (a face
// of the overlay ending in "。" 6 DIPs wider than with 0.4.8). MyGo keeps
// it for the next launch as the app's language (AppleLanguages in its
// defaults), which AppKit's own texts then say. Main thread, once the app
// is ready.
func showLanguage(tag string) {
	if mygo.App.Locale() != tag {
		mygo.App.SetLocale(tag)
	}
}

// The options of the selects, and what they mean.
func languageOptions() []string {
	opts := []string{i18n.T("Follow the system")}
	for _, tag := range i18n.Supported() {
		opts = append(opts, i18n.Name(tag))
	}
	return opts
}

// mainLanguages are the languages the main language select offers (lang
// codes), most spoken by VoxtNew's users first.
var mainLanguages = []string{"zh", "en", "ja", "ko", "yue", "fr", "de", "es", "pt", "ru", "it", "vi", "th", "id", "ar"}

func mainLanguageOptions() []string {
	opts := []string{i18n.T("Same as the app")}
	for _, code := range mainLanguages {
		opts = append(opts, uitext.LangLabel(code))
	}
	return opts
}
