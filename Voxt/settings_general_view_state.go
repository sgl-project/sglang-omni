package main

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type settingsLanguageViewState struct {
	Code  string `json:"code"`
	Label string `json:"label"`
}

type settingsFeatureViewState struct {
	ID    string `json:"id"`
	Title string `json:"title"`
	Shown bool   `json:"shown"`
}

type settingsGeneralViewState struct {
	Language            string                      `json:"language"`
	LanguageOptions     []settingsLanguageViewState `json:"language_options"`
	MainLanguage        string                      `json:"main_language"`
	MainLanguageOptions []settingsLanguageViewState `json:"main_language_options"`
	OtherLanguages      []settingsLanguageViewState `json:"other_languages"`
	AvailableLanguages  []settingsLanguageViewState `json:"available_languages"`
	CopyResult          bool                        `json:"copy_result"`
	MenuBar             bool                        `json:"menu_bar"`
	Dock                bool                        `json:"dock"`
	CloseToTray         bool                        `json:"close_to_tray"`
	OpenAtLogin         bool                        `json:"open_at_login"`
	Features            []settingsFeatureViewState  `json:"features"`
}

// mainLanguages are the languages the main language select offers (lang
// codes), most spoken by VoxtNew's users first.
var mainLanguages = []string{"zh", "en", "ja", "ko", "yue", "fr", "de", "es", "pt", "ru", "it", "vi", "th", "id", "ar"}

func buildSettingsGeneralViewState(translator *i18n.Translator, prefs settings.Settings) settingsGeneralViewState {
	state := settingsGeneralViewState{
		LanguageOptions: settingsAppLanguageOptions(translator), MainLanguageOptions: settingsMainLanguageOptions(translator),
		CopyResult: prefs.General.CopyResult, MenuBar: prefs.General.MenuBar, Dock: !prefs.General.HideDock,
		CloseToTray: prefs.General.CloseToTray, OpenAtLogin: prefs.General.LaunchAtLogin,
	}
	state.Language = state.LanguageOptions[0].Label
	for _, option := range state.LanguageOptions[1:] {
		if option.Code == prefs.General.Language {
			state.Language = option.Label
		}
	}
	state.MainLanguage = state.MainLanguageOptions[0].Label
	if index := slices.Index(mainLanguages, prefs.General.MainLanguage); index >= 0 {
		state.MainLanguage = state.MainLanguageOptions[index+1].Label
	}

	mainCode := lang.Canonical(session.MainLanguage(prefs, translator.Locale()))
	otherCodes := withoutLanguage(prefs.General.OtherLanguages, mainCode)
	for _, code := range otherCodes {
		state.OtherLanguages = append(state.OtherLanguages, settingsLanguageViewState{Code: code, Label: settingsLanguageLabel(translator, code)})
	}
	for _, code := range mainLanguages {
		if lang.Canonical(code) == mainCode || slices.ContainsFunc(otherCodes, func(other string) bool {
			return lang.Canonical(other) == lang.Canonical(code)
		}) {
			continue
		}
		state.AvailableLanguages = append(state.AvailableLanguages, settingsLanguageViewState{Code: code, Label: settingsLanguageLabel(translator, code)})
	}
	for _, featureID := range []string{settings.HideTranslate, settings.HideRewrite, settings.HideAppEnhancement} {
		state.Features = append(state.Features, settingsFeatureViewState{
			ID: featureID, Title: settingsFeatureTitle(translator, featureID), Shown: !prefs.Features.IsHidden(featureID),
		})
	}
	return state
}

func settingsAppLanguageOptions(translator *i18n.Translator) []settingsLanguageViewState {
	options := []settingsLanguageViewState{{Label: translator.T("Follow the system")}}
	for _, tag := range i18n.Supported() {
		options = append(options, settingsLanguageViewState{Code: tag, Label: i18n.Name(tag)})
	}
	return options
}

func settingsMainLanguageOptions(translator *i18n.Translator) []settingsLanguageViewState {
	options := []settingsLanguageViewState{{Label: translator.T("Same as the app")}}
	for _, code := range mainLanguages {
		options = append(options, settingsLanguageViewState{Code: code, Label: settingsLanguageLabel(translator, code)})
	}
	return options
}

func settingsLanguageLabel(translator *i18n.Translator, code string) string {
	language, ok := lang.Lookup(code)
	if !ok {
		return code
	}
	name := language.NameIn(translator.Locale())
	if language.Native != "" && language.Native != name {
		return name + " · " + language.Native
	}
	return name
}

func settingsFeatureTitle(translator *i18n.Translator, featureID string) string {
	switch featureID {
	case settings.HideTranslate:
		return translator.T("Translate")
	case settings.HideRewrite:
		return translator.T("Rewrite")
	case settings.HideAppEnhancement:
		return translator.T("App enhancement")
	}
	return featureID
}

// withoutLanguage is list without code (in any spelling: "zh", "zh-Hans").
func withoutLanguage(list []string, code string) []string {
	canonical := lang.Canonical(code)
	return slices.DeleteFunc(slices.Clone(list), func(other string) bool {
		return other == code || canonical != "" && lang.Canonical(other) == canonical
	})
}
