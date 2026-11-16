package main

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) generalSettingsModel() settingsGeneralViewState {
	return buildSettingsGeneralViewState(i18n.CurrentTranslator(), a.prefs)
}

func (a *app) setAppLanguage(languageTag string) {
	a.updateSettings(func(current *settings.Settings) { current.General.Language = languageTag })
}

// setMainLanguage sets the main language (a lang code, "" = the app's)
// and takes it off the other languages.
func (a *app) setMainLanguage(languageCode string) {
	a.updateSettings(func(current *settings.Settings) {
		current.General.MainLanguage = languageCode
		current.General.OtherLanguages = withoutLanguage(current.General.OtherLanguages, session.MainLanguage(*current, i18n.Locale()))
	})
}

func (a *app) addOtherLanguage(languageCode string) {
	a.updateSettings(func(current *settings.Settings) {
		if !slices.Contains(current.General.OtherLanguages, languageCode) {
			current.General.OtherLanguages = append(slices.Clone(current.General.OtherLanguages), languageCode)
		}
	})
}

func (a *app) removeOtherLanguage(languageCode string) {
	a.updateSettings(func(current *settings.Settings) {
		current.General.OtherLanguages = withoutLanguage(current.General.OtherLanguages, languageCode)
	})
}

func (a *app) setCopyResult(on bool) {
	a.updateSettings(func(current *settings.Settings) { current.General.CopyResult = on })
}

func (a *app) setMenuBar(on bool) {
	a.updateSettings(func(current *settings.Settings) {
		current.General.MenuBar = on
		if !on {
			current.General.HideDock = false // VoxtNew must stay reachable
		}
	})
}

func (a *app) setDock(on bool) {
	a.updateSettings(func(current *settings.Settings) {
		current.General.HideDock = !on
		if !on {
			current.General.MenuBar = true // VoxtNew must stay reachable
		}
	})
}

func (a *app) setCloseToTray(on bool) {
	a.updateSettings(func(current *settings.Settings) { current.General.CloseToTray = on })
}

func (a *app) setOpenAtLogin(on bool) {
	a.updateSettings(func(current *settings.Settings) { current.General.LaunchAtLogin = on })
}

func (a *app) setFeatureShown(featureID string, shown bool) {
	a.updateSettings(func(current *settings.Settings) {
		current.Features.Hidden = slices.DeleteFunc(current.Features.Hidden, func(hidden string) bool { return hidden == featureID })
		if !shown {
			current.Features.Hidden = append(current.Features.Hidden, featureID)
		}
	})
}
