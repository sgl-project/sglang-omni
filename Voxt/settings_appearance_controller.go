package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) appearanceSettingsModel() settingsAppearanceViewState {
	return buildSettingsAppearanceViewState(i18n.CurrentTranslator(), a.prefs.Appearance)
}

func (a *app) setTheme(theme string) {
	a.updateSettings(func(current *settings.Settings) { current.Appearance.Theme = theme })
}

func (a *app) setInterfaceScale(scale float64) {
	a.updateSettings(func(current *settings.Settings) { current.Appearance.Scale = scale })
}

func (a *app) setSidebarChoice(choice string) {
	a.updateSettings(func(current *settings.Settings) {
		switch choice {
		case settingsSidebarAutomatic:
			current.Appearance.Rail, current.Appearance.RailChosen = false, false
		case settingsSidebarIcons:
			current.Appearance.Rail, current.Appearance.RailChosen = true, true
		default:
			current.Appearance.Rail, current.Appearance.RailChosen = false, true
		}
	})
}

func (a *app) setReduceMotion(on bool) {
	a.updateSettings(func(current *settings.Settings) { current.Appearance.ReduceMotion = on })
}
