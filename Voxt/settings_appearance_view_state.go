package main

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

const (
	settingsSidebarAutomatic = "automatic"
	settingsSidebarNames     = "names"
	settingsSidebarIcons     = "icons"
)

type settingsAppearanceViewState struct {
	Theme          int       `json:"theme"`
	ThemeOptions   []string  `json:"theme_options"`
	ThemeValues    []string  `json:"theme_values"`
	Scale          int       `json:"scale"`
	ScaleOptions   []string  `json:"scale_options"`
	ScaleValues    []float64 `json:"scale_values"`
	Sidebar        int       `json:"sidebar"`
	SidebarOptions []string  `json:"sidebar_options"`
	SidebarValues  []string  `json:"sidebar_values"`
	ReduceMotion   bool      `json:"reduce_motion"`
}

var scaleLabels = []string{"90%", "100%", "110%", "125%"}

func buildSettingsAppearanceViewState(translator *i18n.Translator, appearance settings.Appearance) settingsAppearanceViewState {
	return settingsAppearanceViewState{
		Theme:        max(0, slices.Index(settings.Themes, appearance.Theme)),
		ThemeOptions: []string{translator.T("System"), translator.T("Light"), translator.T("Dark")},
		ThemeValues:  slices.Clone(settings.Themes),
		Scale:        max(0, slices.Index(settings.Scales, appearance.Scale)), ScaleOptions: slices.Clone(scaleLabels),
		ScaleValues: slices.Clone(settings.Scales), Sidebar: sidebarChoice(appearance),
		SidebarOptions: []string{translator.T("Automatic"), translator.T("Names and icons"), translator.T("Icons only")},
		SidebarValues:  []string{settingsSidebarAutomatic, settingsSidebarNames, settingsSidebarIcons},
		ReduceMotion:   appearance.ReduceMotion,
	}
}

// sidebarChoice chooses the sidebar's layout: automatic, names and icons,
// icons only.
func sidebarChoice(appearance settings.Appearance) int {
	switch {
	case appearance.Rail:
		return 2
	case appearance.RailChosen:
		return 1
	}
	return 0
}
