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
	// Palette is the saved colour ("#0d9384"), DefaultPalette the one
	// Reset color restores.
	Palette        string `json:"palette"`
	DefaultPalette string `json:"default_palette"`
	// Transparent shows the Window transparency slider, from
	// MinTransparency to MaxTransparency percent.
	Transparent     bool    `json:"transparent"`
	MinTransparency float64 `json:"min_transparency"`
	MaxTransparency float64 `json:"max_transparency"`
}

var scaleLabels = []string{"90%", "100%", "110%", "125%"}

func buildSettingsAppearanceViewState(translator *i18n.Translator, appearance settings.Appearance) settingsAppearanceViewState {
	return settingsAppearanceViewState{
		Theme:        max(0, slices.Index(settings.Themes, appearance.Theme)),
		ThemeOptions: []string{translator.T("System"), translator.T("Dark"), translator.T("Light"), translator.T("Transparent")},
		ThemeValues:  slices.Clone(settings.Themes),
		Scale:        max(0, slices.Index(settings.Scales, appearance.Scale)), ScaleOptions: slices.Clone(scaleLabels),
		ScaleValues: slices.Clone(settings.Scales), Sidebar: sidebarChoice(appearance),
		SidebarOptions: []string{translator.T("Automatic"), translator.T("Names and icons"), translator.T("Icons only")},
		SidebarValues:  []string{settingsSidebarAutomatic, settingsSidebarNames, settingsSidebarIcons},
		ReduceMotion:   appearance.ReduceMotion,
		Palette:        appearance.Palette, DefaultPalette: settings.DefaultPalette,
		Transparent:     appearance.Theme == settings.ThemeTransparent,
		MinTransparency: settings.MinWindowTransparency, MaxTransparency: settings.MaxWindowTransparency,
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
