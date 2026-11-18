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
	Palette        string    `json:"palette,omitempty"`
	CustomPalette  bool      `json:"custom_palette,omitempty"`
	Scale          int       `json:"scale"`
	ScaleOptions   []string  `json:"scale_options"`
	ScaleValues    []float64 `json:"scale_values"`
	Sidebar        int       `json:"sidebar"`
	SidebarOptions []string  `json:"sidebar_options"`
	SidebarValues  []string  `json:"sidebar_values"`
	ReduceMotion   bool      `json:"reduce_motion"`

	Transparent        bool    `json:"transparent,omitempty"`
	WindowTransparency float64 `json:"window_transparency,omitempty"`
	MinTransparency    float64 `json:"min_window_transparency,omitempty"`
	MaxTransparency    float64 `json:"max_window_transparency,omitempty"`
}

var scaleLabels = []string{"90%", "100%", "110%", "125%"}

func buildSettingsAppearanceViewState(translator *i18n.Translator, appearance settings.Appearance) settingsAppearanceViewState {
	palette := paletteSnapshot(appearance.Palette)
	state := settingsAppearanceViewState{
		Theme:        max(0, slices.Index(settings.Themes, appearance.Theme)),
		ThemeOptions: []string{translator.T("System"), translator.T("Dark"), translator.T("Light"), translator.T("Transparent")},
		ThemeValues:  slices.Clone(settings.Themes),
		Palette:      palette, CustomPalette: palette != "",
		Scale: max(0, slices.Index(settings.Scales, appearance.Scale)), ScaleOptions: slices.Clone(scaleLabels),
		ScaleValues: slices.Clone(settings.Scales), Sidebar: sidebarChoice(appearance),
		SidebarOptions: []string{translator.T("Automatic"), translator.T("Names and icons"), translator.T("Icons only")},
		SidebarValues:  []string{settingsSidebarAutomatic, settingsSidebarNames, settingsSidebarIcons},
		ReduceMotion:   appearance.ReduceMotion,
	}
	if appearance.Theme == settings.ThemeTransparent {
		state.Transparent = true
		state.WindowTransparency = float64(appearance.WindowTransparency)
		state.MinTransparency = settings.MinWindowTransparency
		state.MaxTransparency = settings.MaxWindowTransparency
	}
	return state
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
