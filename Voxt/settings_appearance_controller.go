package main

import (
	"context"

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

// syncSavedAppearancePreview puts the saved palette and transparency in
// appearance, except those of a preview still being saved.
func (a *app) syncSavedAppearancePreview(appearance *settings.Appearance) {
	saved := a.settings.Get().Appearance
	if a.paletteSavePending {
		saved.Palette = a.prefs.Appearance.Palette
	}
	if a.transparencySavePending {
		saved.WindowTransparency = a.prefs.Appearance.WindowTransparency
	}
	appearance.Palette, appearance.WindowTransparency = saved.Palette, saved.WindowTransparency
}

// loadAppearancePreview sets the slider from a.prefs, leaving one being
// dragged alone (the picker follows the view state's Palette).
func (a *app) loadAppearancePreview() {
	if !a.transparencyDragging {
		a.windowTransparency = float64(a.prefs.Appearance.WindowTransparency)
	}
}

func (a *app) setPaletteColor(color string) {
	if a.prefs.Appearance.Palette == color {
		return
	}
	a.prefs.Appearance.Palette = color
	a.settingsRevision++
	a.paletteSavePending = true
	a.invalidatePaletteOverlays()
	store := a.settings
	revision := a.paletteRevision.Add(1)
	a.async(func() {
		err := store.Update(context.Background(), func(s *settings.Settings) {
			if a.paletteRevision.Load() == revision {
				s.Appearance.Palette = color
			}
		})
		a.post(func() {
			if a.paletteRevision.Load() != revision {
				return
			}
			a.paletteSavePending = false
			if err != nil {
				saved := store.Get().Appearance.Palette
				a.prefs.Appearance.Palette = saved
				a.settingsRevision++
				a.invalidatePaletteOverlays()
				a.toast(i18n.T("Could not save the settings: %v", err))
			}
		})
	})
}

// saveWindowTransparency commits the preview after the drag ends, or
// immediately for a keyboard or accessibility edit.
func (a *app) saveWindowTransparency() {
	percent := int(a.windowTransparency)
	a.prefs.Appearance.WindowTransparency = percent
	a.settingsRevision++
	a.transparencySavePending = true
	store := a.settings
	revision := a.transparencyRevision.Add(1)
	a.async(func() {
		err := store.Update(context.Background(), func(s *settings.Settings) {
			if a.transparencyRevision.Load() == revision {
				s.Appearance.WindowTransparency = percent
			}
		})
		a.post(func() {
			if a.transparencyRevision.Load() != revision {
				return
			}
			a.transparencySavePending = false
			if err != nil {
				saved := store.Get().Appearance.WindowTransparency
				a.prefs.Appearance.WindowTransparency = saved
				a.settingsRevision++
				if !a.transparencyDragging {
					a.windowTransparency = float64(saved)
				}
				a.toast(i18n.T("Could not save the settings: %v", err))
			}
		})
	})
}
