package main

import (
	"context"
	"sync/atomic"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type settingsAppearanceState struct {
	paletteRevision         atomic.Uint64
	paletteSavePending      bool
	transparencySavePending bool
	transparencyRevision    atomic.Uint64
}

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

func (a *app) loadAppearancePreview() {
	local := &a.settingsForm.settingsAppearanceLocalUI
	palette := a.prefs.Appearance.Palette
	local.colorPalette = appearanceColor(palette)
	local.colorPaletteSource, local.colorPaletteLoaded = paletteSnapshot(palette), true
	if !local.transparencyDragging {
		local.windowTransparency = float64(a.prefs.Appearance.WindowTransparency)
	}
}

// invalidateAppearancePreviewSaves makes an appearance reset win over any
// preview save that is still in flight.
func (a *app) invalidateAppearancePreviewSaves() {
	a.paletteRevision.Add(1)
	a.transparencyRevision.Add(1)
}

func (a *app) setPaletteColor(color string) {
	if color == "" {
		color = settings.DefaultPalette
	}
	if a.prefs.Appearance.Palette == color && !a.paletteSavePending {
		return
	}
	a.settingsRevision++
	a.featureRevision++
	a.settingsForm.colorPalette = appearanceColor(color)
	a.settingsForm.colorPaletteSource, a.settingsForm.colorPaletteLoaded = paletteSnapshot(color), true
	a.prefs.Appearance.Palette = color
	a.paletteSavePending = true
	a.invalidatePaletteOverlays()
	store := a.settings
	revision := a.paletteRevision.Add(1)
	a.async(func() {
		err := store.Update(context.Background(), func(current *settings.Settings) {
			if a.paletteRevision.Load() == revision {
				current.Appearance.Palette = color
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
				a.featureRevision++
				a.loadAppearancePreview()
				a.invalidatePaletteOverlays()
				a.toast(i18n.T("Could not save the settings: %v", err))
			}
		})
	})
}

func (a *app) resetPaletteColor() {
	a.setPaletteColor(settings.DefaultPalette)
}

// saveWindowTransparency commits the preview after the drag ends, or
// immediately for a keyboard or accessibility edit.
func (a *app) saveWindowTransparency() {
	percent := min(max(int(a.settingsForm.windowTransparency), settings.MinWindowTransparency), settings.MaxWindowTransparency)
	if a.prefs.Appearance.WindowTransparency == percent && !a.transparencySavePending {
		return
	}
	a.settingsRevision++
	a.featureRevision++
	a.prefs.Appearance.WindowTransparency = percent
	a.transparencySavePending = true
	store := a.settings
	revision := a.transparencyRevision.Add(1)
	a.async(func() {
		err := store.Update(context.Background(), func(current *settings.Settings) {
			if a.transparencyRevision.Load() == revision {
				current.Appearance.WindowTransparency = percent
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
				a.featureRevision++
				a.loadAppearancePreview()
				a.toast(i18n.T("Could not save the settings: %v", err))
			}
		})
	})
}
