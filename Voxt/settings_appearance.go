package main

// Settings → Appearance: the sidebar's layout (from AudioSmith), with an
// automatic one: icons only while the window is narrow (shell_nav.go).

import (
	"context"
	"fmt"
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// sidebarRow chooses the sidebar's layout: automatic, names and icons,
// icons only.
func sidebarChoice(ap settings.Appearance) int {
	switch {
	case ap.Rail:
		return 2
	case ap.RailChosen:
		return 1
	}
	return 0
}

func (a *app) sidebarRow(c *ui.Context) {
	f := &a.settingsForm
	settingRow(c, "sidebar-style", i18n.T("Sidebar"), i18n.T("Automatic shows icons only while the window is narrow."), func() {
		if ui.Segmented(c.Key("sidebar-seg"), &f.sidebar, i18n.T("Automatic"), i18n.T("Names and icons"), i18n.T("Icons only")).Label(i18n.T("Sidebar")).Changed() {
			if f.sidebar == 0 {
				a.updateSettings(func(s *settings.Settings) { s.Appearance.Rail, s.Appearance.RailChosen = false, false })
				return
			}
			a.setRail(f.sidebar == 2)
		}
	})
}

func (a *app) appearanceSettings(c *ui.Context) {
	settingRow(c, "theme", i18n.T("Theme"), "", func() {
		mode := a.prefs.Appearance.Theme
		if mode == settings.ThemeSystem {
			mode = settings.ThemeLight
			if c.Theme().Dark {
				mode = settings.ThemeDark
			}
		}
		selected := slices.Index(settings.Themes, mode)
		if ui.Segmented(c.Key("theme-seg"), &selected, i18n.T("Light"), i18n.T("Dark"), i18n.T("Transparent")).Label(i18n.T("Theme")).Changed() {
			a.setAppearanceMode(settings.Themes[selected])
		}
	})
	a.paletteSettings(c)
	if a.prefs.Appearance.Theme == settings.ThemeTransparent {
		a.windowTransparencySettings(c)
	}
	settingRow(c, "scale", i18n.T("Interface size"), i18n.T("Scales text and spacing."), func() {
		if ui.Segmented(c.Key("scale-seg"), &a.scale, scaleLabels...).Label(i18n.T("Interface size")).Changed() {
			scale := settings.Scales[a.scale]
			a.updateSettings(func(s *settings.Settings) { s.Appearance.Scale = scale })
		}
	})
	a.sidebarRow(c) // settings_appearance.go
	settingRow(c, "motion", i18n.T("Reduce motion"), i18n.T("Also follows the system setting."), func() {
		if ui.Switch(c.Key("motion-switch"), &a.reduceMotion).Label(i18n.T("Reduce motion")).Changed() {
			on := a.reduceMotion
			a.updateSettings(func(s *settings.Settings) { s.Appearance.ReduceMotion = on })
		}
	})
}

func (a *app) setAppearanceMode(mode string) {
	a.updateSettings(func(s *settings.Settings) { s.Appearance.Theme = mode })
}

func (a *app) paletteSettings(c *ui.Context) {
	title := i18n.T("Color")
	if !settingsSearchShows(title, "Color palette") {
		return
	}
	t := c.Theme()
	k := tok(t)
	ui.Column(c.Key("palette-setting")).Shrink(0).Children(func() {
		parts := ui.CollapsibleBase(c, &a.colorEditorOpen)
		trigger := parts.Trigger.MinHeight(px(t, 32)).Gap(px(t, 4)).AlignItems(ui.Center).Radius(px(t, radiusControl)).Label(title)
		trigger.Children(func() {
			ui.Text(c, title).SingleLine().Shrink(0)
			arrow := ui.DisclosureArrow(c, 90*parts.Progress())
			if !trigger.Hovered() && !trigger.FocusVisible() {
				arrow.Opacity(0)
			}
			ui.Spacer(c)
			color := a.colorPalette
			ui.Box(c).Size(px(t, 23), px(t, 23)).Radius(px(t, 6)).Background(k.Subtle).Border(1, k.Line).Padding(px(t, 2)).Children(func() {
				ui.Box(c).Fill().Radius(px(t, 4)).Border(1, ui.RGBA(255, 255, 255, 0.16)).
					Gradient(color.Mix(ui.RGB(255, 255, 255), 0.09), color.Mix(ui.RGB(0, 0, 0), 0.09), 180)
			})
		})
		trigger.Changed()
		panel := parts.Panel(func() {
			ui.Row(c).Wrap().Gap(px(t, sp4)).AlignItems(ui.Start).Children(func() {
				picker := ui.OpaqueColorPicker(c.Key("theme-color"), &a.colorPalette, func() {
					if quietButton(c.Key("palette-reset"), icons.Refresh, i18n.T("Reset color")).Disabled(a.prefs.Appearance.Palette == settings.DefaultPalette).Clicked() {
						a.setPaletteColor(settings.DefaultPalette)
					}
				}).Width(px(t, 246)).Label(title)
				if picker.Changed() {
					color := a.colorPalette
					a.setPaletteColor(fmt.Sprintf("#%02x%02x%02x", color.R, color.G, color.B))
				}
				ui.Column(c).Basis(px(t, 158)).Grow(1).Gap(px(t, sp2)).Label(i18n.T("Preview")).Children(func() {
					palettePreview(c, a.prefs.Appearance.Palette, false, i18n.T("Light"))
					palettePreview(c, a.prefs.Appearance.Palette, true, i18n.T("Dark"))
				})
			})
		})
		if panel.Valid() {
			panel.Padding(px(t, sp2), 0, 0)
		}
	})
}

func palettePreview(c *ui.Context, color string, dark bool, label string) {
	t := c.Theme()
	k := paletteTokens(color, dark)
	ui.Column(c).Height(px(t, 92)).Radius(px(t, radiusCard)).Border(1, k.Line).Background(k.Bg).Clip().Children(func() {
		ui.Row(c).Height(px(t, 25)).Shrink(0).Padding(0, px(t, sp3)).AlignItems(ui.Center).Gap(px(t, sp2)).Children(func() {
			ui.Box(c).Size(px(t, 7), px(t, 7)).Radius(px(t, 3.5)).Background(k.Accent).Shrink(0)
			ui.Text(c, label).FontSize(px(t, fsMicro)).FontWeight(600).TextColor(k.Text2)
		})
		ui.Box(c).Height(1).Shrink(0).Background(k.Line)
		ui.Row(c).Grow(1).MinHeight(0).AlignItems(ui.Stretch).Children(func() {
			ui.Column(c).Width(px(t, 44)).Shrink(0).Padding(px(t, 6)).Gap(px(t, 5)).Background(k.Sidebar).Children(func() {
				ui.Box(c).Height(px(t, 15)).Radius(px(t, 4)).Background(k.NavSelected).Center().Children(func() {
					ui.Icon(c, icons.VoxtTranscribe.Bold).Size(px(t, 9), px(t, 9)).TextColor(k.AccentInk)
				})
				ui.Box(c).Height(px(t, 3)).Width(px(t, 18)).Radius(px(t, 1.5)).Background(k.G[5]).Margin(0, 0, 0, px(t, 4))
			})
			ui.Box(c).Width(1).Shrink(0).Background(k.Line)
			ui.Column(c).Grow(1).MinWidth(0).Padding(px(t, sp2)).Justify(ui.Center).Children(func() {
				ui.Row(c).Height(px(t, 30)).Gap(px(t, sp2)).Padding(0, px(t, sp2)).AlignItems(ui.Center).
					Radius(px(t, radiusControl)).Background(k.TranscriptFace).Children(func() {
					ui.Icon(c, icons.VoxtTranscribe.Bold).Size(px(t, 12), px(t, 12)).TextColor(k.AccentInk).Shrink(0)
					ui.Text(c, i18n.T("Transcribe")).FontSize(px(t, fsMicro)).FontWeight(600).TextColor(k.Text).SingleLine().Ellipsis("…").MinWidth(0)
				})
			})
		})
	})
}

func (a *app) setPaletteColor(color string) {
	if a.prefs.Appearance.Palette == color {
		return
	}
	a.colorPalette = ui.Hex(color)
	a.prefs.Appearance.Palette = color
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
				a.prefs.Appearance.Palette, a.colorPalette = saved, ui.Hex(saved)
				a.invalidatePaletteOverlays()
				a.toast(i18n.T("Could not save the settings: %v", err))
			}
		})
	})
}

func (a *app) windowTransparencySettings(c *ui.Context) {
	title := i18n.T("Window transparency")
	t := c.Theme()
	settingRow(c, "window-transparency", title, "", func() {
		slider := ui.Slider(c.Key("transparency-slider"), &a.windowTransparency, settings.MinWindowTransparency, settings.MaxWindowTransparency).
			Step(1).Label(title).Width(px(t, 200)).MinWidth(px(t, 100)).Shrink(1)
		_, _, a.transparencyDragging = slider.Dragged()
	})
}

// saveWindowTransparency commits the preview after the drag ends, or
// immediately for a keyboard or accessibility edit.
func (a *app) saveWindowTransparency() {
	percent := int(a.windowTransparency)
	a.prefs.Appearance.WindowTransparency = percent
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
				if !a.transparencyDragging {
					a.windowTransparency = float64(saved)
				}
				a.toast(i18n.T("Could not save the settings: %v", err))
			}
		})
	})
}
