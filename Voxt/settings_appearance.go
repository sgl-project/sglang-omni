package main

// Settings → Appearance: the sidebar's layout (from AudioSmith), with an
// automatic one: icons only while the window is narrow (shell_nav.go).

import (
	"fmt"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type settingsAppearanceLocalUI struct {
	theme        int
	scale        int
	sidebar      int
	reduceMotion bool

	colorPalette         ui.Color
	colorPaletteSource   string
	colorPaletteLoaded   bool
	colorEditorOpen      bool
	windowTransparency   float64
	transparencyDragging bool
}

func appearanceSettings(c *ui.Context, state settingsAppearanceViewState, actions settingsAppearanceActions, local *settingsAppearanceLocalUI) {
	local.transparencyDragging = false
	syncAppearanceColor(state, local)
	settingRow(c, "theme", i18n.T("Theme"), "", func() {
		if ui.Segmented(c.Key("theme-seg"), &local.theme, state.ThemeOptions...).Label(i18n.T("Theme")).Changed() {
			actions.setTheme(state.ThemeValues[local.theme])
		}
	})
	paletteSettings(c, state, actions, local)
	if state.Transparent {
		windowTransparencySettings(c, state, local)
		if !local.transparencyDragging && int(local.windowTransparency) != int(state.WindowTransparency) {
			actions.saveWindowTransparency()
		}
	}
	settingRow(c, "scale", i18n.T("Interface size"), i18n.T("Scales text and spacing."), func() {
		if ui.Segmented(c.Key("scale-seg"), &local.scale, state.ScaleOptions...).Label(i18n.T("Interface size")).Changed() {
			actions.setInterfaceScale(state.ScaleValues[local.scale])
		}
	})
	sidebarRow(c, state, actions, local)
	settingRow(c, "motion", i18n.T("Reduce motion"), i18n.T("Also follows the system setting."), func() {
		if ui.Switch(c.Key("motion-switch"), &local.reduceMotion).Label(i18n.T("Reduce motion")).Changed() {
			actions.setReduceMotion(local.reduceMotion)
		}
	})
}

func appearanceColor(palette string) ui.Color {
	if palette == "" {
		palette = originalPalette
	}
	return ui.Hex(palette)
}

func syncAppearanceColor(state settingsAppearanceViewState, local *settingsAppearanceLocalUI) {
	if local.colorPaletteLoaded && local.colorPaletteSource == state.Palette {
		return
	}
	local.colorPalette = appearanceColor(state.Palette)
	local.colorPaletteSource, local.colorPaletteLoaded = state.Palette, true
}

func paletteSettings(c *ui.Context, state settingsAppearanceViewState, actions settingsAppearanceActions, local *settingsAppearanceLocalUI) {
	title := i18n.T("Color")
	if !settingsSearchShows(title, "Color palette") {
		return
	}
	t := c.Theme()
	k := tok(t)
	ui.Column(c.Key("palette-setting")).Shrink(0).Children(func() {
		parts := ui.CollapsibleBase(c, &local.colorEditorOpen)
		trigger := parts.Trigger.MinHeight(px(t, 32)).Gap(px(t, 4)).AlignItems(ui.Center).Radius(px(t, radiusControl)).Label(title)
		trigger.Children(func() {
			ui.Text(c, title).SingleLine().Shrink(0)
			arrow := ui.DisclosureArrow(c, 90*parts.Progress())
			if !trigger.Hovered() && !trigger.FocusVisible() {
				arrow.Opacity(0)
			}
			ui.Spacer(c)
			color := local.colorPalette
			ui.Box(c).Size(px(t, 23), px(t, 23)).Radius(px(t, 6)).Background(k.Subtle).Border(1, k.Line).Padding(px(t, 2)).Children(func() {
				ui.Box(c).Fill().Radius(px(t, 4)).Border(1, ui.RGBA(255, 255, 255, 0.16)).
					Gradient(color.Mix(ui.RGB(255, 255, 255), 0.09), color.Mix(ui.RGB(0, 0, 0), 0.09), 180)
			})
		})
		trigger.Changed()
		panel := parts.Panel(func() {
			ui.Row(c).Wrap().Gap(px(t, sp4)).AlignItems(ui.Start).Children(func() {
				picker := ui.OpaqueColorPicker(c.Key("theme-color"), &local.colorPalette, func() {
					if quietButton(c.Key("palette-reset"), icons.Refresh, i18n.T("Reset color")).Disabled(!state.CustomPalette).Clicked() {
						actions.resetPaletteColor()
					}
				}).Width(px(t, 246)).Label(title)
				if picker.Changed() {
					color := local.colorPalette
					palette := fmt.Sprintf("#%02x%02x%02x", color.R, color.G, color.B)
					local.colorPaletteSource = palette
					actions.setPaletteColor(palette)
				}
				ui.Column(c).Basis(px(t, 158)).Grow(1).Gap(px(t, sp2)).Label(i18n.T("Preview")).Children(func() {
					palettePreview(c, state.Palette, false, i18n.T("Light"))
					palettePreview(c, state.Palette, true, i18n.T("Dark"))
				})
			})
		})
		if panel.Valid() {
			panel.Padding(px(t, sp2), 0, 0)
		}
	})
}

func palettePreview(c *ui.Context, palette string, dark bool, label string) {
	t := c.Theme()
	k := paletteTokens(palette, dark)
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

func windowTransparencySettings(c *ui.Context, state settingsAppearanceViewState, local *settingsAppearanceLocalUI) {
	title := i18n.T("Window transparency")
	t := c.Theme()
	settingRow(c, "window-transparency", title, "", func() {
		slider := ui.Slider(c.Key("transparency-slider"), &local.windowTransparency, state.MinTransparency, state.MaxTransparency).
			Step(1).Label(title).Width(px(t, 200)).MinWidth(px(t, 100)).Shrink(1)
		_, _, local.transparencyDragging = slider.Dragged()
	})
}

// sidebarRow chooses the sidebar's layout: automatic, names and icons,
// icons only.
func sidebarRow(c *ui.Context, state settingsAppearanceViewState, actions settingsAppearanceActions, local *settingsAppearanceLocalUI) {
	settingRow(c, "sidebar-style", i18n.T("Sidebar"), i18n.T("Automatic shows icons only while the window is narrow."), func() {
		if ui.Segmented(c.Key("sidebar-seg"), &local.sidebar, state.SidebarOptions...).Label(i18n.T("Sidebar")).Changed() {
			actions.setSidebarChoice(state.SidebarValues[local.sidebar])
		}
	})
}
