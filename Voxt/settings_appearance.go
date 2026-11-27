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
	opening      bool
	openingSpeed float64 // the speed slider's stop, OpeningLength

	colorPalette         ui.Color // the picker's colour
	colorPaletteOf       string   // the saved palette colorPalette was last set from
	colorEditorOpen      bool
	windowTransparency   float64 // percent, the slider's (a preview while dragged)
	transparencyDragging bool
}

func appearanceSettings(c *ui.Context, state settingsAppearanceViewState, actions settingsAppearanceActions, local *settingsAppearanceLocalUI) {
	settingRow(c, "theme", i18n.T("Theme"), "", func() {
		if ui.Segmented(c.Key("theme-seg"), &local.theme, state.ThemeOptions...).Label(i18n.T("Theme")).Changed() {
			actions.setTheme(state.ThemeValues[local.theme])
		}
	})
	paletteSettings(c, state, actions, local)
	if state.Transparent {
		windowTransparencySettings(c, state, local)
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
	// Reduce motion skips the opening, so its switch rests, off, while it
	// is on (state.Opening).
	settingRow(c, "opening", i18n.T("Play the opening at start"), "", func() {
		sw := ui.Switch(c.Key("opening-switch"), &local.opening).Label(i18n.T("Play the opening at start")).Disabled(state.ReduceMotion)
		if sw.Changed() {
			actions.setOpening(local.opening)
		}
	})
	// Its speed shows only while it plays: a slider of three stops from
	// the turtle to the rabbit, as macOS's key repeat; a click on either
	// animal (Slower, Faster to a screen reader; its seconds as the
	// tooltip) goes to its end.
	if state.Opening && len(state.OpeningLengthValues) > 0 {
		settingRow(c, "opening-speed", i18n.T("Opening speed"), "", func() {
			t := c.Theme()
			last := len(state.OpeningLengthValues) - 1
			choose := func(stop int) {
				local.openingSpeed = float64(stop)
				actions.setOpeningLength(state.OpeningLengthValues[stop])
			}
			ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).Children(func() {
				if iconButton(c.Key("opening-slow"), icons.OpeningSlow, i18n.T("Slower")).Tooltip(state.OpeningLengthOptions[0]).Clicked() {
					choose(0)
				}
				slider := ui.Slider(c.Key("opening-speed-slider"), &local.openingSpeed, 0, float64(last)).Step(1).
					Label(i18n.T("Opening speed")).Width(px(t, 64)).Shrink(0)
				stop := min(max(int(local.openingSpeed+.5), 0), last)
				slider.Tooltip(state.OpeningLengthOptions[stop])
				if slider.Changed() {
					choose(stop)
				}
				if iconButton(c.Key("opening-fast"), icons.OpeningFast, i18n.T("Faster")).Tooltip(state.OpeningLengthOptions[last]).Clicked() {
					choose(last)
				}
			})
		})
	}
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

// paletteSettings is the Color row: Default or Custom unfolds the picker,
// with previews of the palette in light and dark.
func paletteSettings(c *ui.Context, state settingsAppearanceViewState, actions settingsAppearanceActions, local *settingsAppearanceLocalUI) {
	title := i18n.T("Color")
	if !settingsSearchShows(title, "Color palette") {
		return
	}
	if local.colorPaletteOf != state.Palette {
		// Reset color, a failed save or another window changed it.
		local.colorPalette, local.colorPaletteOf = ui.Hex(state.Palette), state.Palette
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
			value := i18n.T("Custom")
			if state.Palette == "" || state.Palette == state.DefaultPalette {
				value = i18n.T("Default")
			}
			ui.Text(c, value).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().Shrink(0)
		})
		trigger.Changed()
		panel := parts.Panel(func() {
			ui.Row(c).Wrap().Gap(px(t, sp4)).AlignItems(ui.Start).Children(func() {
				picker := ui.OpaqueColorPicker(c.Key("theme-color"), &local.colorPalette, func() {
					if quietButton(c.Key("palette-reset"), icons.Refresh, i18n.T("Reset color")).Disabled(state.Palette == state.DefaultPalette).Clicked() {
						actions.setPaletteColor(state.DefaultPalette)
					}
				}).Width(px(t, 246)).Label(title)
				if picker.Changed() {
					color := local.colorPalette
					local.colorPaletteOf = fmt.Sprintf("#%02x%02x%02x", color.R, color.G, color.B)
					actions.setPaletteColor(local.colorPaletteOf)
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

// windowTransparencySettings is the transparent theme's slider: it
// previews while dragged; the controller saves it once it rests.
func windowTransparencySettings(c *ui.Context, state settingsAppearanceViewState, local *settingsAppearanceLocalUI) {
	title := i18n.T("Window transparency")
	t := c.Theme()
	settingRow(c, "window-transparency", title, "", func() {
		slider := ui.Slider(c.Key("transparency-slider"), &local.windowTransparency, state.MinTransparency, state.MaxTransparency).
			Step(1).Label(title).Width(px(t, 200)).MinWidth(px(t, 100)).Shrink(1)
		_, _, local.transparencyDragging = slider.Dragged()
	})
}
