package main

// Settings → Appearance: the sidebar's layout (from AudioSmith), with an
// automatic one: icons only while the window is narrow (shell_nav.go).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type settingsAppearanceLocalUI struct {
	theme        int
	scale        int
	sidebar      int
	reduceMotion bool
}

func appearanceSettings(c *ui.Context, state settingsAppearanceViewState, actions settingsAppearanceActions, local *settingsAppearanceLocalUI) {
	settingRow(c, "theme", i18n.T("Theme"), "", func() {
		if ui.Segmented(c.Key("theme-seg"), &local.theme, state.ThemeOptions...).Label(i18n.T("Theme")).Changed() {
			actions.setTheme(state.ThemeValues[local.theme])
		}
	})
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

// sidebarRow chooses the sidebar's layout: automatic, names and icons,
// icons only.
func sidebarRow(c *ui.Context, state settingsAppearanceViewState, actions settingsAppearanceActions, local *settingsAppearanceLocalUI) {
	settingRow(c, "sidebar-style", i18n.T("Sidebar"), i18n.T("Automatic shows icons only while the window is narrow."), func() {
		if ui.Segmented(c.Key("sidebar-seg"), &local.sidebar, state.SidebarOptions...).Label(i18n.T("Sidebar")).Changed() {
			actions.setSidebarChoice(state.SidebarValues[local.sidebar])
		}
	})
}
