package main

// Settings → Appearance: the sidebar's layout (from AudioSmith), with an
// automatic one: icons only while the window is narrow (shell_nav.go).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
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
	settingRow(c, "sidebar-style", i18n.T("Sidebar"), i18n.T("Icons only leaves more room for the pages; their names show when the pointer rests on them. Automatic shows icons only while the window is narrow."), func() {
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
		if ui.Segmented(c.Key("theme-seg"), &a.theme, i18n.T("System"), i18n.T("Light"), i18n.T("Dark")).Label(i18n.T("Theme")).Changed() {
			theme := settings.Themes[a.theme]
			a.updateSettings(func(s *settings.Settings) { s.Appearance.Theme = theme })
		}
	})
	settingRow(c, "scale", i18n.T("Interface size"), i18n.T("Scales text and spacing."), func() {
		if ui.Segmented(c.Key("scale-seg"), &a.scale, scaleLabels...).Label(i18n.T("Interface size")).Changed() {
			scale := settings.Scales[a.scale]
			a.updateSettings(func(s *settings.Settings) { s.Appearance.Scale = scale })
		}
	})
	a.sidebarRow(c) // settings_appearance.go
	settingRow(c, "motion", i18n.T("Reduce motion"), i18n.T("Pages and panels change without animating. The system's setting is followed too."), func() {
		if ui.Switch(c.Key("motion-switch"), &a.reduceMotion).Label(i18n.T("Reduce motion")).Changed() {
			on := a.reduceMotion
			a.updateSettings(func(s *settings.Settings) { s.Appearance.ReduceMotion = on })
		}
	})
}
