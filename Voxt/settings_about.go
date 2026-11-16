package main

// Settings → About (诊断与关于; AudioSmith's, for VoxtNew): versions
// and the machine, a self-check, diagnostic reports and legal notices.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// aboutUIContext keeps clipboard and link opening in the controller without
// making that controller a view file.
type aboutUIContext = ui.Context

// aboutDetails is Settings → About's version, system and legal information.
func aboutDetails(c *ui.Context, state settingsAboutViewState) {
	t := c.Theme()
	sectionTitle(c, i18n.T("About"))
	ui.Grid(c.Key("about-grid")).ColumnTracks(ui.Fixed(px(t, 150)), ui.Fr(1)).GapX(px(t, sp3)).GapY(px(t, 6)).Children(func() {
		for _, row := range state.Rows {
			ui.Text(c, row.Title).TextColor(t.TextMuted)
			ui.Text(c, row.Value).Selectable()
		}
	})
	// The legal notices: the copyright, the licence (Apache-2.0), the
	// source, and NOTICE (where the code came from, third parties).
	caption(c, i18n.T("Copyright © 2026 the SGLang-Omni Team."))
	caption(c, i18n.T("Voxt is licensed under the Apache License 2.0 and comes with no warranty."))
	ui.Row(c.Key("about-licence")).Gap(px(t, sp3)).AlignItems(ui.Center).Wrap().Children(func() {
		ui.Link(c.Key("about-licence-apache"), i18n.T("Licence"), licenceURL).FontSize(px(t, fsCaption))
		ui.Link(c.Key("about-licence-source"), i18n.T("Source code"), sourceURL).FontSize(px(t, fsCaption))
		ui.Link(c.Key("about-licence-notices"), i18n.T("Third-party notices"), noticeURL).FontSize(px(t, fsCaption))
	})
	ui.Row(c.Key("about-icons")).Gap(px(t, 6)).AlignItems(ui.Center).Wrap().Children(func() {
		caption(c, i18n.T("Icons: Solar by 480 Design, under CC BY 4.0 (@iconify-json/solar %s).", icons.SolarVersion))
		ui.Link(c, i18n.T("Licence"), "https://creativecommons.org/licenses/by/4.0/").FontSize(px(t, fsCaption))
	})
}

// aboutDiagnostics is Settings → About's self-check and problem reporting.
func aboutDiagnostics(c *ui.Context, state settingsAboutViewState, actions settingsAboutActions) {
	t := c.Theme()
	sectionTitle(c, i18n.T("Self-check"))
	settingRow(c, "about-check", i18n.T("Check Voxt"), "", func() {
		if state.SelfChecking {
			ui.Spinner(c)
		}
		if ui.Button(c.Key("about-check"), i18n.T("Run check")).Disabled(state.SelfChecking).Clicked() {
			actions.runSelfCheck()
		}
	})
	for _, check := range state.Checks {
		ui.Row(c.Key("check-" + check.ID)).Gap(px(t, sp2)).AlignItems(ui.Start).Children(func() {
			mark, col := icons.CheckCircle, t.Success
			switch check.Status {
			case "warn":
				mark, col = icons.DangerTriangle, t.Warning
			case "fail":
				mark, col = icons.CloseCircle, t.Danger
			}
			ui.Icon(c, mark.Linear).Size(px(t, 14), px(t, 14)).TextColor(col).Shrink(0).Margin(px(t, 2), 0, 0)
			ui.Text(c, check.Title).Width(px(t, 150))
			ui.Text(c, check.Detail).FontSize(px(t, fsCaption)).TextColor(t.TextMuted).Grow(1).MinWidth(0)
		})
	}

	sectionTitle(c, i18n.T("Report a problem"))
	settingRow(c, "about-copy", i18n.T("Copy diagnostics"), i18n.T("Versions, the Mac, models, folders and self-check results. No text or audio."), func() {
		if ui.Button(c.Key("about-copy"), i18n.T("Copy")).Clicked() {
			actions.copyDiagnostics(c)
		}
	})
	settingRow(c, "about-bundle", i18n.T("Save a diagnostic bundle"), i18n.T("A ZIP of the diagnostics, logs and settings to attach to a report."), func() {
		if ui.Button(c.Key("about-bundle"), i18n.T("Save…")).Clicked() {
			actions.saveBundle()
		}
	})
	settingRow(c, "about-bug", i18n.T("Report a bug"), i18n.T("Opens a new issue on GitHub in your browser, with the diagnostics filled in. Nothing is sent until you submit it."), func() {
		if ui.Button(c.Key("about-bug"), i18n.T("Report…")).Clicked() {
			actions.reportBug(c)
		}
	})
}
