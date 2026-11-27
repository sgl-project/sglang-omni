package main

import (
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type updatesLocalUI struct {
	autoCheck   bool
	autoInstall bool
	beta        bool
	relOpen     bool
	relMD       markdown.Options
}

// updateSettingsSection is the Updates part of Settings → About; a build
// without the updater only says that it does not update.
func updateSettingsSection(c *ui.Context, state updatesViewState, actions updatesActions, local *updatesLocalUI) {
	if !state.Available {
		sectionTitle(c, i18n.T("Updates"))
		caption(c, i18n.T("This version does not update automatically."))
		return
	}
	sectionTitle(c, i18n.T("Updates"))
	if state.BundleUpdated {
		settingRowNote(c, "about-relaunch", i18n.T("Update installed"), i18n.T("The new version runs once Voxt restarts."), func() {
			if primaryButton(c.Key("about-relaunch"), i18n.T("Restart Voxt")).Clicked() {
				actions.relaunch()
			}
		})
	}
	settingRowNote(c, "about-update", i18n.T("Check for updates"), state.CheckDescription, func() {
		if ui.Button(c.Key("about-update"), i18n.T("Check for Updates…")).Clicked() {
			actions.checkForUpdates()
		}
	})
	if !state.Known || !state.Enabled {
		return
	}
	settingRow(c, "about-update-auto", i18n.T("Check automatically"), state.AutomaticDescription, func() {
		if ui.Switch(c.Key("about-update-auto-switch"), &local.autoCheck).Label(i18n.T("Check for updates automatically")).Changed() {
			actions.setAutoCheck(local.autoCheck)
		}
	})
	settingRowNote(c, "about-update-beta", i18n.T("Beta updates"), state.BetaNote, func() {
		if ui.Switch(c.Key("about-update-beta-switch"), &local.beta).Label(i18n.T("Get beta versions")).Changed() {
			actions.setBeta(local.beta)
		}
	})
	if state.CheckOnly {
		caption(c, i18n.T("Voxt does not install updates itself: install a new version from its disk image, or with brew upgrade --cask voxt."))
		return
	}
	settingRow(c, "about-update-install", i18n.T("Install updates automatically"), i18n.T("Voxt asks only to restart."), func() {
		if ui.Switch(c.Key("about-update-install-switch"), &local.autoInstall).Label(i18n.T("Install updates without asking")).Disabled(!local.autoCheck).Changed() {
			actions.setAutoInstall(local.autoInstall)
		}
	})
}

// releaseDialog is Check for Updates…'s dialog: checking, then the new
// version, up to date, or the failure. Escape, × and Cancel close it.
func releaseDialog(c *ui.Context, state releaseViewState, actions releaseActions, local *updatesLocalUI) {
	if !local.relOpen || !state.Available {
		return
	}
	t := c.Theme()
	dialogFrame(c, "release", &local.relOpen, state.Title, state.Subtitle, 520, func() {
		switch {
		case !state.Done:
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Spinner(c).Size(px(t, 16), px(t, 16))
				caption(c, i18n.T("Looking for the newest release on GitHub."))
			})
		case state.ErrorText != "":
			ui.Text(c, state.ErrorText).Selectable()
		case state.Newer:
			if state.Notes != "" {
				markdown.View(c, state.Notes, &local.relMD)
			}
			caption(c, i18n.T("Download gets it from GitHub. With Homebrew, run brew upgrade --cask voxt instead."))
		}
	}, func() {
		ui.Spacer(c)
		switch {
		case !state.Done:
			if ui.Button(c.Key("release-cancel"), i18n.T("Cancel")).Clicked() {
				local.relOpen = false
			}
		case state.ErrorText != "":
			if ui.Button(c.Key("release-cancel"), i18n.T("Cancel")).Clicked() {
				local.relOpen = false
			}
			if primaryButton(c.Key("release-retry"), i18n.T("Try again")).Clicked() {
				actions.checkReleasesNow()
			}
		case state.Newer:
			if ui.Button(c.Key("release-later"), i18n.T("Later")).Clicked() {
				local.relOpen = false
			}
			if primaryButton(c.Key("release-download"), i18n.T("Download")).Clicked() {
				local.relOpen = false
				actions.downloadRelease(state.Download)
			}
		default:
			if primaryButton(c.Key("release-ok"), i18n.T("Done")).Clicked() {
				local.relOpen = false
			}
		}
	})
}

// sidebarUpdateControl is the update action at the foot of the sidebar:
// a translucent tinted bar in the expanded sidebar (or solid accent once
// installed and waiting to restart), and a circular icon button in the
// collapsed icon rail.
func sidebarUpdateControl(c *ui.Context, state *sidebarUpdateViewState, actions sidebarUpdateActions, rail bool) {
	if state == nil {
		return
	}
	t := c.Theme()
	k := tok(t)
	ic := icons.DownloadMinimalistic
	fg := k.AccentInk
	var bg, border ui.Color
	if state.Ready {
		ic = icons.Refresh
		fg = k.OnAccent
	}
	var b ui.Element
	if rail {
		b = ui.ButtonBase(c.Key("sidebar-update-rail")).
			Size(px(t, 32), px(t, 32)).
			Radius(px(t, 999)).
			Center().
			Shrink(0).
			Label(state.Label).
			Tooltip(state.Tooltip)
	} else {
		b = ui.ButtonBase(c.Key("sidebar-update-bar")).
			FillWidth().
			Padding(px(t, 7), px(t, 10)).
			Gap(px(t, sp2)).
			Radius(px(t, radiusControl)).
			AlignItems(ui.Center).
			Justify(ui.Start).
			Shrink(0).
			Label(state.Label).
			Tooltip(state.Tooltip)
	}
	if state.Ready {
		bg = k.Accent
		switch {
		case b.Pressed():
			bg = k.Accent.Mix(ui.Hex("#000000"), 0.16)
		case b.Hovered():
			bg = k.Accent.Mix(ui.Hex("#000000"), 0.08)
		}
		border = ui.Color{}
	} else {
		bg = k.AccentSoft
		switch {
		case b.Pressed():
			bg = k.SelectedHover
		case b.Hovered():
			bg = k.Selected
		}
		border = k.Accent.Alpha(0.24)
	}
	b.Background(bg).Border(1, border)
	b.Children(func() {
		if rail {
			ui.Icon(c, ic.Linear).Size(px(t, 16), px(t, 16)).TextColor(fg)
			return
		}
		ui.Icon(c, ic.Linear).Size(px(t, 15), px(t, 15)).TextColor(fg).Shrink(0)
		ui.Text(c, state.Label).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(fg).SingleLine().Ellipsis("…").Grow(1).MinWidth(0)
	})
	if b.Clicked() {
		if state.Ready {
			actions.relaunch()
		} else {
			actions.checkForUpdates()
		}
	}
}
