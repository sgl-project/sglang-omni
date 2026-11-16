package main

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type settingsUpdatesActions interface {
	checkForUpdates()
	relaunch()
	setAutoCheck(on bool)
	setBeta(on bool)
	setAutoInstall(on bool)
}

type settingsUpdatesLocalUI struct {
	AutoCheck   bool
	AutoInstall bool
	Beta        bool
}

// updateSettingsSection is the Updates part of Settings → About.
func updateSettingsSection(c *ui.Context, state settingsUpdatesViewState, actions settingsUpdatesActions, local *settingsUpdatesLocalUI) {
	sectionTitle(c, i18n.T("Updates"))
	if !state.Present {
		caption(c, i18n.T("This version does not update automatically."))
		return
	}
	if state.BundleUpdated {
		settingRowNote(c, "about-relaunch", i18n.T("Update installed"), i18n.T("The new version runs once Voxt restarts."), func() {
			if primaryButton(c.Key("about-relaunch"), i18n.T("Restart Voxt")).Clicked() {
				actions.relaunch()
			}
		})
	}
	settingRowNote(c, "about-update", i18n.T("Check for updates"), state.Description, func() {
		if ui.Button(c.Key("about-update"), i18n.T("Check for Updates…")).Clicked() {
			actions.checkForUpdates()
		}
	})
	if !state.Known || !state.Enabled {
		return
	}
	settingRow(c, "about-update-auto", i18n.T("Check automatically"), state.AutoCheckText, func() {
		if ui.Switch(c.Key("about-update-auto-switch"), &local.AutoCheck).Label(i18n.T("Check for updates automatically")).Changed() {
			actions.setAutoCheck(local.AutoCheck)
		}
	})
	settingRowNote(c, "about-update-beta", i18n.T("Beta updates"), state.BetaNote, func() {
		if ui.Switch(c.Key("about-update-beta-switch"), &local.Beta).Label(i18n.T("Get beta versions")).Changed() {
			actions.setBeta(local.Beta)
		}
	})
	if state.CheckOnly {
		caption(c, i18n.T("Voxt does not install updates itself: install a new version from its disk image, or with brew upgrade --cask voxt."))
		return
	}
	settingRow(c, "about-update-install", i18n.T("Install updates automatically"), i18n.T("Voxt asks only to restart."), func() {
		if ui.Switch(c.Key("about-update-install-switch"), &local.AutoInstall).Label(i18n.T("Install updates without asking")).Disabled(state.AutoInstallDisabled).Changed() {
			actions.setAutoInstall(local.AutoInstall)
		}
	})
}
