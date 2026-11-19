package main

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func meetingDemoContent(host *meetingDemoHost, localize func()) mygo.Content {
	return ui.View(func(c *ui.Context) {
		if localize != nil {
			localize()
		}
		meetingDemoView(c, host)
	})
}

func meetingDemoView(c *ui.Context, host *meetingDemoHost) {
	fixture := host.current()
	ui.Column(c).Fill().Children(func() {
		ui.Row(c).Padding(8, 12).Gap(8).AlignItems(ui.Center).Shrink(0).Children(func() {
			if ghostButton(c.Key("demo-previous"), i18n.T("Back")).Clicked() {
				host.selectMeetingFixture(-1)
			}
			if ghostButton(c.Key("demo-next"), i18n.T("Continue")).Clicked() {
				host.selectMeetingFixture(1)
			}
			caption(c, fixture.name)
		})
		ui.Box(c).Grow(1).Children(func() {
			switch fixture.state.Kind {
			case "page":
				meetingsPage(c, fixture.state.Page, host, &host.pageUI, 800)
				meetingDeleteDialog(c, &host.pageUI, host)
			case "overlay":
				state := *fixture.state.Overlay
				state.Collapsed = host.overlayUI.collapsed
				state.ShowExpanded = state.ShowExpanded && !state.Collapsed
				if state.Visible {
					model := meetingOverlayLayoutModel(state, host.overlayUI.menu)
					meetingOverlayView(c, state, model, host, &host.overlayUI, meetingOverlayAnimation{}, host.faceStyle(model))
				}
			case "session-overlay":
				sessionDemoOverlayView(c, host)
			case "start":
				ui.Column(c).Padding(24).Children(func() {
					fillCard(c).Padding(16).Children(func() {
						ui.Text(c, fixture.state.Start.Text)
					})
				})
			case "history":
				historyPage(c, fixture.state.History, host, &host.historyUI, 800)
				historyDeleteDialog(c, &host.historyUI, host)
			case "history-settings":
				historySettingsDialog(c, fixture.state.HistorySettings, host, &host.historySettingsUI)
			case "history-access":
				ui.Scroll(c.Key("demo-history-access")).Grow(1).Children(func() {
					pageColumn(c).Children(func() {
						historyAccessSection(c, fixture.state.HistoryAccess, host, &host.historyAccessUI)
					})
				})
			case "updates-settings":
				ui.Scroll(c.Key("demo-updates-settings")).Grow(1).Children(func() {
					pageColumn(c).Children(func() {
						updateSettingsSection(c, *fixture.state.UpdatesSettings, host, &host.updatesUI)
					})
				})
			case "updates-release":
				releaseDialog(c, *fixture.state.UpdatesRelease, host, &host.updatesUI)
			case "home":
				homePage(c, fixture.state.Home, host, &host.historyUI, nil)
				historyDeleteDialog(c, &host.historyUI, host)
			case "onboarding":
				onboardingView(c, *fixture.state.Onboarding, host, &host.onboardingUI)
			case "appenhance":
				appEnhancementPage(c, host.appEnhancementState(), host, &host.appEnhancementUI, 800, host)

			case "dictionary":
				dictionaryPage(c, fixture.state.Dictionary, host, &host.dictionaryUI)
			case "notes":
				notesPage(c, fixture.state.Notes, host, &host.notesUI)
			case "remote":
				local := remoteLocalUI{wait: &host.remoteWaitUI, dialog: &host.remoteDialogUI}
				ui.Scroll(c.Key("demo-remote")).Grow(1).Children(func() {
					pageColumn(c).Children(func() {
						remoteSettingsView(c, fixture.state.Remote, host, &local, 800)
					})
				})
				if fixture.state.Remote.Dialog != nil { // shell.go draws it over the window in the app
					remoteProviderDialogView(c, fixture.state.Remote.Dialog, host, local.dialog)
				}
			case "remote-cards":
				ui.Scroll(c.Key("demo-remote-cards")).Grow(1).Children(func() {
					pageColumn(c).Children(func() {
						remoteProviderCards(c, fixture.state.RemoteCards, host, 800)
					})
				})
			case "remote-dialog":
				remoteProviderDialogView(c, fixture.state.RemoteDialog, host, &host.remoteDialogUI)
			case "remote-debug":
				state := host.remoteDebugState()
				llmDebugView(c, &state, host, &host.remoteDebugUI)
			case "models":
				state := host.modelsViewState()
				ui.Scroll(c.Key("demo-models")).Grow(1).Children(func() {
					pageColumn(c).Children(func() {
						modelsSettingsSection(c, &settingsModelsViewState{Models: *state}, host, &host.modelsUI)
					})
				})
				// The app draws these over Settings (shell_controller.go).
				modelDeleteDialogs(c, &state.Delete, host, &host.modelsUI)
				networkDialog(c, &state.Network, host, &host.modelsUI)
				modelAssignmentDialog(c, state.Assignment, host, &host.modelsUI.assign)
			case "feature":
				state := host.featureViewState(fixture.state.Feature)
				featurePage(c, &state, host, &host.feature.local, host)
			case "settings":
				settingsDialog(c, *fixture.state.Settings, host, host.settingsDialogUI)
			case "meeting-settings":
				dialogFrame(c, "mt-set", &host.pageUI.setOpen, i18n.T("Meeting settings"), "", 560, func() {
					state := *fixture.state.MeetingSettings
					meetingCaptureSettings(c, state, host, &host.settingsMeetingUI)
					meetingASRSettings(c, state, host)
					meetingLLMSettings(c, state, host)
					meetingCorrectionStyleSettings(c, state, host, &host.settingsMeetingUI)
					meetingConsent(c)
				}, func() {
					ui.Spacer(c)
					if primaryButton(c.Key("mt-set-done"), i18n.T("Done")).Clicked() {
						host.pageUI.setOpen = false
					}
				})
			case "shortcut-prompt":
				shortcutPromptDialog(c, fixture.state.ShortcutPrompt, host)
			case "key-recorder":
				ui.Column(c).Padding(24).Children(func() {
					keysField(c, &host.keyRow, i18n.T("Custom keys of shortcut %d", 1), host.saveKeys, *fixture.state.KeyRecorder, host)
				})
			case "shell":
				shellView(c, fixture.state.Shell, host, &host.shellUI, func() {
					ui.Column(c).Grow(1).Center().Children(func() { ui.Text(c, fixture.name) })
				}, func() {
					announcementDialog(c, fixture.state.Announcement, &host.announcementOpen)
				})
			case "shell-nav":
				sidebarFrame(c, fixture.state.Nav, fixture.state.Notices, host, host, &host.navUI, func() {
					ui.Column(c).Grow(1).Center().Children(func() { ui.Text(c, fixture.name) })
				})
			case "shell-notices":
				ui.Column(c).Width(260).Padding(8).Children(func() {
					noticeStack(c, fixture.state.Notices, host)
				})
			case "shell-logs":
				logsView(c, fixture.state.Logs, host, &host.logsUI, false)
			case "shell-palette":
				ui.Column(c).Grow(1).Center().Children(func() { ui.Text(c, fixture.name) })
				paletteView(c, fixture.state.Palette, host, &host.paletteUI)
			case "shell-sheet":
				shortcutSheetView(c, fixture.state.Sheet, &host.sheetOpen)
			case "shell-announcement":
				announcementDialog(c, fixture.state.Announcement, &host.announcementOpen)
			}
		})
	})
}
