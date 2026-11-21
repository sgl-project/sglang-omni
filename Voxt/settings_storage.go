package main

// Settings → Storage (AudioSmith's, for VoxtNew's files): what takes
// room, opening the folders, moving the data folder and the reset tiers.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type settingsStorageLocalUI struct {
	resetAsk  string // "settings" | "everything": the confirmation shown
	resetText string // typed confirmation
}

// storageSettings is Settings → Storage.
func storageSettings(c *ui.Context, state settingsStorageViewState, actions settingsStorageActions, local *settingsStorageLocalUI) {
	t := c.Theme()
	muted := func(s string) { ui.Text(c, s).FontSize(px(t, fsCaption)).TextColor(t.TextMuted) }
	open := func(key, dir string) {
		if dir != "" && state.CanOpen && ui.Button(c.Key(key), i18n.T("Open")).Clicked() {
			actions.openStorageFolder(dir)
		}
	}

	sectionTitle(c, i18n.T("Data folder"))
	settingRow(c, "st-data", state.DataPath, i18n.T("Your history and its recordings, your dictionary and the database."), func() {
		open("st-open-data", state.DataDirectory)
		if ui.Button(c.Key("st-move"), i18n.T("Move…")).Disabled(state.Moving).Clicked() {
			actions.chooseStorageDataFolder()
		}
	})
	if state.RecordingsPath != "" {
		// Recordings kept elsewhere (history_settings.go).
		settingRowNote(c, "st-audio", i18n.T("Recordings folder"), state.RecordingsPath, func() {
			if ui.Button(c.Key("st-audio-settings"), i18n.T("History settings…")).Clicked() {
				actions.showHistorySettings()
			}
		})
	}
	if state.PreviousPath != "" {
		settingRowNote(c, "st-prev", i18n.T("The previous data folder is still there"), state.PreviousPath, func() {
			if ui.Button(c.Key("st-prev-rm"), i18n.T("Remove it")).Clicked() {
				actions.removePrevious()
			}
		})
	}
	if state.FailedMove != "" {
		settingRowNote(c, "st-failed", i18n.T("The last move failed"), state.FailedMove, func() {
			if ui.Button(c.Key("st-failed-ok"), i18n.T("Dismiss")).Clicked() {
				actions.dismissMoveFailure()
			}
		})
	}
	if state.MoveError != "" {
		ui.Text(c, state.MoveError).FontSize(px(t, fsCaption)).TextColor(t.Danger)
	}

	modelFolderRows(c, state, actions, open)

	sectionTitle(c, i18n.T("What takes up space"))
	if state.Report == nil {
		ui.Row(c).Gap(px(t, sp2)).Children(func() {
			ui.Spinner(c)
			muted(i18n.T("Measuring…"))
		})
	} else {
		muted(i18n.T("Data folder: %s · free on its disk: %s", state.Report.DataSize, state.Report.FreeSize))
		ui.Grid(c.Key("st-cats")).ColumnTracks(ui.Fr(1), ui.Fixed(px(t, 90)), ui.Fixed(px(t, 80))).GapX(px(t, sp3)).GapY(px(t, 6)).AlignItems(ui.Center).Children(func() {
			for _, category := range state.Report.Categories {
				ui.Text(c, category.Title)
				ui.Text(c, category.Size).TextColor(t.TextMuted).TextAlign(ui.End)
				ui.Row(c.Key("cat-" + category.ID)).Children(func() { open("open", category.Directory) })
			}
		})
	}

	sectionTitle(c, i18n.T("Reset"))
	settingRow(c, "st-reset-look", i18n.T("Reset the look"), i18n.T("Theme, size, sidebar and language."), func() {
		if ui.Button(c.Key("st-reset-look"), i18n.T("Reset")).Clicked() {
			actions.resetLook()
		}
	})
	settingRow(c, "st-reset-settings", i18n.T("Reset all settings"), i18n.T("Every setting goes back to its default. Your history and dictionary stay."), func() {
		if ui.Button(c.Key("st-reset-settings"), i18n.T("Reset…")).Clicked() {
			local.resetAsk = "settings"
		}
	})
	settingRow(c, "st-erase", i18n.T("Erase everything"), i18n.T("Deletes all your data and settings, then restarts Voxt from scratch."), func() {
		if ui.Button(c.Key("st-erase"), i18n.T("Erase…")).Disabled(state.Erasing).Clicked() {
			actions.askErase()
		}
	})
	storageDialogs(c, state, actions, local)
}

// storageDialogs are the confirmations and the move's progress.
func storageDialogs(c *ui.Context, state settingsStorageViewState, actions settingsStorageActions, local *settingsStorageLocalUI) {
	t := c.Theme()
	open := local.resetAsk == "settings"
	ui.Modal(c.Key("st-ask-settings"), &open, func() {
		ui.Text(c, i18n.T("Reset all settings?")).Bold()
		ui.Text(c, i18n.T("Every setting goes back to its default. Your history and dictionary stay.")).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("st-ask-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				local.resetAsk = ""
			}
			if ui.Button(c.Key("st-ask-ok"), i18n.T("Reset settings")).Clicked() {
				local.resetAsk = ""
				actions.resetSettings()
			}
		})
	})
	if !open && local.resetAsk == "settings" {
		local.resetAsk = ""
	}
	eraseDialog(c, state, actions, local)
	modelMoveDialog(c, state, actions)
	moving := state.Moving
	ui.DialogBase(c.Key("st-moving"), &moving, func(backdrop, panel ui.Element) {
		backdrop.Background(ui.RGBA(0, 0, 0, 0.3))
		panel.Width(px(t, 440)).Radius(px(t, radiusCard)).Background(t.Background).Border(1, t.Border).Padding(px(t, sp5)).Label(i18n.T("Moving the data folder"))
		ui.Column(c).Gap(px(t, 10)).Children(func() {
			ui.Text(c, i18n.T("Moving the data folder")).Bold()
			ui.Text(c, i18n.T("Copying to %s. Once the copy is checked, Voxt restarts to finish the move.", state.MovePath)).FontSize(px(t, fsCaption))
			ui.Progress(c, state.MoveProgress)
			ui.Text(c, state.MoveProgressText).FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
			ui.Row(c).Justify(ui.End).Children(func() {
				if ui.Button(c.Key("st-move-cancel"), i18n.T("Cancel")).Disabled(!state.CanCancelMove).Clicked() {
					actions.cancelStorageDataMove()
				}
			})
		})
	})
}
