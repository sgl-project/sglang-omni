package main

// Settings → Storage's model folder and Erase everything confirmation.

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// modelFolderRows are the model library's folder, shared by the family,
// and the room VoxtNew's models take in it. With no other ActNow app
// installed and nothing shared, the family goes unsaid: one row of VoxtNew's
// models.
func modelFolderRows(c *ui.Context, state settingsStorageViewState, actions settingsStorageActions, open func(key, dir string)) {
	t := c.Theme()
	sectionTitle(c, i18n.T("Model folder"))
	settingRowNote(c, "st-models", state.ModelPath, state.ModelNote, func() {
		open("st-open-models", state.ModelDirectory)
		if state.CanChangeModelFolder &&
			ui.Button(c.Key("st-models-change"), i18n.T("Change…")).Disabled(state.ModelMoving).Clicked() {
			actions.chooseStorageModelFolder()
		}
	})
	if state.ShowModelBackup {
		// Time Machine (family spec §7.12): the family's models.backup,
		// off by default.
		on := state.ModelBackup
		settingRow(c, "st-models-backup", i18n.T("Back up models with Time Machine"), state.ModelBackupHint, func() {
			if ui.Switch(c.Key("st-models-backup-switch"), &on).Label(i18n.T("Back up models with Time Machine")).Changed() {
				actions.setStorageModelBackup(on)
			}
		})
	}
	if len(state.ModelUsage) == 0 {
		return
	}
	ui.Grid(c.Key("st-models-use")).ColumnTracks(ui.Fr(1), ui.Fixed(px(t, 90)), ui.Fixed(px(t, 80))).GapX(px(t, sp3)).GapY(px(t, 6)).AlignItems(ui.Center).Children(func() {
		for _, row := range state.ModelUsage {
			ui.Text(c, row.Title)
			ui.Text(c, row.Size).TextColor(t.TextMuted).TextAlign(ui.End)
			ui.Box(c)
		}
	})
}

// modelMoveDialog is the model folder change's progress.
func modelMoveDialog(c *ui.Context, state settingsStorageViewState, actions settingsStorageActions) {
	t := c.Theme()
	moving := state.ModelMoving
	ui.DialogBase(c.Key("st-models-moving"), &moving, func(backdrop, panel ui.Element) {
		backdrop.Background(ui.RGBA(0, 0, 0, 0.3))
		panel.Width(px(t, 440)).Radius(px(t, radiusCard)).Background(t.Background).Border(1, t.Border).Padding(px(t, sp5)).Label(i18n.T("Moving the models"))
		ui.Column(c).Gap(px(t, 10)).Children(func() {
			ui.Text(c, i18n.T("Moving the models")).Bold()
			ui.Text(c, state.ModelMoveText).FontSize(px(t, fsCaption))
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Spinner(c)
				ui.Row(c).Grow(1)
				if ui.Button(c.Key("st-models-move-cancel"), i18n.T("Cancel")).Disabled(!state.CanCancelModelMove).Clicked() {
					actions.cancelStorageModelMove()
				}
			})
		})
	})
}

// eraseDialog confirms Erase everything.
func eraseDialog(c *ui.Context, state settingsStorageViewState, actions settingsStorageActions, local *settingsStorageLocalUI) {
	t := c.Theme()
	erase := local.resetAsk == "everything"
	ui.Modal(c.Key("st-ask-erase"), &erase, func() {
		ui.Text(c, i18n.T("Erase everything?")).Bold()
		ui.Text(c, i18n.T("This deletes your history and meetings with their recordings, your dictionary, app enhancement rules, every setting, the logs and the local models no other app uses. It cannot be undone. Voxt then restarts from scratch, and the models download again.")).Width(px(t, 420))
		if state.KeptModelsText != "" {
			ui.Text(c, state.KeptModelsText).Width(px(t, 420))
		}
		ui.Text(c, i18n.T("Type DELETE to confirm.")).FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
		ui.TextInput(c.Key("st-erase-text"), &local.resetText).Placeholder("DELETE").Width(px(t, 200)).Label(i18n.T("Type DELETE to confirm."))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("st-erase-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				local.resetAsk = ""
			}
			if recordButton(c.Key("st-erase-ok"), i18n.T("Erase and restart")).Disabled(strings.TrimSpace(local.resetText) != "DELETE" || state.Erasing).Clicked() {
				actions.eraseAll()
			}
		})
	})
	if !erase && local.resetAsk == "everything" {
		local.resetAsk = ""
	}
}
