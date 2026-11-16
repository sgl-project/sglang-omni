package main

// The History page's settings dialog: recordings and automatic clean-up.

import (
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type historySettingsLocalUI struct {
	open              bool
	cleanOn           bool
	daysSel, countSel string
	formFor           string
	exportSel         string
	pruneAsk          bool
	pruneN            int
	delAudioAsk       bool
	textScope         string
	textFormat        string
}

func historySettingsDialog(c *ui.Context, state *historySettingsViewState, actions historySettingsActions, local *historySettingsLocalUI) {
	if !state.Open {
		return
	}
	dialogFrame(c, "hist-set", &local.open, i18n.T("History settings"), "", 560, func() {
		histRecordingsSection(c, state, actions, local)
		histCleanupSection(c, state, actions, local)
	}, func() {
		ui.Spacer(c)
		if primaryButton(c.Key("hist-set-done"), i18n.T("Done")).Clicked() {
			local.open = false
		}
	})
	histSetQuestions(c, state, actions, local)
}

func histRecordingsSection(c *ui.Context, state *historySettingsViewState, actions historySettingsActions, local *historySettingsLocalUI) {
	t := c.Theme()
	sectionTitleText(c, i18n.T("Recordings"))
	move := i18n.T("Choosing another folder moves the recordings there.")
	settingRowNoteTip(c, "hist-set-dir", i18n.T("Storage location"), move, state.Directory, func() {
		if ui.Button(c.Key("hist-set-choose"), i18n.T("Choose…")).Disabled(state.Busy).Clicked() {
			actions.chooseHistoryRecordingFolder()
		}
		if iconButton(c.Key("hist-set-reveal"), icons.FolderOpen, i18n.T("Show in Finder")).Clicked() {
			actions.showHistoryRecordingFolder()
		}
		if state.CanUseDefault && ui.Button(c.Key("hist-set-default"), i18n.T("Use default")).Disabled(state.Busy).Clicked() {
			actions.useDefaultHistoryRecordingFolder()
		}
	})
	if state.MissingNotice != "" {
		notice(c, "hist-set-missing", noticeWarn, state.MissingNotice, "", nil, nil)
	}
	settingRowNote(c, "hist-set-usage", i18n.T("Saved recordings"), state.UsageText, func() {
		if ui.Button(c.Key("hist-set-delete"), i18n.T("Delete all…")).Disabled(!state.CanDelete).Clicked() {
			local.delAudioAsk = true
		}
	})
	settingRow(c, "hist-set-export", i18n.T("Export recordings"), "", func() {
		if !slices.Contains(state.RecordingScopes, local.exportSel) {
			local.exportSel = state.RecordingScope
		}
		ui.Select(c.Key("hist-set-scope"), &local.exportSel, state.RecordingScopes).Label(i18n.T("Which recordings")).Width(px(t, 200)).Shrink(1).MinWidth(px(t, 120))
		if ui.Button(c.Key("hist-set-export"), i18n.T("Export…")).Disabled(!state.CanExport).Clicked() {
			shown := len(state.RecordingScopes) > 1 && local.exportSel == state.RecordingScopes[1]
			actions.exportHistoryRecordings(shown)
		}
	})
	histExportTextRow(c, state, actions, local)
	if state.Busy {
		histWorkRow(c, state, actions)
	}
}

func histWorkRow(c *ui.Context, state *historySettingsViewState, actions historySettingsActions) {
	t := c.Theme()
	ui.Row(c.Key("hist-set-busy")).Gap(px(t, sp3)).AlignItems(ui.Center).Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 6)).Children(func() {
			caption(c, state.BusyLabel)
			ui.Progress(c, state.Progress)
		})
		if ui.Button(c.Key("hist-set-cancel"), i18n.T("Cancel")).Disabled(!state.CanCancel).Clicked() {
			actions.cancelHistoryWork()
		}
	})
}

func histCleanupSection(c *ui.Context, state *historySettingsViewState, actions historySettingsActions, local *historySettingsLocalUI) {
	t := c.Theme()
	sectionTitleText(c, i18n.T("Automatic clean-up")).Margin(px(t, sp4), 0, 0, 0)
	settingRow(c, "hist-set-clean", i18n.T("Clean up automatically"), "", func() {
		if ui.Switch(c.Key("hist-set-clean"), &local.cleanOn).Label(i18n.T("Clean up history")).Changed() {
			actions.setHistoryCleanup(local.cleanOn, historyChoiceValue(local.daysSel, state.DayOptions), historyChoiceValue(local.countSel, state.CountOptions))
		}
	})
	if !local.cleanOn {
		return
	}
	settingRow(c, "hist-set-days", i18n.T("Keep entries for"), "", func() {
		options := historyChoiceLabels(state.DayOptions)
		if ui.Select(c.Key("hist-set-days"), &local.daysSel, options).Label(i18n.T("Keep entries for")).Width(px(t, 160)).Changed() {
			actions.setHistoryCleanup(true, historyChoiceValue(local.daysSel, state.DayOptions), historyChoiceValue(local.countSel, state.CountOptions))
		}
	})
	settingRow(c, "hist-set-count", i18n.T("Keep at most"), i18n.T("For each feature."), func() {
		options := historyChoiceLabels(state.CountOptions)
		if ui.Select(c.Key("hist-set-count"), &local.countSel, options).Label(i18n.T("Keep at most")).Width(px(t, 160)).Changed() {
			actions.setHistoryCleanup(true, historyChoiceValue(local.daysSel, state.DayOptions), historyChoiceValue(local.countSel, state.CountOptions))
		}
	})
}

func historyChoiceLabels(options []historyChoiceViewState) []string {
	labels := make([]string, len(options))
	for index, option := range options {
		labels[index] = option.Label
	}
	return labels
}

func historyChoiceValue(label string, options []historyChoiceViewState) int {
	for _, option := range options {
		if option.Label == label {
			return option.Value
		}
	}
	return options[0].Value
}

func histSetQuestions(c *ui.Context, state *historySettingsViewState, actions historySettingsActions, local *historySettingsLocalUI) {
	t := c.Theme()
	if local.pruneAsk {
		ui.Modal(c.Key("hist-prune-ask"), &local.pruneAsk, func() {
			ui.Text(c, state.PrunePrompt).Bold()
			ui.Text(c, state.PruneDetail).Width(px(t, 380))
			ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
				if ui.Button(c.Key("hist-prune-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
					local.pruneAsk = false
				}
				if recordButton(c.Key("hist-prune-ok"), i18n.T("Delete")).Clicked() {
					local.pruneAsk = false
					actions.savePendingHistoryCleanup()
				}
			})
		})
		if !local.pruneAsk && local.formFor == "asking" {
			local.formFor = ""
		}
	}
	if local.delAudioAsk {
		ui.Modal(c.Key("hist-delaudio-ask"), &local.delAudioAsk, func() {
			ui.Text(c, i18n.T("Delete every recording?")).Bold()
			ui.Text(c, i18n.T("All recordings will be deleted. The entries and their text stay. This cannot be undone.")).Width(px(t, 380))
			ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
				if ui.Button(c.Key("hist-delaudio-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
					local.delAudioAsk = false
				}
				if recordButton(c.Key("hist-delaudio-ok"), i18n.T("Delete recordings")).Clicked() {
					local.delAudioAsk = false
					actions.deleteHistoryRecordings()
				}
			})
		})
	}
}
