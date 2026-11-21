package main

// Settings → Privacy's Other programs section.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type historyAccessLocalUI struct {
	historyAccess    bool
	historySelection bool
	ask              bool
	listOpen         bool
}

func historyAccessSection(c *ui.Context, state *historyAccessViewState, actions historyAccessActions, local *historyAccessLocalUI) {
	sectionTitle(c, i18n.T("Other programs"))
	settingRow(c, "priv-access", i18n.Platform("Let programs on this Mac read the history", "Let programs on this PC read the history"), i18n.T("When off, Voxt's read command stops working. Programs running under your account can still open Voxt's data files directly."), func() {
		if ui.Switch(c.Key("priv-access"), &local.historyAccess).Label(i18n.T("Allow reading")).Changed() {
			if local.historyAccess {
				local.historyAccess = false
				local.ask = true
			} else {
				actions.setHistoryAccess(false)
			}
		}
	})
	settingRow(c, "priv-access-sel", i18n.T("Also include the text you selected"), i18n.T("When off, text you selected is not included in what programs read or in Export text."), func() {
		if ui.Switch(c.Key("priv-access-sel"), &local.historySelection).Label(i18n.T("Include selected text")).Disabled(!state.On).Changed() {
			actions.setHistorySelection(local.historySelection)
		}
	})
	if state.ShowLastRead {
		settingRowNote(c, "priv-access-last", i18n.T("Last read"), state.LastRead, func() {
			if ui.Button(c.Key("priv-access-log"), i18n.T("Read log…")).Clicked() {
				local.listOpen = true
			}
		})
	}
	if state.On {
		settingRowNote(c, "priv-access-cmd", i18n.T("Command line"), state.Command, func() {
			if ui.Button(c.Key("priv-access-copy"), i18n.T("Copy")).Clicked() {
				actions.copyHistoryAccessCommand()
			}
			if ghostButton(c.Key("priv-access-docs"), i18n.T("How to use it")).Clicked() {
				actions.openHistoryAccessDocs()
			}
		})
	}
	historyAccessDialogs(c, state, actions, local)
}

func historyAccessDialogs(c *ui.Context, state *historyAccessViewState, actions historyAccessActions, local *historyAccessLocalUI) {
	t := c.Theme()
	if local.ask {
		ui.Modal(c.Key("priv-access-ask"), &local.ask, func() {
			ui.Text(c, i18n.Platform("Let programs on this Mac read the history?", "Let programs on this PC read the history?")).Bold()
			ui.Text(c, i18n.T("Any program you run can then read the text Voxt transcribed, translated and rewrote and the app names, but not the recordings or web addresses. Voxt cannot see where a program sends it.")).Width(px(t, 380))
			ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
				if ui.Button(c.Key("priv-access-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
					local.ask = false
				}
				if primaryButton(c.Key("priv-access-ok"), i18n.T("Turn on")).Clicked() {
					local.ask = false
					actions.setHistoryAccess(true)
				}
			})
		})
	}
	if local.listOpen {
		dialogFrame(c, "priv-access-reads", &local.listOpen, i18n.T("Read log"), i18n.T("Recent reads by other programs. No text is kept here."), 480, func() {
			if len(state.ReadLines) == 0 {
				caption(c, i18n.T("No reads yet"))
			}
			for _, line := range state.ReadLines {
				caption(c, line)
			}
		}, func() {
			ui.Spacer(c)
			if primaryButton(c.Key("priv-access-reads-done"), i18n.T("Done")).Clicked() {
				local.listOpen = false
			}
		})
	}
}
