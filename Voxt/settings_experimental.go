package main

// Settings → Experimental (/settings/experimental): features that are
// still experiments, with their shortcuts and usage notes.

import (
	"strconv"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type settingsExperimentalLocalUI struct {
	sessionsOn     bool
	sessions       [4]keyRow
	sessionEditing [4]bool
}

func experimentalSettings(c *ui.Context, state settingsExperimentalViewState, actions settingsExperimentalActions, local *settingsExperimentalLocalUI) {
	t := c.Theme()
	settingRow(c, "ex-sessions", i18n.T("Sessions (experimental)"), i18n.T("Each session keeps its own card and conversation. The session row on the card, the Rewrite card included, switches sessions, adds one with +, deletes the current one with × and renames one on a double-click; the menu bar lists them too. Privacy's history keeps sessions and their titles after a restart."), func() {
		if ui.Switch(c.Key("ex-sessions-switch"), &local.sessionsOn).Label(i18n.T("Sessions (experimental)")).Changed() {
			actions.setSessionsOn(local.sessionsOn)
		}
	})
	if state.SessionsUnsaved {
		notice(c, "ex-sessions-unsaved", noticeWarn,
			i18n.T("History is off, so sessions end when Voxt quits. Turn on Keep a history in Privacy to keep them."),
			"", func() {
				if ui.Button(c.Key("ex-open-privacy"), i18n.T("Open Privacy")).Clicked() {
					actions.openShortcutFeature("/settings/privacy")
				}
			}, nil)
	}
	for n := range state.Sessions {
		if !state.SessionsOn {
			break
		}
		slot := &state.Sessions[n]
		i := slot.Slot
		settingRowUnder(c, "ex-session-"+strconv.Itoa(i+1), slot.Title, "", func() {
			if local.sessionEditing[i] {
				keysField(c, &local.sessions[i], slot.Title, func(keys string) {
					actions.setSessionKeys(i, keys)
					local.sessionEditing[i] = false
				}, slot.Recorder, actions)
				return
			}
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				kbd(c, slot.Recorder.DisplayedKeys)
				if ghostButton(c.Key("change"), i18n.T("Change…")).Clicked() {
					local.sessionEditing[i] = true
					local.sessions[i] = keyRow{}
				}
			})
		})
		if len(slot.Problems) > 0 {
			problemLines(c, slot.Problems)
		}
	}
}
