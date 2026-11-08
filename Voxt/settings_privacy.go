package main

// Settings → Privacy: what leaves this Mac (only text, to the language
// model the user set up; the speech models run here; the network is also
// for update checks and model downloads), and what VoxtNew keeps here:
// the history, its recordings, the text in the logs.
// Where the recordings go and how long the history keeps entries are in
// the History settings dialog (history_settings.go), which a row opens.

import (
	"context"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) privacySettings(c *ui.Context) {
	t := c.Theme()
	sectionTitle(c, i18n.T("What leaves this Mac"))
	online := i18n.T("Otherwise Voxt goes online only to download models and to read Voxt's announcements, which sends nothing about you.")
	if a.upd != nil { // a build with an update feed (updates.go)
		online = i18n.T("Otherwise Voxt goes online only to check for updates, to download models and to read Voxt's announcements, which sends nothing about you.")
	}
	ui.Column(c.Key("priv-list")).Gap(px(t, sp1)).Children(func() {
		for _, s := range []string{
			i18n.T("Your recordings are recognised on this Mac, unless a feature uses a speech service you set up (the Speech models page): its recordings then go to that service. Voxt has no server of its own."),
			i18n.T("The text to clean up, translate or rewrite goes to the language model you set up (the Language models page): it stays on this Mac with a server here, such as Ollama, and goes to the service otherwise."),
			online,
			i18n.T("The address of the browser tab in front is read only to pick your app enhancement rules: the language model never sees it, only the app's name."),
			i18n.T("Voxt collects no usage data, crash reports or identifiers."),
		} {
			ui.Text(c, "• "+s).FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
		}
	})

	sectionTitle(c, i18n.T("Kept on this Mac"))
	settingRow(c, "priv-history", i18n.T("Keep a history"), i18n.T("Each transcription, translation and rewrite is saved with its source text, so you can find, copy or redo it."), func() {
		if ui.Switch(c.Key("priv-history"), &a.settingsForm.keepHistory).Label(i18n.T("Save each result")).Changed() {
			on := a.settingsForm.keepHistory
			a.updateSettings(func(s *settings.Settings) {
				s.Privacy.KeepHistory = on
				if !on {
					s.Privacy.KeepAudio = false
				}
			})
		}
	})
	settingRow(c, "priv-audio", i18n.T("Keep the recordings"), i18n.T("Saves the audio with each history entry, to listen again or recognise it once more. When this is off, a recording whose recognition failed, or that you cancelled, is still kept for 7 days, so that you can retry it."), func() {
		if ui.Switch(c.Key("priv-audio"), &a.settingsForm.keepAudio).Label(i18n.T("Save each recording")).Disabled(!a.prefs.Privacy.KeepHistory).Changed() {
			on := a.settingsForm.keepAudio
			a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepAudio = on })
		}
	})
	settingRow(c, "priv-hist-settings", i18n.T("History settings"), i18n.T("Where the recordings are kept, exporting them, and the automatic clean-up of old entries."), func() {
		if ui.Button(c.Key("priv-hist-settings"), i18n.T("Open…")).Clicked() {
			a.showHistorySettings()
		}
	})
	settingRow(c, "priv-clear", i18n.T("Clear the history"), i18n.T("Deletes every entry and its recording. Your dictionary and settings stay."), func() {
		if ui.Button(c.Key("priv-clear"), i18n.T("Clear…")).Clicked() {
			a.clearHistoryAsk = true
		}
	})

	sectionTitle(c, i18n.T("Logs"))
	settingRow(c, "priv-log-text", i18n.T("Write text to the log"), i18n.T("Off, the log records only lengths and timings of what you said and got back, never the text. Turn it on to report a problem, then off again."), func() {
		if ui.Switch(c.Key("priv-log-text"), &a.settingsForm.logText).Label(i18n.T("Log what was said")).Changed() {
			on := a.settingsForm.logText
			a.updateSettings(func(s *settings.Settings) { s.Privacy.LogText = on })
		}
	})
	a.clearHistoryDialog(c)
}

// clearHistoryDialog confirms Clear the history.
func (a *app) clearHistoryDialog(c *ui.Context) {
	t := c.Theme()
	ui.Modal(c.Key("priv-clear-ask"), &a.clearHistoryAsk, func() {
		ui.Text(c, i18n.T("Clear the history?")).Bold()
		ui.Text(c, i18n.T("Every transcription, translation and rewrite, and their recordings, are deleted. This cannot be undone.")).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("priv-clear-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				a.clearHistoryAsk = false
			}
			if recordButton(c.Key("priv-clear-ok"), i18n.T("Delete the history")).Clicked() {
				a.clearHistoryAsk = false
				a.clearHistory()
			}
		})
	})
}

// clearHistory deletes every history entry, in the background.
func (a *app) clearHistory() {
	h := a.lib.store.History()
	a.async(func() {
		n, err := h.DeleteAll(context.Background(), "")
		a.post(func() {
			if n > 0 {
				a.historyChanged() // history_page.go
			}
			if err != nil {
				a.toast(i18n.T("Could not clear the history: %v", err))
				return
			}
			a.toast(i18n.N(n, "Deleted %d entry", "Deleted %d entries", n))
		})
	})
}
