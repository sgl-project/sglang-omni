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
	sectionTitle(c, i18n.Platform("What leaves this Mac", "What leaves this PC"))
	online := i18n.T("Otherwise, Voxt goes online only to download models and read its announcements. Reading them sends nothing about you.")
	if a.upd != nil { // a build with an update feed (updates.go)
		online = i18n.T("Otherwise, Voxt goes online only to check for updates, download models and read its announcements. Reading them sends nothing about you.")
	}
	ui.Column(c.Key("priv-list")).Gap(px(t, sp1)).Children(func() {
		for _, s := range []string{
			i18n.Platform("Your recordings are recognised on this Mac. If a feature uses a speech service you set up on the Speech models page, its recordings go to that service. Voxt has no server of its own.", "Your recordings are recognised on this PC. If a feature uses a speech service you set up on the Speech models page, its recordings go to that service. Voxt has no server of its own."),
			i18n.Platform("Text to clean up, translate or rewrite goes to the language model you set up on the Language models page. With a local server such as Ollama, it stays on this Mac. Otherwise it goes to that service.", "Text to clean up, translate or rewrite goes to the language model you set up on the Language models page. With a local server such as Ollama, it stays on this PC. Otherwise it goes to that service."),
			i18n.T("With ChatGPT (Codex), that text goes to OpenAI under your account and its terms."),
			online,
			i18n.T("Voxt reads the address of the current browser tab only to pick your app enhancement rules. The language model never sees it, only the app's name."),
			i18n.T("Voxt collects no usage data, crash reports or identifiers."),
			i18n.Platform("If you let programs on this Mac read the history (below), where that text goes next is up to them.", "If you let programs on this PC read the history (below), where that text goes next is up to them."),
		} {
			ui.Text(c, "• "+s).FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
		}
	})

	sectionTitle(c, i18n.Platform("Kept on this Mac", "Kept on this PC"))
	settingRow(c, "priv-history", i18n.T("Keep a history"), "", func() {
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
	settingRow(c, "priv-audio", i18n.T("Keep recordings"), i18n.T("When off, failed or cancelled recordings are still kept for 7 days so you can retry them."), func() {
		if ui.Switch(c.Key("priv-audio"), &a.settingsForm.keepAudio).Label(i18n.T("Save each recording")).Disabled(!a.prefs.Privacy.KeepHistory).Changed() {
			on := a.settingsForm.keepAudio
			a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepAudio = on })
		}
	})
	settingRow(c, "priv-hist-settings", i18n.T("History settings"), "", func() {
		if ui.Button(c.Key("priv-hist-settings"), i18n.T("Open…")).Clicked() {
			a.showHistorySettings()
		}
	})
	settingRow(c, "priv-clear", i18n.T("Clear history"), i18n.T("Deletes every entry and its recording. Your dictionary and settings stay."), func() {
		if ui.Button(c.Key("priv-clear"), i18n.T("Clear…")).Clicked() {
			a.clearHistoryAsk = true
		}
	})

	a.historyAccessSection(c) // history_access.go

	sectionTitle(c, i18n.T("Logs"))
	settingRow(c, "priv-log-text", i18n.T("Write text to the log"), i18n.T("When off, the log has no text, only lengths and timings. Turn it on only to report a problem."), func() {
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
		ui.Text(c, i18n.T("Clear history?")).Bold()
		ui.Text(c, i18n.T("All transcriptions, translations and rewrites are deleted, with their recordings. This cannot be undone.")).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("priv-clear-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				a.clearHistoryAsk = false
			}
			if recordButton(c.Key("priv-clear-ok"), i18n.T("Delete history")).Clicked() {
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
