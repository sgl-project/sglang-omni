package main

// Settings → Privacy: what leaves this Mac (text, and a screenshot of the
// window or the copied image when the user turns that on, to the language
// model the user set up; the speech models run here; the network is also
// for update checks and model downloads), and what VoxtNew keeps here:
// the history, its recordings, the text in the logs.
// Where the recordings go and how long the history keeps entries are in
// the History settings dialog (history_settings.go), which a row opens.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type settingsPrivacyLocalUI struct {
	keepHistory                bool
	keepAudio                  bool
	logText                    bool
	sendWindow                 bool
	sendClipboard              bool
	sendClipboardWithSelection bool
	clearHistoryAsk            bool
}

func privacySettings(c *ui.Context, state settingsPrivacyViewState, actions settingsPrivacyActions, local *settingsPrivacyLocalUI, historyLocal *historyAccessLocalUI) {
	t := c.Theme()
	sectionTitle(c, i18n.T("What leaves this Mac"))
	ui.Column(c.Key("priv-list")).Gap(px(t, sp1)).Children(func() {
		for _, s := range []string{
			i18n.T("Your recordings are recognised on this Mac. If a feature uses a speech service you set up in Settings → Speech models, its recordings go to that service. Voxt has no server of its own."),
			i18n.T("The language model you set up in Settings → Language models can get text Voxt recognised, text you selected or typed, meeting transcripts and earlier conversation content, your speech context, your rules and enabled dictionary content. With the switches below on, it can also get a screenshot of the window or an image you copied."),
			i18n.T("A request to a language-model server on this Mac, such as Ollama, stays on this Mac. A request to an external API goes to the provider handling it. If Translate uses your fallback model, that provider gets the same request too."),
			i18n.T("With ChatGPT (Codex), that text goes to OpenAI under your account and its terms."),
			state.Online,
			i18n.T("Voxt reads the address of the current browser tab only to pick your app enhancement rules. The language model never sees it, only the app's name."),
			i18n.T("Voxt collects no usage data, crash reports or identifiers."),
			i18n.T("If you let programs on this Mac read the history (below), where that text goes next is up to them."),
		} {
			ui.Text(c, "• "+s).FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
		}
	})
	settingRow(c, "priv-window", i18n.T("Send a screenshot"), i18n.T("For language models that read images: when you start speaking, Voxt takes a screenshot of the window you speak into and sends it to the language model with the text, so names on screen come out right. Needs Screen Recording permission. A model that takes no images gets the text alone."), func() {
		if ui.Switch(c.Key("priv-window"), &local.sendWindow).Label(i18n.T("Send the window to the language model")).Changed() {
			actions.setSendWindow(local.sendWindow)
		}
	})
	settingRow(c, "priv-copied", i18n.T("Send a copied image"), i18n.T("For language models that read images: when you rewrite, Voxt sends the image you copied (⌘C) to the language model with what you said. With text selected, only the text goes unless Text and a copied image together is also on. Copied text, and copied files that are not images, are not sent. A model that takes no images gets the text alone."), func() {
		if ui.Switch(c.Key("priv-copied"), &local.sendClipboard).Label(i18n.T("Send the copied image with Rewrite")).Changed() {
			actions.setSendClipboardImage(local.sendClipboard)
		}
	})
	caption(c, i18n.T("With both on, the copied image goes, not the screenshot."))
	settingRow(c, "priv-copied-with-text", i18n.T("Text and a copied image together"), i18n.T("Off by default: a text selection sends only the text, so an image you copied earlier cannot take over a text rewrite. Turn it on to send the selected text and the copied image together."), func() {
		if ui.Switch(c.Key("priv-copied-with-text"), &local.sendClipboardWithSelection).Label(i18n.T("Send both with Rewrite")).Changed() {
			actions.setSendClipboardImageWithSelection(local.sendClipboardWithSelection)
		}
	})

	sectionTitle(c, i18n.T("Kept on this Mac"))
	settingRow(c, "priv-history", i18n.T("Keep a history"), i18n.T("When off, the menu bar and Dock menus do not list cards to reopen."), func() {
		if ui.Switch(c.Key("priv-history"), &local.keepHistory).Label(i18n.T("Save each result")).Changed() {
			actions.setKeepHistory(local.keepHistory)
		}
	})
	settingRow(c, "priv-audio", i18n.T("Keep recordings"), i18n.T("When off, failed or cancelled recordings are still kept for 7 days so you can retry them."), func() {
		if ui.Switch(c.Key("priv-audio"), &local.keepAudio).Label(i18n.T("Save each recording")).Disabled(!state.KeepAudioEnabled).Changed() {
			actions.setKeepAudio(local.keepAudio)
		}
	})
	settingRow(c, "priv-hist-settings", i18n.T("History settings"), "", func() {
		if ui.Button(c.Key("priv-hist-settings"), i18n.T("Open…")).Clicked() {
			actions.showHistorySettings()
		}
	})
	settingRow(c, "priv-clear", i18n.T("Clear history"), i18n.T("Deletes every entry and its recording. Your dictionary and settings stay."), func() {
		if ui.Button(c.Key("priv-clear"), i18n.T("Clear…")).Clicked() {
			local.clearHistoryAsk = true
		}
	})

	historyAccessSection(c, &state.HistoryAccess, actions, historyLocal) // history_access.go

	sectionTitle(c, i18n.T("Logs"))
	settingRow(c, "priv-log-text", i18n.T("Write text to the log"), i18n.T("When off, the log has no text, only lengths and timings. Turn it on only to report a problem."), func() {
		if ui.Switch(c.Key("priv-log-text"), &local.logText).Label(i18n.T("Log what was said")).Changed() {
			actions.setLogText(local.logText)
		}
	})
	clearHistoryDialog(c, actions, local)
}

// clearHistoryDialog confirms Clear the history.
func clearHistoryDialog(c *ui.Context, actions settingsPrivacyActions, local *settingsPrivacyLocalUI) {
	t := c.Theme()
	ui.Modal(c.Key("priv-clear-ask"), &local.clearHistoryAsk, func() {
		ui.Text(c, i18n.T("Clear history?")).Bold()
		ui.Text(c, i18n.T("All transcriptions, translations and rewrites are deleted, with their recordings. This cannot be undone.")).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("priv-clear-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				local.clearHistoryAsk = false
			}
			if recordButton(c.Key("priv-clear-ok"), i18n.T("Delete history")).Clicked() {
				local.clearHistoryAsk = false
				actions.clearHistory()
			}
		})
	})
}
