package main

// The shortcuts other ActNow apps also use (family spec §10.4): a line
// "Also used by Auralink (Transcribe)" under each of VoxtNew's shortcuts
// an installed family app uses too (Settings → Shortcuts, the setup
// guide), the same under the recorder of a feature's page while it shows
// keys another app uses (saving stays allowed), and a prompt, once per
// pair, for the conflicts VoxtNew is the later app of (the one that got
// its keys last shows it, the other only warns).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// usedByView is the recorder's warning: the lines of usedByLines.
func usedByView(c *ui.Context, lines []string) {
	for _, line := range lines {
		warnLine(c, line)
	}
}

// shortcutPromptDialog is the prompt: the keys, the other app and its
// function; Change shortcut (the feature's page), Open <app> settings
// (its manifest's link, when it has one), Ignore.
func shortcutPromptDialog(c *ui.Context, state *shortcutPromptViewState, actions shortcutPromptActions) {
	if state == nil {
		return
	}
	t := c.Theme()
	open, answered := true, false
	ui.Modal(c.Key("sc-prompt"), &open, func() {
		ui.Text(c, i18n.T("Shortcut used by another app")).Bold()
		ui.Text(c, i18n.T("%s is also the %s shortcut in %s. Pressing it will trigger both apps.",
			state.Keys, state.Function, state.OtherApp)).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ghostButton(c.Key("sc-prompt-ignore"), i18n.T("Ignore")).Clicked() {
				answered = true
				actions.answerShortcutPrompt()
			}
			if state.Link != "" && ui.Button(c.Key("sc-prompt-other"), i18n.T("Open %s settings", state.OtherApp)).Clicked() {
				answered = true
				actions.openShortcutPromptLink(c.Services(), state.Link)
			}
			if primaryButton(c.Key("sc-prompt-change"), i18n.T("Change shortcut")).AutoFocus().Clicked() {
				answered = true
				actions.changeShortcutPrompt(state.FeaturePath)
			}
		})
	})
	if !open && !answered { // Escape: as Ignore
		actions.answerShortcutPrompt()
	}
}
