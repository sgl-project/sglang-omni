package main

// The Dictionary settings dialog: whether the dictionary is used, where
// its terms go (speech hints, the language model's glossary) and where
// suggestions come from (corrections, Scan history). Each switch carries
// what it does in its ⓘ (docs/behavior.md 7.10a).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func dictSettingsDialog(c *ui.Context, state *dictionaryViewState, actions dictionaryActions, local *dictionaryLocalUI) {
	if !local.settingsOpen {
		return
	}
	t := c.Theme()
	off := !state.Enabled
	dialogFrame(c, "dict-settings-dlg", &local.settingsOpen, i18n.T("Dictionary settings"), "", 480, func() {
		formRow(c, "dict-on-row", i18n.T("Use the dictionary"), "", func() {
			on := state.Enabled
			if ui.Switch(c.Key("dict-on"), &on).Label(i18n.T("Use the dictionary")).Changed() {
				actions.setDictionaryEnabled(on)
			}
		})
		dictSettingsRule(c)
		formRow(c, "dict-hints-row", i18n.T("Speech hints"), i18n.T("Terms go to the speech model as words to expect."), func() {
			hints := state.ASRHints
			if ui.Switch(c.Key("dict-hints"), &hints).Label(i18n.T("Speech hints")).Disabled(off).Changed() {
				actions.setDictionaryASRHints(hints)
			}
		})
		formRow(c, "dict-glossary-row", i18n.T("Glossary for the language model"),
			i18n.T("The language model writes terms as spelled here. With more than %d terms, it uses only those in the text, up to %d.", state.GlossaryAllUpTo, state.MaxGlossaryTerms), func() {
				glossary := state.Glossary
				if ui.Switch(c.Key("dict-glossary"), &glossary).Label(i18n.T("Glossary for the language model")).Disabled(off).Changed() {
					actions.setDictionaryGlossary(glossary)
				}
			})
		dictSettingsRule(c)
		formRow(c, "dict-learn-row", i18n.T("Learn from my corrections"),
			i18n.T("For %d seconds after inserting, Voxt reads the text around it. If you correct a word, it suggests your spelling. Nothing it reads is saved or sent.", state.LearnWatchSeconds), func() {
				learnOn := state.Learn
				if ui.Switch(c.Key("dict-learn"), &learnOn).Label(i18n.T("Learn from my corrections")).Disabled(off).Changed() {
					actions.setDictionaryLearn(learnOn)
				}
			})
		ui.Row(c.Key("dict-scan-row")).Wrap().GapX(px(t, sp4)).GapY(px(t, 6)).AlignItems(ui.Center).MinHeight(px(t, 32)).Children(func() {
			ui.Column(c).Grow(1).Basis(px(t, 200)).MinWidth(px(t, 140)).Gap(px(t, 2)).Children(func() {
				ui.Text(c, i18n.T("Scan history"))
				if !state.CanScan {
					caption(c, i18n.T("Choose a language model for Transcribe first"))
				}
			})
			if !state.CanScan {
				if ui.Button(c.Key("dict-scan-model"), i18n.T("Open Transcribe")).Clicked() {
					local.settingsOpen = false
					actions.openDictionaryTranscribe()
				}
				return
			}
			if ui.Button(c.Key("dict-scan"), i18n.T("Scan…")).Label(i18n.T("Scan history…")).Clicked() {
				local.settingsOpen = false
				actions.openDictionaryScan()
			}
		})
	}, func() {
		ui.Spacer(c)
		if primaryButton(c.Key("dict-settings-done"), i18n.T("Done")).Clicked() {
			local.settingsOpen = false
		}
	})
}

// dictSettingsRule is a hairline between the dialog's groups of rows.
func dictSettingsRule(c *ui.Context) {
	ui.Box(c).Height(1).Shrink(0).Background(tok(c.Theme()).Line)
}
