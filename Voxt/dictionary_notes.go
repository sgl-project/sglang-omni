package main

// The Personal dictionary view: Markdown and find. Each feature page's
// preview shows it in that feature's whole prompt (feature_prompts.go).

import (
	"math"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func notesPlaceholder() string {
	return i18n.T("| Standard spelling | Correction mapping | Condition |\n|---|---|---|\n| 101; I-80 | `one oh one` → 101 | Bay Area highways |\n| Roth | `wrath` → Roth | retirement accounts |\n| NVDA | — | tickers, all caps |")
}

// dictGuide says where a term applies, folded under the switches
// (docs/behavior.md 5.16, 7.9, 7.9a).
func dictGuide(c *ui.Context, state *dictionaryViewState, local *dictionaryLocalUI) {
	collapsible(c.Key("dict-guide"), i18n.T("How the dictionary works"), &local.guideOpen, nil, func() {
		caption(c, i18n.T("Each enabled term is used in three places:"))
		caption(c, i18n.T("Speech recognition: with Speech hints on, terms go to the speech model as words to expect. Aliases are not sent."))
		caption(c, i18n.T("Language model: with Glossary for the language model on, the model writes terms as spelled here. With more than %d terms, only those found in the text are sent, up to %d.", state.GlossaryAllUpTo, state.MaxGlossaryTerms))
		caption(c, i18n.T("Aliases: misspellings Voxt replaces with the term, even without a language model."))
		caption(c, i18n.T("Rules a term cannot hold, such as when a spelling applies, go in the Personal dictionary: Markdown notes for the language model only, with an example of the prompt."))
	})
}

func dictionaryNotesView(c *ui.Context, state *dictionaryNotesViewState, actions dictionaryActions, local *dictionaryLocalUI) {
	t := c.Theme()
	ui.Scroll(c.Key("dict-notes-scroll")).Grow(1).MinHeight(0).Children(func() {
		ui.Column(c).Padding(px(t, sp2), px(t, sp6), px(t, sp6)).Gap(px(t, sp2)).Children(func() {
			off := !state.Enabled
			formRow(c, "dict-notes-on-row", i18n.T("Add to the language model prompt"), "", func() {
				on := !state.NotesOff
				if ui.Switch(c.Key("dict-notes-on"), &on).Label(i18n.T("Add to the language model prompt")).Disabled(off).Changed() {
					actions.setDictionaryNotesOff(!on)
				}
			})
			if off {
				notice(c, "dict-notes-off", noticeInfo, i18n.T("The dictionary is off, so the personal dictionary is not sent."), "", nil, nil)
			}
			matches := findFolded(local.notes.text, local.notesQuery)
			moved := local.notesFind.advance(len(matches))
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				local.notesFind.field(c, "dict-notes-find", &local.notesQuery, i18n.T("Find in the personal dictionary"), px(t, 320)).Label(i18n.T("Search personal dictionary"))
				local.notesFind.count(c, local.notesQuery, len(matches))
			})
			input := ui.TextArea(c.Key("dict-notes"), &local.notes.text).Label(i18n.T("Personal dictionary in Markdown")).Placeholder(notesPlaceholder()).Lines(8, math.MaxInt32)
			if len(matches) > 0 {
				ranges := make([]ui.TextRange, len(matches))
				for index, match := range matches {
					ranges[index] = ui.TextRange{Start: match[0], End: match[1], Color: tok(t).Danger, Weight: 700}
					if index == local.notesFind.cur-1 {
						ranges[index].Weight = 900
					}
				}
				input.TextRanges(ranges...)
			}
			if moved {
				input.SetTextSelection(matches[local.notesFind.cur-1][0], matches[local.notesFind.cur-1][1])
			}
			actions.autosaveDictionaryNotes(input.Changed(), input.Focused())
		})
	})
}
