package main

// The Personal dictionary view: Markdown for the language model, found
// with the search in the page's toolbar (dictionary_page.go). Each
// feature page's preview shows it in that feature's whole prompt
// (feature_prompts.go).

import (
	"math"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func notesPlaceholder() string {
	return i18n.T("| Standard spelling | Heard as | Condition |\n|---|---|---|\n| 101; I-80 | `one oh one` → 101 | Bay Area highways |\n| Roth | `wrath` → Roth | retirement accounts |\n| NVDA | — | tickers, all caps |")
}

// dictionaryNotesView is the switch that sends it, then the Markdown,
// with matches, the find's positions in it.
func dictionaryNotesView(c *ui.Context, state *dictionaryNotesViewState, actions dictionaryActions, local *dictionaryLocalUI, matches [][2]int) {
	t := c.Theme()
	moved := local.notesFind.advance(len(matches))
	ui.Scroll(c.Key("dict-notes-scroll")).Grow(1).MinHeight(0).Children(func() {
		ui.Column(c).Gap(px(t, sp3)).Children(func() {
			off := !state.Enabled
			if off {
				notice(c, "dict-notes-off", noticeInfo, i18n.T("The dictionary is off, so the personal dictionary is not used."), "", func() {
					if ghostButton(c.Key("dict-notes-turn-on"), i18n.T("Turn on")).Label(i18n.T("Turn on the dictionary")).Clicked() {
						actions.setDictionaryEnabled(true)
					}
				}, nil)
			}
			ui.Row(c.Key("dict-notes-on-row")).Padding(px(t, sp3), px(t, sp4)).Radius(px(t, radiusCard)).Background(tok(t).Subtle).
				Gap(px(t, sp4)).AlignItems(ui.Center).Children(func() {
				ui.Text(c, i18n.T("Add to the language model prompt")).Grow(1).MinWidth(0)
				on := !state.NotesOff
				if ui.Switch(c.Key("dict-notes-on"), &on).Label(i18n.T("Add to the language model prompt")).Disabled(off).Changed() {
					actions.setDictionaryNotesOff(!on)
				}
			})
			input := ui.TextArea(c.Key("dict-notes"), &local.notes.text).Label(i18n.T("Personal dictionary in Markdown")).Placeholder(notesPlaceholder()).Lines(10, math.MaxInt32)
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
