package main

// The Dictionary page's Personal dictionary view (Dictionary.Notes):
// Markdown the user writes once, which Transcribe, Translate and Rewrite
// send in the language model's system prompt, after the task's rules
// (compose.Input.Dictionary); speech recognition never gets it. It is
// saved as the Speech context is (autosave in models_context.go). Each
// feature page's preview shows it in that feature's whole prompt
// (feature_prompts.go).

import (
	"math"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// The Dictionary page's views (dictView.tab).
const (
	dictTermsTab = iota
	dictNotesTab
)

// notesPlaceholder is the editor's example: the triples of standard
// spelling, correction mapping and condition (issue #3815).
func notesPlaceholder() string {
	return i18n.T("| Standard spelling | Correction mapping | Condition |\n|---|---|---|\n| 101; I-80 | `one oh one` → 101 | Bay Area highways |\n| Roth | `wrath` → Roth | retirement accounts |\n| NVDA | — | tickers, all caps |")
}

// saveNotes saves the editor as typed.
func (a *app) saveNotes() {
	n := &a.dv.notes
	n.dirty = false
	n.seen = n.text
	if v := n.text; v != a.prefs.Dictionary.Notes {
		a.updateSettings(func(s *settings.Settings) { s.Dictionary.Notes = v })
	}
}

func (a *app) notesView(c *ui.Context) {
	t := c.Theme()
	n := &a.dv.notes
	if saved := a.prefs.Dictionary.Notes; !n.dirty && saved != n.seen {
		n.text, n.seen = saved, saved
	}
	ui.Scroll(c.Key("dict-notes-scroll")).Grow(1).MinHeight(0).Children(func() {
		ui.Column(c).Padding(px(t, sp2), px(t, sp6), px(t, sp6)).Gap(px(t, sp2)).Children(func() {
			caption(c, i18n.T("Write your names, terms and numbers once, in Markdown: for example a table of the standard spelling, what speech recognition writes instead, and when it applies. Transcribe, Translate and Rewrite add it to the language model's prompt. Speech recognition does not get it."))
			off := !a.prefs.Dictionary.Enabled
			formRow(c, "dict-notes-on-row", i18n.T("Add to the language model prompt"), "", func() {
				on := !a.prefs.Dictionary.NotesOff
				if ui.Switch(c.Key("dict-notes-on"), &on).Label(i18n.T("Add to the language model prompt")).Disabled(off).Changed() {
					v := !on
					a.updateSettings(func(s *settings.Settings) { s.Dictionary.NotesOff = v })
				}
			})
			if off {
				notice(c, "dict-notes-off", noticeInfo, i18n.T("The dictionary is off, so the personal dictionary is not sent."), "", nil, nil)
			}
			d := &a.dv
			matches := findFolded(n.text, d.notesQuery)
			moved := d.notesFind.advance(len(matches))
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				d.notesFind.field(c, "dict-notes-find", &d.notesQuery, i18n.T("Find in the personal dictionary"), px(t, 320)).Label(i18n.T("Search personal dictionary"))
				d.notesFind.count(c, d.notesQuery, len(matches))
			})
			// As high as its text, however long (llm_debug.go): one scroll bar, the page's.
			in := ui.TextArea(c.Key("dict-notes"), &n.text).Label(i18n.T("Personal dictionary in Markdown")).Placeholder(notesPlaceholder()).Lines(8, math.MaxInt32)
			if len(matches) > 0 {
				ranges := make([]ui.TextRange, len(matches))
				for i, m := range matches {
					// As Find in the prompt (feature_prompts.go): the current match heavier, and selected.
					ranges[i] = ui.TextRange{Start: m[0], End: m[1], Color: tok(t).Danger, Weight: 700}
					if i == d.notesFind.cur-1 {
						ranges[i].Weight = 900
					}
				}
				in.TextRanges(ranges...)
			}
			if moved {
				in.SetTextSelection(matches[d.notesFind.cur-1][0], matches[d.notesFind.cur-1][1])
			}
			a.autosave(in, n, a.saveNotes)
			// Its place in each prompt shows on the feature pages (#4032).
			caption(c, i18n.T("Each feature's page shows its own whole request, with the variables filled in: open Your rules, then What the language model receives."))
		})
	})
}
