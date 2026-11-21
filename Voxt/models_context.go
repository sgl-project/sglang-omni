package main

// The Speech models page's Speech context editor. The controller owns
// settings saves and the typing timer. While no feature's speech model
// takes the context, a caption says so.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func speechContextSection(c *ui.Context, unused bool, saved string, actions modelsActions, sc *speechContextState) {
	t := c.Theme()
	if !sc.dirty && saved != sc.seen {
		sc.text, sc.seen = saved, saved
	}
	sectionTitle(c, i18n.T("Speech context"))
	caption(c, i18n.T("Tell the speech model who is speaking and about what."))
	if unused { // still offered: meeting transcript correction reads it (meeting_link.go)
		caption(c, i18n.T("The speech models in use do not take it while they recognise. The language model still gets it when it corrects a meeting's transcript, and your dictionary still guides its clean-up."))
	}
	in := ui.TextArea(c.Key("models-speech-context"), &sc.text).Label(i18n.T("Speech context")).
		Placeholder(i18n.T("For example: I work on LLM serving and speak Chinese with English technical terms.")).MinHeight(px(t, 72))
	actions.autosaveSpeechContext(in.Changed(), in.Focused())
}
