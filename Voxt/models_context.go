package main

// The Speech models page's Speech context editor. The controller owns
// settings saves and the typing timer.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func speechContextSection(c *ui.Context, saved string, actions modelsActions, sc *speechContextState) {
	t := c.Theme()
	if !sc.dirty && saved != sc.seen {
		sc.text, sc.seen = saved, saved
	}
	sectionTitle(c, i18n.T("Speech context"))
	in := ui.TextArea(c.Key("models-speech-context"), &sc.text).Label(i18n.T("Speech context")).
		Placeholder(i18n.T("For example: I work on LLM serving and speak Chinese with English technical terms.")).MinHeight(px(t, 72))
	actions.autosaveSpeechContext(in.Changed(), in.Focused())
}
