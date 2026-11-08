package main

// The Speech models page's Speech context (Dictionary.SpeechContext): free
// text the speech model gets with every recording, before the
// dictionary's terms. It is saved as Your rules are (feature_prompts.go):
// promptSaveDelay after the last key, or when the field loses the focus.

import (
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// speechContextState is the field as typed, the saved text it last took,
// whether typing waits to be saved and the typing's number (only the last
// key's timer saves).
type speechContextState struct {
	text, seen string
	dirty      bool
	seq        int
}

// saveSpeechContext saves the field as typed.
func (a *app) saveSpeechContext() {
	sc := &a.speechContext
	sc.dirty = false
	v := strings.TrimSpace(sc.text)
	sc.seen = v
	if v != a.prefs.Dictionary.SpeechContext {
		a.updateSettings(func(s *settings.Settings) { s.Dictionary.SpeechContext = v })
	}
}

func (a *app) speechContextSection(c *ui.Context) {
	t := c.Theme()
	sc := &a.speechContext
	if saved := a.prefs.Dictionary.SpeechContext; !sc.dirty && saved != sc.seen {
		sc.text, sc.seen = saved, saved
	}
	sectionTitle(c, i18n.T("Speech context"))
	caption(c, i18n.T("Tell the speech model who speaks and about what: names, fields, words to expect. It goes with every recording, before the dictionary's terms; a speech service that takes no prompt skips it."))
	in := ui.TextArea(c.Key("models-speech-context"), &sc.text).Label(i18n.T("Speech context")).
		Placeholder(i18n.T("For example: I work on LLM serving and speak Chinese with English technical terms.")).MinHeight(px(t, 72))
	if in.Changed() {
		sc.dirty = true
		sc.seq++
		if seq := sc.seq; a.win != nil {
			time.AfterFunc(promptSaveDelay, func() {
				a.post(func() {
					if sc.dirty && sc.seq == seq {
						a.saveSpeechContext()
					}
				})
			})
		}
	} else if sc.dirty && !in.Focused() {
		a.saveSpeechContext() // it lost the focus
	}
}
