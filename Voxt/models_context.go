package main

// The Speech models page's Speech context (Dictionary.SpeechContext): free
// text the speech model gets with every recording, before the
// dictionary's terms. It is saved as Your rules are (feature_prompts.go):
// promptSaveDelay after the last key, or when the field loses the focus.
// While no feature's speech model takes it, the field is not offered.

import (
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
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
	if a.speechContextUnused() {
		caption(c, i18n.T("The speech models in use do not take a speech context. Your dictionary still guides the language model's clean-up."))
		return
	}
	caption(c, i18n.T("Tell the speech model who is speaking and about what."))
	in := ui.TextArea(c.Key("models-speech-context"), &sc.text).Label(i18n.T("Speech context")).
		Placeholder(i18n.T("For example: I work on LLM serving and speak Chinese with English technical terms.")).MinHeight(px(t, 72))
	a.autosave(in, sc, a.saveSpeechContext)
}

// autosave runs save promptSaveDelay after the last change to field in,
// whose typing st tracks, or when it loses the focus.
func (a *app) autosave(in ui.Element, st *speechContextState, save func()) {
	if in.Changed() {
		st.dirty = true
		st.seq++
		if seq := st.seq; a.win != nil {
			time.AfterFunc(promptSaveDelay, func() {
				a.post(func() {
					if st.dirty && st.seq == seq {
						save()
					}
				})
			})
		}
	} else if st.dirty && !in.Focused() {
		save() // it lost the focus
	}
}

// speechContextUnused says that Transcribe, Translate and Rewrite all use
// a local speech model that takes no context (MOSS-Transcribe-Diarize,
// Whisper…), so a speech context would reach none of them (#4067).
func (a *app) speechContextUnused() bool {
	for _, f := range []string{hotkey.Transcribe, hotkey.Translate, hotkey.Rewrite} {
		m, ok := a.featureModel(local.ASR, f)
		if !ok {
			return false
		}
		spec, curated := omni.Lookup(m.ID)
		if !curated || spec.ASR == nil || spec.ASR.ContextTerms > 0 {
			return false
		}
	}
	return true
}
