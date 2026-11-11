package main

// The feature pages: Transcribe (/transcribe), Translate
// (/translate) and Rewrite (/rewrite) share one structure: the feature's
// shortcuts (and, on Transcribe, the shared "paste the last result"), the
// models it uses, its own settings (Transcribe's clean-up layers;
// Translate's target language and clean-up; Rewrite's instruction
// clean-up), the answer card (one switch for the three, a size each), the
// user's rules with a preview of the whole prompt, and a Try area that
// runs a session whose result stays on the page.
//
// Every change goes through a.updateSettings, which also sends the
// shortcuts to the helper again. A page keeps only what its controls
// edit (the selects' labels, a custom shortcut being typed, the rules
// before they are saved), set again from the settings when they change
// elsewhere; each feature has its own (featureUI).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// transcribeState is the feature pages' view state, one featureUI per
// feature (hotkey.Transcribe, Translate, Rewrite).
type transcribeState struct {
	features map[string]*featureUI
}

// featureUI is a feature page's view state.
type featureUI struct {
	// rows are the shortcuts' controls, paste paste-last's and cancel
	// cancel-transcription's (Transcribe); seen is what they were last
	// set from.
	rows   []keyRow
	paste  keyRow
	cancel keyRow
	seen   string
	// asr and llm are the models' selects; svcSeen what they were
	// last set from.
	asr, llm string
	svcSeen  string
	// fallback is Translate's fallback model select.
	fallback string
	// lang is Translate's target select.
	lang string
	// prompt is the rules as typed; promptSeen the saved value it was
	// last set from; promptDirty an edit not saved yet, promptSeq which
	// edit (its timer saves only the last one).
	prompt      string
	promptSeen  string
	promptDirty bool
	promptSeq   int
	// rules and preview: the Your rules section and its full prompt, open.
	rules   bool
	preview bool
	// find is the text to find in the rules (⌘F); found the match that
	// Return selected last, from 1 (0: none yet); findFocus and
	// findSelect ask, for the next build, for the find field's focus and
	// for the rules field to select match found.
	find       string
	found      int
	findFocus  bool
	findSelect bool
	// block says why the Try area's Start did not start.
	block *sessionNotice
	// endWord is the end word's select and typed word (Transcribe).
	endWord keyRow
}

// fui is feature's page state.
func (a *app) fui(feature string) *featureUI {
	if a.features == nil {
		a.features = map[string]*featureUI{}
	}
	u := a.features[feature]
	if u == nil {
		u = &featureUI{}
		a.features[feature] = u
	}
	return u
}

// featureKind is a feature's session kind.
func featureKind(feature string) store.Kind {
	switch feature {
	case hotkey.Translate:
		return store.KindTranslate
	case hotkey.Rewrite:
		return store.KindRewrite
	}
	return store.KindTranscribe
}

// ---- The page ----

// featureSettings is a feature's section of the settings dialog: how it
// starts and its models, then what it does with the text, its card and
// its rules.
func (a *app) featureSettings(c *ui.Context, feature string) {
	a.featureShortcuts(c, feature)
	a.featureModels(c, feature)
	switch feature {
	case hotkey.Translate:
		a.translateSettings(c)
	case hotkey.Rewrite:
		a.rewriteSettings(c)
	default:
		a.trEnhance(c)
	}
	a.cardSetting(c, uitext.FeaturePrefix(feature), feature)
	a.featureRules(c, feature)
}
