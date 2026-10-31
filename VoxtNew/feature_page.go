package main

// The feature pages: Transcribe (/transcribe), Translate
// (/translate) and Rewrite (/rewrite) share one structure: the feature's
// shortcuts (and, on Transcribe, the shared "paste the last result"), the
// models it uses, its own settings (Transcribe's clean-up layers;
// Translate's target language, clean-up and card; Rewrite's instruction
// clean-up and when an answer goes to the card), the user's rules with a
// preview of the whole prompt, and a Try area that runs a session whose
// result stays on the page.
//
// Every change goes through a.updateSettings, which also sends the
// shortcuts to the helper again. A page keeps only what its controls
// edit (the selects' labels, a custom shortcut being typed, the rules
// before they are saved), set again from the settings when they change
// elsewhere; each feature has its own (featureUI).

import (
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// transcribeState is the feature pages' view state, one featureUI per
// feature (hotkey.Transcribe, Translate, Rewrite).
type transcribeState struct {
	features map[string]*featureUI
}

// featureUI is a feature page's view state.
type featureUI struct {
	// rows are the shortcuts' controls, paste paste-last's (Transcribe);
	// seen is what they were last set from.
	rows  []keyRow
	paste keyRow
	seen  string
	// asr and llm are the models' selects; svcSeen what they were
	// last set from.
	asr, llm string
	svcSeen  string
	// lang is Translate's target select.
	lang string
	// prompt is the rules as typed; promptSeen the saved value it was
	// last set from; promptDirty an edit not saved yet, promptSeq which
	// edit (its timer saves only the last one).
	prompt      string
	promptSeen  string
	promptDirty bool
	promptSeq   int
	preview     bool
	// block says why the Try area's Start did not start.
	block *sessionNotice
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

// transcribePage builds the Transcribe page.
func (a *app) transcribePage(c *ui.Context) { a.featurePage(c, hotkey.Transcribe) }

// translatePage builds the Translate page.
func (a *app) translatePage(c *ui.Context) { a.featurePage(c, hotkey.Translate) }

// rewritePage builds the Rewrite page.
func (a *app) rewritePage(c *ui.Context) { a.featurePage(c, hotkey.Rewrite) }

// featurePage builds a feature's page.
func (a *app) featurePage(c *ui.Context, feature string) {
	t := c.Theme()
	p := uitext.FeaturePrefix(feature)
	pageHeader(c, uitext.FeatureTitle(feature), nil)
	if a.sess.Try && a.sess.State == session.Recording {
		c.After(250 * time.Millisecond) // the Try area's clock
	}
	ui.Scroll(c.Key(p + "-scroll")).Grow(1).MinHeight(0).Children(func() {
		ui.Column(c).Padding(0, px(t, sp6), px(t, sp6)).Gap(px(t, sp3)).MaxWidth(px(t, 880)).Children(func() {
			switch feature {
			case hotkey.Translate:
				caption(c, i18n.T("Select text and press the shortcut to translate it in place; with nothing selected, speak, and the translation is typed where your cursor is."))
			case hotkey.Rewrite:
				caption(c, i18n.T("Select text and say how to change it (\"shorter\", \"more formal\"); with nothing selected, say what to write. The answer goes where your cursor is."))
			default:
				caption(c, i18n.T("Press a shortcut anywhere, speak, and the text is typed where your cursor is."))
			}
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
			a.featureRules(c, feature)
			a.featureTry(c, feature)
		})
	})
}
