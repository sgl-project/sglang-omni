package main

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type transcribeState struct {
	features map[string]*featureUI
}

func (a *app) fui(feature string) *featureUI {
	if a.features == nil {
		a.features = map[string]*featureUI{}
	}
	local := a.features[feature]
	if local == nil {
		local = &featureUI{}
		a.features[feature] = local
	}
	return local
}

func featureKind(feature string) store.Kind {
	switch feature {
	case hotkey.Translate:
		return store.KindTranslate
	case hotkey.Rewrite:
		return store.KindRewrite
	}
	return store.KindTranscribe
}

func (a *app) featurePageModel(feature string) featureViewState {
	return a.featurePageModelAt(feature, time.Now())
}

func (a *app) featurePageModelAt(feature string, now time.Time) featureViewState {
	translator := i18n.CurrentTranslator()
	state := featureViewState{
		Feature: feature, Prefix: uitext.FeaturePrefix(feature), Title: uitext.FeatureTitle(feature),
		RefreshRecording: a.sess.Try && a.sess.State == session.Recording,
		Shortcuts:        a.featureShortcutsModel(feature), Models: a.featureModelsModel(feature), Prompt: a.featurePromptModel(feature),
	}
	switch feature {
	case hotkey.Translate:
		state.Caption = translator.T("Translates the selected text in place, or what you say when nothing is selected.")
	case hotkey.Rewrite:
		state.Caption = translator.T("Changes the selected text as you say (\"shorter\", \"more formal\"), or writes what you ask for.")
	default:
		state.Caption = translator.T("Press a shortcut anywhere, speak, and the text is typed where your cursor is.")
	}
	options := a.featureTryOptions(feature)
	state.Try, _ = a.tryCardStateAt(options, now)
	return state
}
