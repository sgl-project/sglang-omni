package main

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
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
	options := a.featureTryOptions(feature)
	tryState, _ := a.tryCardStateAt(options, now)
	return buildFeatureViewState(i18n.CurrentTranslator(), featurePageStateInput{
		Feature: feature, RefreshRecording: a.sess.Try && a.sess.State == session.Recording,
		Shortcuts: a.featureShortcutsModel(feature), Models: a.featureModelsModel(feature), Prompt: a.featurePromptModel(feature), Try: tryState,
	})
}
