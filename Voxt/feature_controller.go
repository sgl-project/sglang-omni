package main

import (
	"fmt"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type transcribeState struct {
	features        map[string]*featureUI
	featureRevision uint64
	views           map[string]featureViewState
	viewKeys        map[string]featureViewCacheKey
}

type featureViewCacheKey struct {
	revision         uint64
	feature          string
	locale           string
	models           string
	local            string
	usedBy           string
	block            string
	settingsOpen     bool
	hotkeysInstalled bool
	helperRecords    bool
	sessionRevision  uint64
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

func (a *app) featurePageModel(feature string) (featureViewState, tryCardActions) {
	localUI := a.fui(feature)
	usedBy := a.featureUsedBy(feature)
	readyASR := a.installed(local.ASR)
	readyLLM := a.installed(local.LLM)
	key := featureViewCacheKey{
		revision: a.featureRevision, feature: feature, locale: i18n.Locale(),
		models: fmt.Sprintf("%#v|%#v", readyASR, readyLLM), local: fmt.Sprintf("%#v", *localUI),
		usedBy: fmt.Sprintf("%#v", usedBy), block: fmt.Sprintf("%#v", localUI.block),
		settingsOpen: a.settingsOpen, hotkeysInstalled: a.hotkeys.Installed, helperRecords: a.helperRecordsKeys(), sessionRevision: a.sess.Rev,
	}
	recordingTry := a.sess.Try && a.sess.State == session.Recording
	if !recordingTry && a.viewKeys[feature] == key {
		return a.views[feature], &tryController{app: a, options: a.featureTryOptions(feature)}
	}
	options := a.featureTryOptions(feature)
	tryState, _ := a.tryCardState(options)
	state := buildFeatureViewState(i18n.CurrentTranslator(), featurePageStateInput{
		Feature: feature, RefreshRecording: recordingTry,
		Shortcuts: a.featureShortcutsModel(feature), Models: a.featureModelsModel(feature), Prompt: a.featurePromptModel(feature), Try: tryState,
	})
	if a.views == nil {
		a.views = map[string]featureViewState{}
		a.viewKeys = map[string]featureViewCacheKey{}
	}
	a.views[feature], a.viewKeys[feature] = state, key
	return state, &tryController{app: a, options: options}
}
