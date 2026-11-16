package main

// Input-keyed caches for remote view state. Keys contain only the values
// that can change a rendered frame; cached slices and maps are cloned when
// stored so later app updates cannot mutate a key retroactively.

import (
	"maps"
	"reflect"
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type remoteFeatureCacheKey struct {
	transcribeASR, transcribeLLM, transcribePrompt string
	transcribeEnhance, transcribeCleanup           bool
	transcribeStructure                            bool
	translateASR, translateLLM, translatePrompt    string
	translateCleanup                               bool
	translateTarget                                string
	rewriteASR, rewriteLLM, rewritePrompt          string
	rewriteCleanup                                 bool
}

type remoteInputsCacheKey struct {
	locale         string
	mainLanguage   string
	otherLanguages []string
	providers      settings.Providers
	features       remoteFeatureCacheKey
	groups         []appctx.Group
	installed      []local.Model
	keyMasks       map[string]string
	keySpaced      map[string]bool
	tests          map[string]providerTest
}

type remoteViewCacheKey struct {
	remoteInputsCacheKey
	wait, awayTo   string
	onboardingOpen bool
	onboardingStep int
}

type remoteCardsCacheKey struct {
	remoteInputsCacheKey
	kind local.Kind
}

type llmDebugCacheKey struct {
	remoteInputsCacheKey
	ui llmDebugState
}

type remoteViewCacheEntry struct {
	key   remoteViewCacheKey
	state remoteViewState
	valid bool
}

type remoteCardsCacheEntry struct {
	key   remoteCardsCacheKey
	state remoteCardsViewState
	valid bool
}

type llmDebugCacheEntry struct {
	key   llmDebugCacheKey
	state llmDebugViewState
	valid bool
}

func remoteFeatureInputs(prefs settings.Settings) remoteFeatureCacheKey {
	features := prefs.Features
	return remoteFeatureCacheKey{
		transcribeASR: features.Transcribe.ASR, transcribeLLM: features.Transcribe.LLM,
		transcribePrompt: features.Transcribe.Prompt, transcribeEnhance: features.Transcribe.Enhance,
		transcribeCleanup: features.Transcribe.Cleanup, transcribeStructure: features.Transcribe.Structure,
		translateASR: features.Translate.ASR, translateLLM: features.Translate.LLM, translatePrompt: features.Translate.Prompt,
		translateCleanup: features.Translate.Cleanup, translateTarget: features.Translate.Target,
		rewriteASR: features.Rewrite.ASR, rewriteLLM: features.Rewrite.LLM, rewritePrompt: features.Rewrite.Prompt,
		rewriteCleanup: features.Rewrite.Cleanup,
	}
}

func (a *app) remoteInputsCacheKey(kind local.Kind) remoteInputsCacheKey {
	key := remoteInputsCacheKey{
		locale: i18n.Locale(), mainLanguage: a.prefs.General.MainLanguage, otherLanguages: a.prefs.General.OtherLanguages,
		providers: a.prefs.Providers, features: remoteFeatureInputs(a.prefs), groups: a.aeRules.Groups,
		keyMasks: a.keyMasks, keySpaced: a.keySpaced, tests: a.provTests,
	}
	if kind != "" {
		key.installed = a.installed(kind)
	}
	return key
}

func (a *app) remotePageCacheKey() remoteViewCacheKey {
	return remoteViewCacheKey{
		remoteInputsCacheKey: a.remoteInputsCacheKey(local.LLM), wait: a.prefs.Models.LLMWait,
		awayTo: a.ob.awayTo, onboardingOpen: a.ob.open, onboardingStep: a.ob.step,
	}
}

func (a *app) remoteCardsCacheKey(kind local.Kind) remoteCardsCacheKey {
	return remoteCardsCacheKey{remoteInputsCacheKey: a.remoteInputsCacheKey(kind), kind: kind}
}

func (a *app) llmDebugCacheKey() llmDebugCacheKey {
	return llmDebugCacheKey{remoteInputsCacheKey: a.remoteInputsCacheKey(""), ui: a.llmDbg}
}

func cloneLocalModels(models []local.Model) []local.Model {
	cloned := slices.Clone(models)
	for index := range cloned {
		cloned[index].Features = slices.Clone(cloned[index].Features)
	}
	return cloned
}

func cloneRemoteInputsCacheKey(key remoteInputsCacheKey) remoteInputsCacheKey {
	key.otherLanguages = slices.Clone(key.otherLanguages)
	key.providers.LLM = slices.Clone(key.providers.LLM)
	key.providers.ASR = slices.Clone(key.providers.ASR)
	key.groups = slices.Clone(key.groups)
	key.installed = cloneLocalModels(key.installed)
	key.keyMasks = maps.Clone(key.keyMasks)
	key.keySpaced = maps.Clone(key.keySpaced)
	key.tests = maps.Clone(key.tests)
	for account, test := range key.tests {
		test.res.Models = slices.Clone(test.res.Models)
		key.tests[account] = test
	}
	return key
}

func (a *app) remotePageModel() remoteViewState {
	a.prepareRemoteDialog()
	key := a.remotePageCacheKey()
	cache := &a.pageCache
	if cache.valid && !a.provDlg.open && reflect.DeepEqual(key, cache.key) {
		return cache.state
	}
	input := remotePageStateInput{
		Wait:  remoteWaitStateInput{Value: a.prefs.Models.LLMWait, Locale: i18n.Locale(), CurrentMinutes: a.prefs.Models.LLMWaitFor().Minutes()},
		Cards: a.remoteCardsStateInput(local.LLM), Dialog: a.remoteDialogStateInput(),
	}
	if a.ob.awayTo == remotePath && !a.ob.open {
		input.BackBarText = i18n.T("The setup guide is paused at the %s step.", obStepTitles()[a.ob.step])
	}
	state := buildRemotePageViewState(i18n.CurrentTranslator(), input)
	if !a.provDlg.open {
		key.remoteInputsCacheKey = cloneRemoteInputsCacheKey(key.remoteInputsCacheKey)
		cache.key, cache.state, cache.valid = key, state, true
	}
	return state
}

func (a *app) remoteCardsModel(kind local.Kind) remoteCardsViewState {
	key := a.remoteCardsCacheKey(kind)
	if a.cardsCache == nil {
		a.cardsCache = map[local.Kind]remoteCardsCacheEntry{}
	}
	cache, ok := a.cardsCache[kind]
	if ok && cache.valid && reflect.DeepEqual(key, cache.key) {
		return cache.state
	}
	cache.state = buildRemoteCardsViewState(i18n.CurrentTranslator(), a.remoteCardsStateInput(kind))
	key.remoteInputsCacheKey = cloneRemoteInputsCacheKey(key.remoteInputsCacheKey)
	cache.key, cache.valid = key, true
	a.cardsCache[kind] = cache
	return cache.state
}

func (a *app) llmDebugModel() llmDebugViewState {
	key := a.llmDebugCacheKey()
	cache := &a.debugCache
	if cache.valid && reflect.DeepEqual(key, cache.key) {
		return cache.state
	}
	a.prepareLLMDebug()
	key = a.llmDebugCacheKey()
	setups := a.llmDebugSetups()
	input := llmDebugStateInput{Presets: a.llmDebugPresets(), SavedPrompts: map[string]string{}, UI: &a.llmDbg}
	for _, setup := range setups {
		input.ModelNames = append(input.ModelNames, setupName(local.LLM, setup))
	}
	if len(setups) > 0 {
		for _, preset := range input.Presets {
			input.SavedPrompts[llmDebugPresetKey(preset)] = a.savedPrompt(preset)
			if preset.title == a.llmDbg.preset {
				input.Selected = preset
				input.SavedPrompt = a.savedPrompt(preset)
			}
		}
	}
	cache.state = buildLLMDebugViewState(i18n.CurrentTranslator(), input)
	key.remoteInputsCacheKey = cloneRemoteInputsCacheKey(key.remoteInputsCacheKey)
	cache.key, cache.valid = key, true
	return cache.state
}
