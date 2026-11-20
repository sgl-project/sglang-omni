package main

import (
	"fmt"
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) featureModelsModel(feature string) featureModelsViewState {
	readyASR := a.installed(local.ASR)
	readyLLM := local.ForFeature(a.installed(local.LLM), feature)
	shownASR := slices.DeleteFunc(slices.Clone(readyASR), func(model local.Model) bool { return a.modelHidden(model.ID) })
	f := featureOf(a.prefs, feature)
	input := featureModelsStateInput{
		Feature: feature, Settings: a.prefs, ReadyASR: readyASR, ShownASR: shownASR, ReadyLLM: readyLLM, Local: a.fui(feature),
		ASRStatus: a.featureModelStatus(local.ASR, feature, f.ASR), LLMStatus: a.featureModelStatus(local.LLM, feature, f.LLM),
		ModelDeciding: a.modelDeciding,
	}
	a.syncFeatureModelControls(input)
	return buildFeatureModelsViewState(i18n.CurrentTranslator(), input)
}

// featureModelStatusKey changes when the models beside feature's pickers
// or the preview's glossary change (featurePageModel's cache).
func (a *app) featureModelStatusKey(feature string) string {
	f := featureOf(a.prefs, feature)
	key := fmt.Sprintf("%p/%d|%s", a.glossaryTerms, len(a.glossaryTerms), a.modelDeciding)
	for _, input := range []featureModelStatusInput{a.featureModelStatus(local.ASR, feature, f.ASR), a.featureModelStatus(local.LLM, feature, f.LLM)} {
		if input.Model != nil {
			key += fmt.Sprintf("|%s %+v", input.Model.ID, input.Status)
		}
		if input.Service != nil {
			key += fmt.Sprintf("|%+v", *input.Service)
		}
	}
	return key
}

// featureModelStatus is the model feature's picker of kind stands for:
// the one chosen (id), else the one Automatic uses, else the recommended
// speech model.
func (a *app) featureModelStatus(kind local.Kind, feature, id string) featureModelStatusInput {
	if id == "" {
		if model, ok := a.featureModel(kind, feature); ok {
			id = model.ID
		}
	}
	if kind == local.ASR {
		if _, ok := models.Lookup(id); !ok {
			_, configured := a.setupOf(kind, id)
			hasInstalledCatalogModel := slices.ContainsFunc(a.installed(local.ASR), func(installedModel local.Model) bool {
				_, ok := models.Lookup(installedModel.ID)
				return ok
			})
			if !configured && !hasInstalledCatalogModel {
				id = models.ASRAccurate
			}
		}
	}
	if model, ok := models.Lookup(id); ok && local.Kind(model.Kind) == kind && !model.Retired {
		return featureModelStatusInput{Model: &model, Status: a.modelStatus[model.ID]}
	}
	setup, configured := a.setupOf(kind, id)
	if !configured {
		return featureModelStatusInput{}
	}
	test, hasTest := a.provTests[providers.Account(kind, id)]
	return featureModelStatusInput{Service: &remoteServiceStateInput{
		Service: lookupService(kind, id), Setup: setup, Configured: true, Missing: a.setupMissing(kind, setup), Test: test, HasTest: hasTest,
	}}
}

func modelOptions(ready, shown []local.Model) (options []string, automatic string) {
	return modelOptionsFor(i18n.CurrentTranslator(), ready, shown)
}

func modelOptionsFor(translator *i18n.Translator, ready, shown []local.Model) (options []string, automatic string) {
	automatic = translator.T("Automatic")
	if len(ready) > 0 {
		automatic = translator.T("Automatic (now: %s)", ready[0].Name)
	}
	options = []string{automatic}
	for _, model := range shown {
		options = append(options, model.Name)
	}
	return options, automatic
}

func modelLabel(ready []local.Model, id, automatic string) string {
	for _, model := range ready {
		if model.ID == id {
			return model.Name
		}
	}
	return automatic
}

func (a *app) syncFeatureModelControls(input featureModelsStateInput) {
	f := featureOf(input.Settings, input.Feature)
	optionsA, automaticA := modelOptions(input.ReadyASR, input.ShownASR)
	optionsL, automaticL := modelOptions(input.ReadyLLM, input.ReadyLLM)
	fallback := input.Settings.Features.Translate.Fallback
	seen := strings.Join([]string{f.ASR, f.LLM, fallback, strings.Join(optionsA, "\x00"), strings.Join(optionsL, "\x00")}, "\x01")
	local := input.Local
	if local.svcSeen == seen {
		return
	}
	local.svcSeen = seen
	fallbackOptions := []string{i18n.T("None")}
	local.asr = modelLabel(input.ReadyASR, f.ASR, automaticA)
	local.llm = modelLabel(input.ReadyLLM, f.LLM, automaticL)
	local.fallback = modelLabel(input.ReadyLLM, fallback, fallbackOptions[0])
}

func (a *app) featureEndWordModel() *featureEndWordViewState {
	model, ok := a.featureModel(local.ASR, hotkey.Transcribe)
	input := featureEndWordStateInput{Current: a.prefs.Features.EndWord, Streaming: ok && model.Stream, Local: &a.fui(hotkey.Transcribe).endWord}
	return buildFeatureEndWordViewState(i18n.CurrentTranslator(), input)
}

func (a *app) openFeatureSpeechModels() {
	a.navigate(modelsPath)
}

func (a *app) openFeatureLanguageModels() {
	a.navigate(remotePath)
}

func (a *app) selectFeatureModel(feature, kind, id string) {
	modelKind := local.ASR
	if kind == "llm" {
		modelKind = local.LLM
	}
	a.updateSettings(func(settings *settings.Settings) {
		setFeatureModel(settings, modelKind, feature, id)
	})
}

func (a *app) selectFeatureFallback(id string) {
	a.updateSettings(func(settings *settings.Settings) {
		settings.Features.Translate.Fallback = id
	})
}

func (a *app) setFeatureEnhance(on bool) {
	a.updateSettings(func(settings *settings.Settings) { settings.Features.Transcribe.Enhance = on })
}

func (a *app) setFeatureCleanup(on bool) {
	a.updateSettings(func(settings *settings.Settings) { settings.Features.Transcribe.Cleanup = on })
}

func (a *app) setFeatureStructure(on bool) {
	a.updateSettings(func(settings *settings.Settings) { settings.Features.Transcribe.Structure = on })
}

func (a *app) setFeatureEndWord(word string) {
	a.setEndWord(normalizeFeatureEndWord(word))
}

func (a *app) setFeatureReturnFinish(on bool) {
	a.updateSettings(func(settings *settings.Settings) { settings.Features.Transcribe.ReturnFinishes = on })
}

func (a *app) setEndWord(word string) {
	a.updateSettings(func(settings *settings.Settings) {
		settings.Features.EndWord = word
	})
}
