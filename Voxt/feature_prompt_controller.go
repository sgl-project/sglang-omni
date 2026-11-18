package main

import (
	"context"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

const targetVar = "{{TARGET_LANGUAGE}}"
const promptSaveDelay = 800 * time.Millisecond

func (a *app) featurePromptModel(feature string) featurePromptViewState {
	a.promptSync(feature)
	return buildFeaturePromptViewState(i18n.CurrentTranslator(), featurePromptStateInput{Feature: feature, Settings: a.prefs, SettingsOpen: a.settingsOpen, KeysListening: a.keysListening, Local: a.fui(feature), Glossary: a.glossaryTerms})
}

// loadGlossaryTerms reads the whole dictionary for the feature pages'
// prompt preview (a session reads the terms itself); only the last read
// lands.
func (a *app) loadGlossaryTerms() {
	a.glossaryReads++
	read := a.glossaryReads
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		terms, err := repo.List(ctx, "")
		a.post(func() {
			if read != a.glossaryReads || err != nil {
				return
			}
			a.glossaryTerms = terms
		})
	})
}

// composedPreview is the request the preview shows: what a session would
// send with the settings and the rules as typed (composedPreviewFor).
func (a *app) composedPreview(feature string) (compose.Output, error) {
	return composedPreviewFor(a.prefs, feature, a.fui(feature).prompt, a.glossaryTerms, i18n.CurrentTranslator())
}

// previewInput is the preview's input before the dictionary is added.
func (a *app) previewInput(feature string) compose.Input {
	return previewInputFor(a.prefs, feature, a.fui(feature).prompt, i18n.CurrentTranslator())
}

func trySample() string {
	return trySampleFor(i18n.CurrentTranslator())
}

func rewriteSample() (instruction, selection string) {
	return rewriteSampleFor(i18n.CurrentTranslator())
}

func variableInfo(name string) string {
	return variableInfoFor(i18n.CurrentTranslator(), name)
}

func featureVariables(feature string) []string {
	if feature == hotkey.Translate {
		return compose.VariableNames
	}
	return slices.DeleteFunc(slices.Clone(compose.VariableNames), func(name string) bool { return name == "TARGET_LANGUAGE" })
}

func promptOf(settingsValue settings.Settings, feature string) string {
	return featureOf(settingsValue, feature).Prompt
}

func (a *app) promptBuiltin(feature string) string {
	return promptBuiltinFor(a.prefs, feature, i18n.CurrentTranslator())
}

func (a *app) promptSync(feature string) {
	local := a.fui(feature)
	saved := promptOf(a.prefs, feature)
	builtin := a.promptBuiltin(feature)
	if !local.promptDirty {
		local.promptSeen = saved
		if saved == "" {
			local.prompt = builtin
		} else {
			local.prompt = saved
		}
	}
}

func (a *app) savePrompt(feature string) {
	local := a.fui(feature)
	local.promptDirty = false
	value := local.prompt
	builtin := a.promptBuiltin(feature)
	if strings.TrimSpace(value) == "" || strings.TrimSpace(value) == strings.TrimSpace(builtin) {
		value = ""
	}
	local.promptSeen = value
	if value == promptOf(a.prefs, feature) {
		return
	}
	a.updateSettings(func(settingsValue *settings.Settings) {
		switch feature {
		case hotkey.Translate:
			settingsValue.Features.Translate.Prompt = value
		case hotkey.Rewrite:
			settingsValue.Features.Rewrite.Prompt = value
		default:
			settingsValue.Features.Transcribe.Prompt = value
		}
	})
}

func (a *app) setCardSize(feature string, step int) {
	a.updateSettings(func(settingsValue *settings.Settings) {
		if step == 0 {
			delete(settingsValue.Features.CardSizes, feature)
			return
		}
		if settingsValue.Features.CardSizes == nil {
			settingsValue.Features.CardSizes = map[string]int{}
		}
		settingsValue.Features.CardSizes[feature] = step
	})
	a.overlayRestyle()
}

func (a *app) selectFeatureTarget(code string) {
	a.updateSettings(func(settingsValue *settings.Settings) {
		settingsValue.Features.Translate.Target = code
	})
}

func (a *app) setFeatureTranslateCleanup(on bool) {
	a.updateSettings(func(settingsValue *settings.Settings) {
		settingsValue.Features.Translate.Cleanup = on
	})
}

func (a *app) setFeatureRewriteCleanup(on bool) {
	a.updateSettings(func(settingsValue *settings.Settings) {
		settingsValue.Features.Rewrite.Cleanup = on
	})
}

func (a *app) setFeatureShowCard(feature string, on bool) {
	a.updateSettings(func(settingsValue *settings.Settings) {
		switch feature {
		case hotkey.Translate:
			settingsValue.Features.Translate.ShowCard = on
		case hotkey.Rewrite:
			settingsValue.Features.Rewrite.ShowCard = on
		default:
			settingsValue.Features.Transcribe.ShowCard = on
		}
	})
}

func (a *app) addFeatureTargetVariable() {
	local := a.fui(hotkey.Translate)
	local.prompt = strings.TrimRight(local.prompt, "\n") + "\n\n" + i18n.T("Translate the text into {{TARGET_LANGUAGE}}.")
	a.savePrompt(hotkey.Translate)
}

func (a *app) resetFeaturePrompt(feature string) {
	local := a.fui(feature)
	local.prompt = a.promptBuiltin(feature)
	a.savePrompt(feature)
}

func (a *app) scheduleFeaturePromptSave(feature string, sequence int) {
	if a.win == nil {
		return
	}
	local := a.fui(feature)
	time.AfterFunc(promptSaveDelay, func() {
		a.post(func() {
			if local.promptDirty && local.promptSeq == sequence {
				a.savePrompt(feature)
			}
		})
	})
}

func (a *app) copyFeatureText(text string) {
	if a.copyText != nil {
		_ = a.copyText(text)
	} else {
		_ = mygo.Clipboard.Write(transfer.TextData(text))
	}
	a.toast(i18n.T("Copied"))
}
