package main

import (
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type featureOptionViewState struct {
	Label  string `json:"label"`
	Value  string `json:"value"`
	Custom bool   `json:"custom"`
}

type featureSelectViewState struct {
	Value    string                   `json:"value"`
	Options  []featureOptionViewState `json:"options"`
	Disabled bool                     `json:"disabled"`
}

type featureModelNoticeViewState struct {
	Key         string `json:"key"`
	Text        string `json:"text"`
	Warning     bool   `json:"warning"`
	Action      string `json:"action"`
	ActionLabel string `json:"action_label"`
}

type featureCleanupViewState struct {
	Enhance   bool `json:"enhance"`
	Cleanup   bool `json:"cleanup"`
	Structure bool `json:"structure"`
}

type featureModelsViewState struct {
	Prefix   string                        `json:"prefix"`
	Before   []featureModelNoticeViewState `json:"before"`
	After    []featureModelNoticeViewState `json:"after"`
	ASRLabel string                        `json:"asr_label"`
	LLMLabel string                        `json:"llm_label"`
	ASR      featureSelectViewState        `json:"asr"`
	LLM      featureSelectViewState        `json:"llm"`
	Fallback *featureSelectViewState       `json:"fallback,omitempty"`
	Cleanup  featureCleanupViewState       `json:"cleanup"`
}

type featureModelsStateInput struct {
	Feature  string
	Settings settings.Settings
	ReadyASR []local.Model
	ShownASR []local.Model
	ReadyLLM []local.Model
	Local    *featureUI
}

func buildFeatureModelsViewState(translator *i18n.Translator, input featureModelsStateInput) featureModelsViewState {
	feature := input.Feature
	f := featureOf(input.Settings, feature)
	enhance := feature != hotkey.Transcribe || input.Settings.Features.Transcribe.Enhance
	optionsA, automaticA := modelOptionsFor(translator, input.ReadyASR, input.ShownASR)
	optionsL, automaticL := modelOptionsFor(translator, input.ReadyLLM, input.ReadyLLM)
	state := featureModelsViewState{
		Prefix: uitext.FeaturePrefix(feature),
		ASR:    featureModelSelect(optionsA, input.ShownASR, modelLabel(input.ReadyASR, f.ASR, automaticA), len(input.ReadyASR) == 0),
		LLM:    featureModelSelect(optionsL, input.ReadyLLM, modelLabel(input.ReadyLLM, f.LLM, automaticL), len(input.ReadyLLM) == 0 || !enhance),
		Cleanup: featureCleanupViewState{
			Enhance: input.Settings.Features.Transcribe.Enhance, Cleanup: input.Settings.Features.Transcribe.Cleanup,
			Structure: input.Settings.Features.Transcribe.Structure,
		},
	}
	if input.Local != nil {
		state.ASR.Value = selectValue(input.Local.asr, state.ASR)
		state.LLM.Value = selectValue(input.Local.llm, state.LLM)
	}
	openModels := featureModelNoticeViewState{Key: uitext.FeaturePrefix(feature) + "-no-asr", Action: "models", ActionLabel: translator.T("Open Speech models")}
	if len(input.ReadyASR) == 0 {
		openModels.Warning = true
		switch feature {
		case hotkey.Translate:
			openModels.Text = translator.T("No speech model is installed, so only selected text can be translated.")
			openModels.Warning = false
		case hotkey.Rewrite:
			openModels.Text = translator.T("No speech model is installed. Install one to rewrite.")
		default:
			openModels.Text = translator.T("No speech model is installed. Install one to transcribe.")
		}
		state.Before = append(state.Before, openModels)
	}
	setUpLLM := featureModelNoticeViewState{Key: uitext.FeaturePrefix(feature) + "-no-llm", Action: "language", ActionLabel: translator.T("Set up a language model"), Warning: true}
	if len(input.ReadyLLM) == 0 {
		switch feature {
		case hotkey.Translate:
			setUpLLM.Text = translator.T("No language model is set up. Set one up to translate.")
			state.Before = append(state.Before, setUpLLM)
		case hotkey.Rewrite:
			setUpLLM.Text = translator.T("No language model is set up. Set one up to rewrite.")
			state.Before = append(state.Before, setUpLLM)
		}
	}
	switch feature {
	case hotkey.Translate:
		state.ASRLabel, state.LLMLabel = translator.T("Speech model for Translate"), translator.T("Language model for Translate")
		fallbackOptions := append([]string{translator.T("None")}, optionsL[1:]...)
		fallback := featureModelSelect(fallbackOptions, input.ReadyLLM, modelLabel(input.ReadyLLM, input.Settings.Features.Translate.Fallback, fallbackOptions[0]), len(input.ReadyLLM) < 2)
		if input.Local != nil {
			fallback.Value = selectValue(input.Local.fallback, fallback)
		}
		state.Fallback = &fallback
	case hotkey.Rewrite:
		state.ASRLabel, state.LLMLabel = translator.T("Speech model for Rewrite"), translator.T("Language model for Rewrite")
	default:
		state.ASRLabel, state.LLMLabel = translator.T("Speech model for Transcribe"), translator.T("Language model for Transcribe")
	}
	if feature == hotkey.Transcribe && len(input.ReadyLLM) == 0 && enhance {
		setUpLLM.Key = "tr-no-llm"
		setUpLLM.Warning = false
		setUpLLM.Text = translator.T("No language model is set up, so the text is typed as recognised, without clean-up.")
		state.After = append(state.After, setUpLLM)
	}
	return state
}

func featureModelSelect(labels []string, models []local.Model, value string, disabled bool) featureSelectViewState {
	state := featureSelectViewState{Value: value, Disabled: disabled}
	for index, label := range labels {
		option := featureOptionViewState{Label: label}
		if index > 0 && index-1 < len(models) {
			option.Value = models[index-1].ID
		}
		state.Options = append(state.Options, option)
	}
	return state
}

func selectValue(value string, state featureSelectViewState) string {
	if slices.ContainsFunc(state.Options, func(option featureOptionViewState) bool { return option.Label == value }) {
		return value
	}
	return state.Value
}

var endWordPresets = []string{"over", "end", "完毕", "好了"}

type featureEndWordViewState struct {
	Value      string                   `json:"value"`
	Current    string                   `json:"current"`
	Custom     bool                     `json:"custom"`
	CustomText string                   `json:"custom_text"`
	Options    []featureOptionViewState `json:"options"`
}

type featureEndWordStateInput struct {
	Current   string
	Streaming bool
	Local     *keyRow
}

func buildFeatureEndWordViewState(translator *i18n.Translator, input featureEndWordStateInput) *featureEndWordViewState {
	if !input.Streaming {
		return nil
	}
	off, custom := translator.T("Off"), translator.T("Custom…")
	state := &featureEndWordViewState{Current: input.Current, Value: off}
	for _, word := range slices.Concat([]string{""}, endWordPresets) {
		label := word
		if word == "" {
			label = off
		}
		state.Options = append(state.Options, featureOptionViewState{Label: label, Value: word})
	}
	state.Options = append(state.Options, featureOptionViewState{Label: custom, Custom: true})
	switch {
	case input.Local != nil && input.Local.custom:
		state.Custom, state.Value, state.CustomText = true, custom, input.Local.text
	case input.Current == "":
	case slices.Contains(endWordPresets, input.Current):
		state.Value = input.Current
	default:
		state.Custom, state.Value, state.CustomText = true, custom, input.Current
	}
	return state
}

func normalizeFeatureEndWord(word string) string {
	return strings.Join(strings.Fields(word), " ")
}
