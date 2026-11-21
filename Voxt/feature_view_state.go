package main

import (
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
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
	// ASRStatus and LLMStatus are the models the pickers stand for, with
	// their actions beside them (no trip to Settings).
	ASRStatus *featureModelStatusViewState `json:"asr_status,omitempty"`
	LLMStatus *featureModelStatusViewState `json:"llm_status,omitempty"`
}

// featureModelStatusViewState is a catalogue model's size and actions
// (download, cancel, delete) with its progress, or a remote service's
// state with Edit….
type featureModelStatusViewState struct {
	Key      string                   `json:"key"`
	Text     string                   `json:"text"`
	ModelID  string                   `json:"model_id,omitempty"`
	Actions  []modelActionViewState   `json:"actions,omitempty"`
	Progress *modelProgressViewState  `json:"progress,omitempty"`
	Error    *modelErrorViewState     `json:"error,omitempty"`
	Service  *featureServiceViewState `json:"service,omitempty"`
}

type featureServiceViewState struct {
	Kind                   string `json:"kind"`
	ID                     string `json:"id"`
	Status                 string `json:"status"`
	EditLabel              string `json:"edit_label"`
	EditAccessibilityLabel string `json:"edit_accessibility_label"`
}

// featureModelStatusInput is the model a picker stands for: a catalogue
// model (Model) or a set-up remote service (Service); neither shows nothing.
type featureModelStatusInput struct {
	Model   *models.Model
	Status  models.Status
	Service *remoteServiceStateInput
}

type featureModelsStateInput struct {
	Feature  string
	Settings settings.Settings
	ReadyASR []local.Model
	ShownASR []local.Model
	ReadyLLM []local.Model
	Local    *featureUI
	// ASRStatus, LLMStatus: the pickers' models; ModelDeciding is the
	// model whose deletion is being weighed.
	ASRStatus, LLMStatus featureModelStatusInput
	ModelDeciding        string
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
	state.ASRStatus = buildFeatureModelStatus(translator, state.Prefix+"-asr", input.ASRStatus, input.ModelDeciding)
	state.LLMStatus = buildFeatureModelStatus(translator, state.Prefix+"-llm", input.LLMStatus, input.ModelDeciding)
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

func buildFeatureModelStatus(translator *i18n.Translator, key string, input featureModelStatusInput, deciding string) *featureModelStatusViewState {
	if model := input.Model; model != nil {
		status := input.Status
		state := &featureModelStatusViewState{
			Key: key + "-model", ModelID: model.ID,
			Text:     uitext.ModelNameFor(translator, model.ID) + " · " + modelSizeTextFor(translator, *model, status),
			Actions:  buildModelActions(translator, modelsViewStateInput{ModelDeciding: deciding}, *model, status),
			Progress: buildModelProgress(translator, *model, status),
		}
		if status.Err != nil && !status.Downloading {
			message, detail := uitext.DownloadErrorTextFor(translator, status.Err)
			state.Error = &modelErrorViewState{Message: message, Detail: detail, RetryLabel: translator.T("Try again")}
		}
		return state
	}
	if input.Service == nil || !input.Service.Configured {
		return nil
	}
	card := buildRemoteProviderCardState(translator, *input.Service)
	return &featureModelStatusViewState{
		Key: key + "-service", Text: card.Name + " · " + card.StatusText,
		Service: &featureServiceViewState{
			Kind: card.Kind, ID: card.ID, Status: card.Status,
			EditLabel: translator.T("Edit…"), EditAccessibilityLabel: translator.T("Edit %s", card.Name),
		},
	}
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
