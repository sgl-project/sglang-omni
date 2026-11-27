package main

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type settingsMeetingViewState struct {
	CaptureModeOptions           []settingsLanguageViewState `json:"capture_mode_options"`
	CaptureModeSelected          string                      `json:"capture_mode_selected"`
	HideFromSharing              bool                        `json:"hide_from_sharing"`
	ASROptions                   []settingsLanguageViewState `json:"asr_options"`
	ASRSelected                  string                      `json:"asr_selected"`
	ASRDisabled                  bool                        `json:"asr_disabled"`
	SpokenLanguageOptions        []settingsLanguageViewState `json:"spoken_language_options"`
	SpokenLanguageSelected       string                      `json:"spoken_language_selected"`
	LLMOptions                   []settingsLanguageViewState `json:"llm_options"`
	LLMSelected                  string                      `json:"llm_selected"`
	NoLLM                        bool                        `json:"no_llm"`
	CorrectionStyle              string                      `json:"correction_style"`
	CorrectionStyleResetDisabled bool                        `json:"correction_style_reset_disabled"`
}

type settingsMeetingStateInput struct {
	Meeting  settings.Meeting
	ReadyASR []local.Model
	ShownASR []local.Model
	// AutomaticASR is the speech model a meeting on Automatic opens: the
	// language gate's first for the meeting's language (meetingTrackASR's
	// rule), so "now:" never names a model the gate would skip (#4067).
	AutomaticASR   local.Model
	ReadyLLM       []local.Model
	CorrectionText string
}

func buildSettingsMeetingViewState(translator *i18n.Translator, input settingsMeetingStateInput) settingsMeetingViewState {
	state := settingsMeetingViewState{
		CaptureModeOptions: []settingsLanguageViewState{
			{Label: translator.T("Full meeting (microphone and system audio)")},
			{Code: "subtitles", Label: translator.T("Subtitles (system audio only)")},
			{Code: "recording", Label: translator.T("Recording (microphone only)")},
		},
		HideFromSharing:              input.Meeting.HideFromSharing,
		ASRDisabled:                  len(input.ReadyASR) == 0,
		NoLLM:                        len(input.ReadyLLM) == 0,
		CorrectionStyle:              input.CorrectionText,
		CorrectionStyleResetDisabled: input.Meeting.CorrectionStyle == "" && strings.TrimSpace(input.CorrectionText) == strings.TrimSpace(meeting.DefaultCorrectionStyle),
	}
	state.CaptureModeSelected = state.CaptureModeOptions[0].Label
	for _, option := range state.CaptureModeOptions {
		if option.Code == input.Meeting.Mode {
			state.CaptureModeSelected = option.Label
		}
	}

	automaticASR := translator.T("Automatic")
	if input.AutomaticASR.ID != "" {
		automaticASR = translator.T("Automatic (now: %s)", input.AutomaticASR.Name)
	}
	state.ASRSelected = automaticASR
	state.ASROptions = append(state.ASROptions, settingsLanguageViewState{Label: automaticASR})
	for _, model := range input.ShownASR {
		state.ASROptions = append(state.ASROptions, settingsLanguageViewState{Code: model.ID, Label: model.Name})
	}
	for _, model := range input.ReadyASR {
		if model.ID == input.Meeting.ASR {
			state.ASRSelected = model.Name
			break
		}
	}

	state.SpokenLanguageOptions = append(state.SpokenLanguageOptions, settingsLanguageViewState{Label: translator.T("Same as the main language")})
	for _, code := range mainLanguages {
		state.SpokenLanguageOptions = append(state.SpokenLanguageOptions, settingsLanguageViewState{Code: code, Label: settingsLanguageLabel(translator, code)})
	}
	state.SpokenLanguageSelected = state.SpokenLanguageOptions[0].Label
	for _, option := range state.SpokenLanguageOptions[1:] {
		if option.Code == input.Meeting.Lang {
			state.SpokenLanguageSelected = option.Label
		}
	}

	state.LLMSelected = translator.T("Automatic")
	state.LLMOptions = append(state.LLMOptions, settingsLanguageViewState{Label: state.LLMSelected})
	for _, model := range input.ReadyLLM {
		state.LLMOptions = append(state.LLMOptions, settingsLanguageViewState{Code: model.ID, Label: model.Name})
	}
	if model, ok := local.Choose(input.ReadyLLM, input.Meeting.LLM); ok {
		state.LLMSelected = model.Name
	}
	return state
}
