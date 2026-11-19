package main

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestMeetingSettingsViewState(t *testing.T) {
	translator := i18n.NewTranslator("en")
	readyASR := []local.Model{{ID: "asr-hidden", Name: "Hidden ASR"}, {ID: "asr-shown", Name: "Shown ASR"}}
	input := settingsMeetingStateInput{
		Meeting: settings.Meeting{
			Feature: settings.Feature{ASR: "asr-hidden", LLM: "llm-second"},
			Mode:    "subtitles", Lang: "ja", HideFromSharing: true,
		},
		ReadyASR: readyASR,
		ShownASR: readyASR[1:],
		ReadyLLM: []local.Model{{ID: "llm-first", Name: "First LLM"}, {ID: "llm-second", Name: "Second LLM"}},
	}
	state := buildSettingsMeetingViewState(translator, input)
	if code := settingsLanguageCode(state.CaptureModeSelected, state.CaptureModeOptions); code != "subtitles" {
		t.Fatalf("capture mode code %q", code)
	}
	if !state.HideFromSharing {
		t.Fatal("screen-sharing privacy was not carried into view state")
	}
	wantASROptions := []settingsLanguageViewState{
		{Label: "Automatic (now: Hidden ASR)"},
		{Code: "asr-shown", Label: "Shown ASR"},
	}
	if !slices.Equal(state.ASROptions, wantASROptions) || state.ASRSelected != "Hidden ASR" {
		t.Fatalf("ASR options %+v, selected %q", state.ASROptions, state.ASRSelected)
	}
	if code := settingsLanguageCode(state.SpokenLanguageSelected, state.SpokenLanguageOptions); code != "ja" {
		t.Fatalf("spoken language code %q", code)
	}
	if state.LLMSelected != "Second LLM" || settingsLanguageCode(state.LLMSelected, state.LLMOptions) != "llm-second" {
		t.Fatalf("LLM options %+v, selected %q", state.LLMOptions, state.LLMSelected)
	}
	input.Meeting.LLM = ""
	if selected := buildSettingsMeetingViewState(translator, input).LLMSelected; selected != "First LLM" {
		t.Fatalf("default LLM label %q", selected)
	}
	if _, err := json.Marshal(state); err != nil {
		t.Fatal(err)
	}

	empty := buildSettingsMeetingViewState(translator, settingsMeetingStateInput{})
	if !empty.ASRDisabled || !empty.NoLLM || empty.ASRSelected != "Automatic" || empty.LLMSelected != "Automatic" {
		t.Fatalf("no-model state %+v", empty)
	}
	if empty.SpokenLanguageSelected != "Same as the main language" {
		t.Fatalf("default spoken language %q", empty.SpokenLanguageSelected)
	}
}
