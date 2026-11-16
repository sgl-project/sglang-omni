package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type featureViewState struct {
	Feature          string                    `json:"feature"`
	Prefix           string                    `json:"prefix"`
	Title            string                    `json:"title"`
	Caption          string                    `json:"caption"`
	RefreshRecording bool                      `json:"refresh_recording"`
	Shortcuts        featureShortcutsViewState `json:"shortcuts"`
	Models           featureModelsViewState    `json:"models"`
	Prompt           featurePromptViewState    `json:"prompt"`
	Try              tryCardViewState          `json:"try"`
}

type featurePageStateInput struct {
	Feature          string
	RefreshRecording bool
	Shortcuts        featureShortcutsViewState
	Models           featureModelsViewState
	Prompt           featurePromptViewState
	Try              tryCardViewState
}

func buildFeatureViewState(translator *i18n.Translator, input featurePageStateInput) featureViewState {
	state := featureViewState{
		Feature: input.Feature, Prefix: uitext.FeaturePrefix(input.Feature), Title: uitext.FeatureTitleFor(translator, input.Feature),
		RefreshRecording: input.RefreshRecording, Shortcuts: input.Shortcuts, Models: input.Models, Prompt: input.Prompt, Try: input.Try,
	}
	switch input.Feature {
	case hotkey.Translate:
		state.Caption = translator.T("Translates the selected text in place, or what you say when nothing is selected.")
	case hotkey.Rewrite:
		state.Caption = translator.T("Changes the selected text as you say (\"shorter\", \"more formal\"), or writes what you ask for.")
	default:
		state.Caption = translator.T("Press a shortcut anywhere, speak, and the text is typed where your cursor is.")
	}
	return state
}
