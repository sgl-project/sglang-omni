package main

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

var notesFeatures = []string{hotkey.Transcribe, hotkey.Translate, hotkey.Rewrite}

type dictionaryPromptMessageViewState struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type dictionaryNotesExampleViewState struct {
	Error         string                             `json:"error"`
	System        string                             `json:"system"`
	Dictionary    string                             `json:"dictionary"`
	UserInput     string                             `json:"user_input"`
	Speech        string                             `json:"speech"`
	ShowUserInput bool                               `json:"show_user_input"`
	Messages      []dictionaryPromptMessageViewState `json:"messages"`
}

type dictionaryNotesViewState struct {
	Enabled    bool                            `json:"enabled"`
	NotesOff   bool                            `json:"notes_off"`
	Feature    string                          `json:"feature"`
	EnhanceOff bool                            `json:"enhance_off"`
	Example    dictionaryNotesExampleViewState `json:"example"`
}

type dictionaryNotesStateInput struct {
	Settings  settings.Settings
	Terms     []store.Term
	Feature   string
	NotesText string
}

type notesExampleCache struct {
	in      compose.Input
	terms   []store.Term
	built   bool
	example dictionaryNotesExampleViewState
}

func buildDictionaryNotesViewState(translator *i18n.Translator, input dictionaryNotesStateInput) dictionaryNotesViewState {
	return dictionaryNotesStateFromExample(input, buildDictionaryNotesExample(translator, input))
}

func dictionaryNotesStateFromExample(input dictionaryNotesStateInput, example dictionaryNotesExampleViewState) dictionaryNotesViewState {
	return dictionaryNotesViewState{
		Enabled: input.Settings.Dictionary.Enabled, NotesOff: input.Settings.Dictionary.NotesOff,
		Feature: input.Feature, EnhanceOff: input.Feature == hotkey.Transcribe && !input.Settings.Features.Transcribe.Enhance,
		Example: example,
	}
}

func buildDictionaryNotesExample(translator *i18n.Translator, input dictionaryNotesStateInput) dictionaryNotesExampleViewState {
	_, full := dictionaryNotesComposeInputs(translator, input)
	out, err := compose.Build(full)
	without := full
	without.Dictionary = ""
	plain, _ := compose.Build(without)
	state := dictionaryNotesExampleViewState{System: plain.System, Dictionary: out.Dictionary, ShowUserInput: full.Task == compose.Rewrite}
	if err != nil {
		state.Error = err.Error()
		return state
	}
	state.UserInput, state.Speech = "", full.Text
	if full.Task == compose.Rewrite {
		state.UserInput, state.Speech = full.Text, full.Instruction
	}
	for _, message := range out.Messages() {
		state.Messages = append(state.Messages, dictionaryPromptMessageViewState{Role: message.Role, Content: message.Content})
	}
	return state
}

func dictionaryNotesComposeInputs(translator *i18n.Translator, input dictionaryNotesStateInput) (compose.Input, compose.Input) {
	preferences := input.Settings
	preferences.Dictionary.Notes = input.NotesText
	features := preferences.Features
	base := compose.Input{
		MainLanguage: session.MainLanguage(preferences, translator.Locale()), OtherLanguages: slices.Clone(preferences.General.OtherLanguages),
		Dictionary: session.PromptDictionary(preferences),
	}
	switch input.Feature {
	case hotkey.Translate:
		base.Task, base.TaskPrompt, base.Cleanup, base.Text = compose.Translate, features.Translate.Prompt, features.Translate.Cleanup, translator.T("um so the meeting moves to Thursday at three, no, at four")
		base.TargetLanguage = session.TranslateTarget(preferences, translator.Locale(), "")
	case hotkey.Rewrite:
		base.Task, base.TaskPrompt, base.Cleanup = compose.Rewrite, features.Rewrite.Prompt, features.Rewrite.Cleanup
		base.Instruction = translator.T("um make it shorter, no, more formal")
		base.Text = translator.T("hey, the release slips to thursday because the tests are still running, sorry!")
	default:
		base.Task, base.TaskPrompt, base.Cleanup, base.Structure, base.Text = compose.Transcribe, features.Transcribe.Prompt, features.Transcribe.Cleanup, features.Transcribe.Structure, translator.T("um so the meeting moves to Thursday at three, no, at four")
	}
	full := base
	if preferences.Dictionary.Enabled && preferences.Dictionary.Glossary {
		for _, term := range input.Terms {
			if term.Enabled {
				full.Glossary = append(full.Glossary, compose.Term{Term: term.Term, Aliases: term.Aliases})
			}
		}
	}
	return base, full
}
