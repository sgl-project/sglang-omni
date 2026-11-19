package main

// Serializable display state for the language model debug page.

import (
	"fmt"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type llmDebugViewState struct {
	Ready        bool                      `json:"ready"`
	Models       []string                  `json:"models,omitempty"`
	Presets      []string                  `json:"presets,omitempty"`
	PresetStates []llmDebugPresetViewState `json:"preset_states,omitempty"`
	SavedPrompt  string                    `json:"saved_prompt,omitempty"`
	PromptLabel  string                    `json:"prompt_label,omitempty"`
	InputLabel   string                    `json:"input_label,omitempty"`
	Instruction  bool                      `json:"instruction,omitempty"`
	Edited       bool                      `json:"edited"`
	CanRun       bool                      `json:"can_run"`
	Running      bool                      `json:"running"`
	Result       *llmDebugResultViewState  `json:"result,omitempty"`
	Form         llmDebugFormViewState     `json:"form"`
}

type llmDebugPresetViewState struct {
	Title       string `json:"title"`
	Feature     string `json:"feature"`
	Group       string `json:"group,omitempty"`
	SavedPrompt string `json:"saved_prompt"`
	PromptLabel string `json:"prompt_label"`
	InputLabel  string `json:"input_label"`
	Instruction bool   `json:"instruction"`
}

type llmDebugFormViewState struct {
	Model       string `json:"model,omitempty"`
	Preset      string `json:"preset,omitempty"`
	Filled      string `json:"filled,omitempty"`
	Prompt      string `json:"prompt,omitempty"`
	Input       string `json:"input,omitempty"`
	Instruction string `json:"instruction,omitempty"`
}

type llmDebugResultViewState struct {
	Answer      string                     `json:"answer,omitempty"`
	Error       string                     `json:"error,omitempty"`
	AnswerTitle string                     `json:"answer_title,omitempty"`
	Messages    []llmDebugMessageViewState `json:"messages,omitempty"`
}

type llmDebugMessageViewState struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type llmDebugStateInput struct {
	ModelNames   []string
	Presets      []llmDebugPreset
	Selected     llmDebugPreset
	SavedPrompt  string
	SavedPrompts map[string]string
	UI           *llmDebugState
}

func buildLLMDebugViewState(translator *i18n.Translator, input llmDebugStateInput) llmDebugViewState {
	ui := input.UI
	state := llmDebugViewState{
		Ready: len(input.ModelNames) > 0, Models: input.ModelNames, SavedPrompt: input.SavedPrompt,
		Edited:  ui.prompt != input.SavedPrompt,
		Running: ui.running, PromptLabel: translator.T("System prompt"), InputLabel: translator.T("Recognised text"),
		Form: llmDebugFormViewState{Model: ui.model, Preset: ui.preset, Filled: ui.filled, Prompt: ui.prompt, Input: ui.input, Instruction: ui.instruction},
	}
	for _, preset := range input.Presets {
		state.Presets = append(state.Presets, preset.title)
		saved := input.SavedPrompts[llmDebugPresetKey(preset)]
		presetState := llmDebugPresetViewStateFor(translator, preset, saved)
		state.PresetStates = append(state.PresetStates, presetState)
		if preset.title == input.Selected.title {
			state.PromptLabel, state.InputLabel, state.Instruction = presetState.PromptLabel, presetState.InputLabel, presetState.Instruction
		}
	}
	state.CanRun = llmDebugCanRun(input.Selected.feature, ui)
	if ui.res != nil {
		state.Result = &llmDebugResultViewState{Answer: ui.res.answer, Error: ui.res.err}
		if ui.res.err == "" {
			state.Result.AnswerTitle = translator.T("Answer in %s", fmt.Sprintf("%.1f s", ui.res.took.Seconds()))
		}
		for _, message := range ui.res.msgs {
			state.Result.Messages = append(state.Result.Messages, llmDebugMessageViewState{Role: message.Role, Content: message.Content})
		}
	}
	return state
}

func llmDebugPresetViewStateFor(translator *i18n.Translator, preset llmDebugPreset, saved string) llmDebugPresetViewState {
	state := llmDebugPresetViewState{Title: preset.title, Feature: preset.feature, Group: preset.group,
		SavedPrompt: saved, PromptLabel: translator.T("System prompt"), InputLabel: translator.T("Recognised text")}
	if preset.group != "" {
		state.PromptLabel = translator.T("Group prompt")
	}
	if preset.feature == hotkey.Rewrite {
		state.Instruction = true
		state.InputLabel = translator.T("Selected text (optional)")
	} else if preset.feature == hotkey.Translate {
		state.InputLabel = translator.T("Text to translate")
	}
	return state
}

func llmDebugPresetKey(preset llmDebugPreset) string { return preset.feature + "\x00" + preset.group }

func llmDebugCanRun(feature string, ui *llmDebugState) bool {
	return !ui.running && (strings.TrimSpace(ui.input) != "" || feature == hotkey.Rewrite && strings.TrimSpace(ui.instruction) != "")
}

func llmDebugLocalState(state llmDebugViewState) llmDebugUI {
	return llmDebugUI{model: state.Form.Model, preset: state.Form.Preset, filled: state.Form.Filled,
		prompt: state.Form.Prompt, input: state.Form.Input, instruction: state.Form.Instruction}
}

// llmDebugStateForLocal recomputes fixture-derived state after a demo edit,
// just as llmDebugModel recomputes production state before each view.
func llmDebugStateForLocal(state llmDebugViewState, local *llmDebugUI) llmDebugViewState {
	for _, preset := range state.PresetStates {
		if preset.Title != local.preset {
			continue
		}
		if local.filled != local.preset {
			local.prompt, local.filled = preset.SavedPrompt, local.preset
			state.Result = nil
		}
		state.SavedPrompt = preset.SavedPrompt
		state.PromptLabel, state.InputLabel, state.Instruction = preset.PromptLabel, preset.InputLabel, preset.Instruction
		controller := llmDebugState{llmDebugUI: *local, running: state.Running}
		state.CanRun = llmDebugCanRun(preset.Feature, &controller)
		break
	}
	state.Edited = local.prompt != state.SavedPrompt
	state.Form = llmDebugFormViewState{Model: local.model, Preset: local.preset, Filled: local.filled,
		Prompt: local.prompt, Input: local.input, Instruction: local.instruction}
	return state
}
