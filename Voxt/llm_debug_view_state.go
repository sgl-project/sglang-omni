package main

// Serializable display state for the language model debug page.

import (
	"fmt"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type llmDebugViewState struct {
	Ready          bool                     `json:"ready"`
	Models         []string                 `json:"models,omitempty"`
	Presets        []string                 `json:"presets,omitempty"`
	SelectedModel  string                   `json:"selected_model,omitempty"`
	SelectedPreset string                   `json:"selected_preset,omitempty"`
	SavedPrompt    string                   `json:"saved_prompt,omitempty"`
	PromptLabel    string                   `json:"prompt_label,omitempty"`
	InputLabel     string                   `json:"input_label,omitempty"`
	Instruction    bool                     `json:"instruction,omitempty"`
	Edited         bool                     `json:"edited"`
	CanRun         bool                     `json:"can_run"`
	Running        bool                     `json:"running"`
	Result         *llmDebugResultViewState `json:"result,omitempty"`
	Form           llmDebugFormViewState    `json:"form"`
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
	ModelNames  []string
	Presets     []llmDebugPreset
	Selected    llmDebugPreset
	SavedPrompt string
	UI          *llmDebugState
}

func buildLLMDebugViewState(translator *i18n.Translator, input llmDebugStateInput) llmDebugViewState {
	ui := input.UI
	state := llmDebugViewState{
		Ready: len(input.ModelNames) > 0, Models: input.ModelNames, SavedPrompt: input.SavedPrompt,
		SelectedModel: ui.model, SelectedPreset: ui.preset, Edited: ui.prompt != input.SavedPrompt,
		Running: ui.running, PromptLabel: translator.T("System prompt"), InputLabel: translator.T("Recognised text"),
		Form: llmDebugFormViewState{Model: ui.model, Preset: ui.preset, Filled: ui.filled, Prompt: ui.prompt, Input: ui.input, Instruction: ui.instruction},
	}
	for _, preset := range input.Presets {
		state.Presets = append(state.Presets, preset.title)
	}
	if input.Selected.group != "" {
		state.PromptLabel = translator.T("Group prompt")
	}
	if input.Selected.feature == hotkey.Rewrite {
		state.Instruction = true
		state.InputLabel = translator.T("Selected text (optional)")
	} else if input.Selected.feature == hotkey.Translate {
		state.InputLabel = translator.T("Text to translate")
	}
	state.CanRun = !ui.running && (strings.TrimSpace(ui.input) != "" || input.Selected.feature == hotkey.Rewrite && strings.TrimSpace(ui.instruction) != "")
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

func llmDebugLocalState(state llmDebugViewState) llmDebugUI {
	return llmDebugUI{model: state.Form.Model, preset: state.Form.Preset, filled: state.Form.Filled,
		prompt: state.Form.Prompt, input: state.Form.Input, instruction: state.Form.Instruction}
}
