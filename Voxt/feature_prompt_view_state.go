package main

import (
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type featureCardViewState struct {
	Show bool `json:"show"`
	Size int  `json:"size"`
}

type featureVariableViewState struct {
	Name  string `json:"name"`
	Token string `json:"token"`
	Text  string `json:"text"`
}

type featureRulesViewState struct {
	Open          bool                       `json:"open"`
	PreviewOpen   bool                       `json:"preview_open"`
	FindQuery     string                     `json:"find_query"`
	SettingsOpen  bool                       `json:"settings_open"`
	Prompt        string                     `json:"prompt"`
	Summary       string                     `json:"summary"`
	ResetDisabled bool                       `json:"reset_disabled"`
	Matches       [][2]int                   `json:"matches"`
	Variables     []featureVariableViewState `json:"variables"`
	EnhanceOff    bool                       `json:"enhance_off"`
	PreviewError  string                     `json:"preview_error"`
	// PreviewPrompt is the whole prompt, its messages joined in order;
	// PreviewValues are the variables' values it marks (issue #4032).
	PreviewPrompt string   `json:"preview_prompt"`
	PreviewValues []string `json:"preview_values"`
}

type featurePromptViewState struct {
	Target        *featureSelectViewState `json:"target,omitempty"`
	TargetWarning bool                    `json:"target_warning"`
	Cleanup       bool                    `json:"cleanup"`
	Card          featureCardViewState    `json:"card"`
	Rules         featureRulesViewState   `json:"rules"`
}

type featurePromptStateInput struct {
	Feature      string
	Settings     settings.Settings
	SettingsOpen bool
	Local        *featureUI
	// Glossary is the whole dictionary (loadGlossaryTerms), for the preview.
	Glossary []store.Term
}

func buildFeaturePromptViewState(translator *i18n.Translator, input featurePromptStateInput) featurePromptViewState {
	feature := input.Feature
	local := input.Local
	if local == nil {
		local = &featureUI{}
	}
	state := featurePromptViewState{Card: featureCardViewState{Size: input.Settings.Features.CardSize(feature)}}
	switch feature {
	case hotkey.Translate:
		state.Card.Show = input.Settings.Features.Translate.ShowCard
		state.Cleanup = input.Settings.Features.Translate.Cleanup
		target := featureTranslateTargetSelect(translator, input.Settings)
		state.Target = &target
		own := promptOf(input.Settings, feature)
		state.TargetWarning = own != "" && !strings.Contains(own, targetVar)
	case hotkey.Rewrite:
		state.Card.Show = input.Settings.Features.Rewrite.ShowCard
		state.Cleanup = input.Settings.Features.Rewrite.Cleanup
	default:
		state.Card.Show = input.Settings.Features.Transcribe.ShowCard
	}
	state.Rules = buildFeatureRulesViewState(translator, input)
	return state
}

func featureTranslateTargetSelect(translator *i18n.Translator, settingsValue settings.Settings) featureSelectViewState {
	automatic := translator.T("Automatic (now: %s)", compose.LanguageName(session.TranslateTarget(settings.Settings{General: settingsValue.General}, translator.Locale(), ""), translator.Locale()))
	state := featureSelectViewState{Value: automatic}
	state.Options = append(state.Options, featureOptionViewState{Label: automatic})
	for _, code := range compose.TargetLanguages {
		state.Options = append(state.Options, featureOptionViewState{Label: compose.LanguageName(code, translator.Locale()), Value: code})
	}
	if index := slices.Index(compose.TargetLanguages, settingsValue.Features.Translate.Target); index >= 0 {
		state.Value = state.Options[index+1].Label
	} else if target := settingsValue.Features.Translate.Target; target != "" {
		state.Value = compose.LanguageName(target, translator.Locale())
		state.Options = append(state.Options, featureOptionViewState{Label: state.Value, Value: target})
	}
	return state
}

func buildFeatureRulesViewState(translator *i18n.Translator, input featurePromptStateInput) featureRulesViewState {
	feature := input.Feature
	local := input.Local
	if local == nil {
		local = &featureUI{}
	}
	state := featureRulesViewState{Open: local.rules, PreviewOpen: local.preview, FindQuery: local.find, SettingsOpen: input.SettingsOpen, Prompt: local.prompt, Summary: translator.T("Not set")}
	if count := utf8.RuneCountInString(strings.TrimSpace(local.prompt)); count > 0 {
		state.Summary = translator.N(count, "%d character", "%d characters", count)
	}
	if local.rules {
		state.Matches = findFolded(local.prompt, local.find)
		builtin := promptBuiltinFor(input.Settings, feature, translator)
		state.ResetDisabled = promptOf(input.Settings, feature) == "" && strings.TrimSpace(local.prompt) == strings.TrimSpace(builtin)
		previewInput := previewInputFor(input.Settings, feature, local.prompt, translator)
		values := compose.Variables(previewInput, compose.TemplateSet(previewInput.MainLanguage))
		for _, name := range featureVariables(feature) {
			value := values[name]
			if strings.TrimSpace(value) == "" {
				value = translator.T("(none)")
			}
			state.Variables = append(state.Variables, featureVariableViewState{
				Name: name, Token: "{{" + name + "}}", Text: variableInfoFor(translator, name) + " " + translator.T("Now: %s", value),
			})
		}
	}
	state.EnhanceOff = feature == hotkey.Transcribe && !input.Settings.Features.Transcribe.Enhance
	if local.rules && local.preview {
		output, err := composedPreviewFor(input.Settings, feature, local.prompt, input.Glossary, translator)
		if err != nil {
			state.PreviewError = err.Error()
		} else {
			state.PreviewPrompt = wholePrompt(output)
			previewInput := previewInputFor(input.Settings, feature, local.prompt, translator)
			values := compose.Variables(previewInput, compose.TemplateSet(previewInput.MainLanguage))
			for _, name := range featureVariables(feature) {
				state.PreviewValues = append(state.PreviewValues, values[name])
			}
		}
	}
	return state
}

func variableInfoFor(translator *i18n.Translator, name string) string {
	switch name {
	case "USER_MAIN_LANGUAGE":
		return translator.T("Your main language, from General settings. Every feature.")
	case "USER_OTHER_LANGUAGES":
		return translator.T("Your other languages, from General settings. Every feature.")
	case "TARGET_LANGUAGE":
		return translator.T("The Target language menu on the Translate page. Translate only.")
	case "APP_NAME":
		return translator.T("The app in front, when an App enhancement group matches it. The features that group applies to.")
	}
	return ""
}

func promptBuiltinFor(settingsValue settings.Settings, feature string, translator *i18n.Translator) string {
	input := previewInputFor(settingsValue, feature, "", translator)
	input.KeepTarget = true
	output, err := compose.Build(input)
	if err != nil {
		return ""
	}
	return output.System
}

// composedPreviewFor is the request a session would send with the
// settings and the rules as typed, for the sample text: the personal
// dictionary (session.PromptDictionary) and the glossary's enabled terms,
// as a session adds them (run_listen.go, write.go).
func composedPreviewFor(settingsValue settings.Settings, feature, prompt string, glossary []store.Term, translator *i18n.Translator) (compose.Output, error) {
	input := previewInputFor(settingsValue, feature, prompt, translator)
	input.Dictionary = session.PromptDictionary(settingsValue)
	if settingsValue.Dictionary.Enabled && settingsValue.Dictionary.Glossary {
		for _, term := range glossary {
			if term.Enabled {
				input.Glossary = append(input.Glossary, compose.Term{Term: term.Term, Aliases: term.Aliases})
			}
		}
	}
	// As savePrompt saves it: the built-in rules are no prompt of one's own,
	// whose layers (the dictionary among them) would come in another order.
	if strings.TrimSpace(input.TaskPrompt) == strings.TrimSpace(promptBuiltinFor(settingsValue, feature, translator)) {
		input.TaskPrompt = ""
	}
	return compose.Build(input)
}

// wholePrompt is output's messages, in order, as one text.
func wholePrompt(output compose.Output) string {
	var parts []string
	for _, message := range output.Messages() {
		parts = append(parts, message.Content)
	}
	return strings.Join(parts, "\n\n")
}

func previewInputFor(settingsValue settings.Settings, feature, prompt string, translator *i18n.Translator) compose.Input {
	general := settingsValue.General
	input := compose.Input{
		TaskPrompt: prompt, MainLanguage: session.MainLanguage(settingsValue, translator.Locale()), OtherLanguages: general.OtherLanguages,
	}
	switch feature {
	case hotkey.Translate:
		featureSettings := settingsValue.Features.Translate
		input.Task, input.Cleanup, input.Text = compose.Translate, featureSettings.Cleanup, trySampleFor(translator)
		input.TargetLanguage = session.TranslateTarget(settingsValue, translator.Locale(), "")
	case hotkey.Rewrite:
		input.Task, input.Cleanup = compose.Rewrite, settingsValue.Features.Rewrite.Cleanup
		input.Instruction, input.Text = rewriteSampleFor(translator)
	default:
		featureSettings := settingsValue.Features.Transcribe
		input.Task, input.Cleanup, input.Structure, input.Text = compose.Transcribe, featureSettings.Cleanup, featureSettings.Structure, trySampleFor(translator)
	}
	return input
}

func trySampleFor(translator *i18n.Translator) string {
	return translator.T("um so the meeting moves to Thursday at three, no, at four")
}

func rewriteSampleFor(translator *i18n.Translator) (instruction, selection string) {
	return translator.T("um make it shorter, no, more formal"), translator.T("hey, the release slips to thursday because the tests are still running, sorry!")
}
