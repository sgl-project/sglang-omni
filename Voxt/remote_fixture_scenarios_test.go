package main

import (
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/codexauth"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func remoteFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	addPage := func(name string, mutate func(*remotePageStateInput)) {
		input := remoteFixturePageInput()
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "remote", remote: input})
	}
	addPage("remote-page-empty", func(input *remotePageStateInput) { input.Cards.Services = nil })
	addPage("remote-page-loading", func(input *remotePageStateInput) {
		input.Cards.Services[0].Mask, input.Cards.Services[0].KeyRead = "", false
		input.Cards.Services[0].Test = providerTest{running: true}
	})
	addPage("remote-page-normal", func(input *remotePageStateInput) {})
	addPage("remote-page-error", func(input *remotePageStateInput) {
		input.Cards.Services[0].Test = remoteFixtureTest(&llm.Error{Code: llm.CodeNetwork})
	})
	addPage("remote-page-key-spaced", func(input *remotePageStateInput) { input.Cards.Services[0].KeySpaced = true })
	addPage("remote-onboarding-back", func(input *remotePageStateInput) { input.BackBarText = "The setup guide is paused at the Models step." })
	for _, missing := range []string{providers.MissingKey, providers.MissingEndpoint, providers.MissingAddress, providers.MissingModel} {
		addPage("remote-missing-"+missing, func(input *remotePageStateInput) { input.Cards.Services[0].Missing = missing })
	}
	for _, wait := range settings.WaitChoices {
		name := wait
		if name == "" {
			name = "automatic"
		}
		addPage("remote-wait-"+name, func(input *remotePageStateInput) {
			input.Wait.Value = wait
			input.Wait.CurrentMinutes = settings.Models{LLMWait: wait}.LLMWaitFor().Minutes()
		})
	}
	addPage("remote-wait-custom", func(input *remotePageStateInput) {
		input.Wait.Value = "45m"
		input.Wait.CurrentMinutes = 45
	})

	addCards := func(name string, mutate func(*remoteCardsStateInput)) {
		input := remoteFixtureCardsInput()
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "remote-cards", remoteCards: input})
	}
	addCards("remote-cards-empty", func(input *remoteCardsStateInput) { input.Services = nil })
	addCards("remote-cards-loading", func(input *remoteCardsStateInput) {
		input.Services[0].Mask, input.Services[0].KeyRead = "", false
		input.Services[0].Test = providerTest{running: true}
	})
	addCards("remote-cards-normal", func(input *remoteCardsStateInput) {})
	addCards("remote-cards-upload", func(input *remoteCardsStateInput) {
		input.Services[0].Service.speech.Uploads = []string{"fixture-model"}
	})
	addCards("remote-cards-error", func(input *remoteCardsStateInput) {
		input.Services[0].Test = remoteFixtureTest(&asr.Error{Kind: asr.KindNetwork, Provider: providers.StepFunASR})
	})

	addDialog := func(name string, mutate func(*remoteDialogStateInput)) {
		input := remoteFixtureDialogInput(local.LLM)
		mutate(input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "remote-dialog", remoteDialog: input})
	}
	addDialog("remote-dialog-empty", func(input *remoteDialogStateInput) {
		input.Dialog.exists, input.Mask = false, ""
	})
	addDialog("remote-dialog-loading", func(input *remoteDialogStateInput) { input.Dialog.test = providerTest{running: true} })
	addDialog("remote-dialog-saving", func(input *remoteDialogStateInput) { input.Dialog.saving = true })
	addDialog("remote-dialog-normal", func(input *remoteDialogStateInput) {})
	addDialog("remote-dialog-error", func(input *remoteDialogStateInput) { input.Dialog.gen.advanced = false })
	scenarios[len(scenarios)-1].err = &llm.Error{Code: llm.CodeConfig}
	addDialog("remote-dialog-unsaved", func(input *remoteDialogStateInput) {
		input.Dialog.endpoint = "changed"
		input.Unsaved = input.Dialog.unsaved()
	})
	addDialog("remote-dialog-key-spaced", func(input *remoteDialogStateInput) {
		input.Dialog.key = "fix ture"
		input.Unsaved = input.Dialog.unsaved()
	})
	addDialog("remote-dialog-remove-key", func(input *remoteDialogStateInput) {
		input.Dialog.removeKey = true
		input.Unsaved = input.Dialog.unsaved()
	})
	addDialog("remote-dialog-custom-model", func(input *remoteDialogStateInput) {
		input.Dialog.model, input.Dialog.customModel = otherModel(), "custom-model"
	})
	addDialog("remote-dialog-models", func(input *remoteDialogStateInput) {
		test := remoteFixtureTest(&llm.Error{Code: llm.CodeModelNotFound})
		test.res.Models = []string{"model-1", "model-2", "model-3", "model-4", "model-5", "model-6", "model-7", "model-8", "model-9"}
		input.Dialog.test = test
	})
	addDialog("remote-dialog-frequency-only", func(input *remoteDialogStateInput) {
		input.Capabilities.Penalties, input.Capabilities.FrequencyOnly = false, true
	})
	for _, mode := range []string{llm.ThinkDefault, llm.ThinkOff, llm.ThinkOn, llm.ThinkEffort, llm.ThinkBudget} {
		name := mode
		if name == "" {
			name = "default"
		}
		addDialog("remote-generation-think-"+name, func(input *remoteDialogStateInput) {
			input.Dialog.gen.think = mode
			if mode == llm.ThinkEffort {
				input.Dialog.gen.effort = "medium"
			}
		})
	}
	for _, effort := range []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"} {
		addDialog("remote-generation-effort-"+effort, func(input *remoteDialogStateInput) {
			input.Dialog.gen.think, input.Dialog.gen.effort = llm.ThinkEffort, effort
		})
	}
	for _, format := range responseFormats {
		name := format
		if name == "" {
			name = "plain"
		}
		addDialog("remote-generation-format-"+name, func(input *remoteDialogStateInput) {
			input.Dialog.gen.format = format
			if format == llm.ResponseJSONSchema {
				input.Dialog.gen.schema = `{"type":"object"}`
			}
		})
	}
	for _, verbosity := range verbosities {
		name := verbosity
		if name == "" {
			name = "default"
		}
		addDialog("remote-generation-verbosity-"+name, func(input *remoteDialogStateInput) { input.Dialog.gen.verbosity = verbosity })
	}
	for _, api := range apiFormats {
		name := api
		if name == "" {
			name = "chat-completions"
		}
		addDialog("remote-generation-api-"+name, func(input *remoteDialogStateInput) { input.Dialog.gen.api = apiFormatTitle(api) })
	}

	for _, code := range []string{
		llm.CodeNoKey, llm.CodeAuth, llm.CodeRateLimited, llm.CodeServer, llm.CodeOverloaded, llm.CodeBadRequest,
		llm.CodeModelNotFound, llm.CodeContextLength, llm.CodeFiltered, llm.CodeTimeout, llm.CodeCancelled,
		llm.CodeNetwork, llm.CodeBadResponse, llm.CodeEmpty, llm.CodeInvalidJSON, llm.CodeSchema, llm.CodeNoRoute,
		llm.CodeUnknown, llm.CodeUnsupported, llm.CodeConfig, llm.CodeAddress, llm.CodeResponsesOnly,
		llm.CodeSignIn, llm.CodeUsageLimit, llm.CodeNotEligible,
	} {
		addDialog("remote-problem-"+strings.ReplaceAll(strings.TrimPrefix(code, "llm."), "_", "-"), func(input *remoteDialogStateInput) {
			input.Dialog.test = remoteFixtureTest(&llm.Error{Code: code})
		})
	}
	for _, kind := range []asr.Kind{asr.KindConfig, asr.KindAuth, asr.KindQuota, asr.KindNetwork, asr.KindProtocol, asr.KindServer, asr.KindAudio, asr.KindCancelled, asr.KindTimeout} {
		scenarios = append(scenarios, meetingFixtureScenario{name: "remote-speech-problem-" + string(kind), kind: "remote-dialog", remoteDialog: remoteFixtureSpeechProblem(kind)})
	}

	for _, streaming := range []bool{true, false} {
		input := remoteFixtureDialogInput(local.ASR)
		input.Dialog.id, input.Service.id = "fixture-speech", "fixture-speech"
		input.Service.name = "Fixture speech"
		input.Service.speech = providers.Speech{Stream: true, Uploads: []string{"upload-model"}}
		if !streaming {
			input.Model = "upload-model"
			input.Dialog.model = "upload-model"
		}
		input.Service.appID = true
		input.Dialog.recognitionPrompt = "Fixture vocabulary"
		name := "remote-dialog-speech-realtime"
		if !streaming {
			name = "remote-dialog-speech-upload"
		}
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "remote-dialog", remoteDialog: input})
	}

	addChat := func(name string, login chatgptLogin, email string, err error) {
		input := remoteFixtureDialogInput(local.LLM)
		input.Service.signIn = true
		input.ChatLogin, input.ChatEmail = login, email
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "remote-dialog", remoteDialog: input, err: err})
	}
	addChat("remote-chatgpt-signin", chatgptLogin{}, "", nil)
	addChat("remote-chatgpt-running", chatgptLogin{running: true}, "", nil)
	addChat("remote-chatgpt-signed-in", chatgptLogin{}, "person@example.invalid", nil)
	for _, kind := range []codexauth.Kind{codexauth.KindSignedOut, codexauth.KindPortInUse, codexauth.KindDenied, codexauth.KindTimeout, codexauth.KindExchange, codexauth.KindNetwork, codexauth.KindStore} {
		addChat("remote-chatgpt-problem-"+string(kind), chatgptLogin{}, "", &codexauth.Error{Kind: kind})
	}

	for _, feature := range []string{hotkey.Transcribe, hotkey.Translate, hotkey.Rewrite} {
		scenarios = append(scenarios, meetingFixtureScenario{name: "remote-debug-" + feature, kind: "remote-debug", remoteDebug: remoteFixtureDebugInput(feature)})
	}
	scenarios = append(scenarios,
		meetingFixtureScenario{name: "remote-debug-empty", kind: "remote-debug"},
		meetingFixtureScenario{name: "remote-debug-loading", kind: "remote-debug", remoteDebug: remoteFixtureDebugRunning(true)},
		meetingFixtureScenario{name: "remote-debug-error", kind: "remote-debug", remoteDebug: remoteFixtureDebugError()},
	)
	return scenarios
}

func remoteFixturePageInput() remotePageStateInput {
	configured := remoteFixtureServiceState(local.LLM)
	configured.Configured = true
	configured.Used = "Rewrite"
	configured.Mask, configured.KeyRead = "••••", true
	configured.Test = remoteFixtureTest(nil)
	configured.HasTest = true
	localService := remoteFixtureServiceState(local.LLM)
	localService.Service.id, localService.Service.name, localService.Service.endpoint = "local-fixture", "Local fixture", "http://localhost"
	onlineService := remoteFixtureServiceState(local.LLM)
	onlineService.Service.id, onlineService.Service.name = "online-fixture", "Online fixture"
	return remotePageStateInput{
		Wait:  remoteWaitStateInput{Locale: "en"},
		Cards: remoteCardsStateInput{Kind: local.LLM, Services: []remoteServiceStateInput{configured, localService, onlineService}},
	}
}

func remoteFixtureCardsInput() remoteCardsStateInput {
	configured := remoteFixtureServiceState(local.ASR)
	configured.Service.speech.Stream = true
	configured.Configured = true
	configured.Mask, configured.KeyRead = "••••", true
	configured.Test = remoteFixtureTest(nil)
	configured.HasTest = true
	unconfigured := remoteFixtureServiceState(local.ASR)
	unconfigured.Service.id, unconfigured.Service.name = "online-speech", "Online speech"
	return remoteCardsStateInput{Kind: local.ASR, Services: []remoteServiceStateInput{configured, unconfigured}}
}

func remoteFixtureServiceState(kind local.Kind) remoteServiceStateInput {
	sv := service{kind: kind, id: "fixture", name: "Fixture service", model: "fixture-model", models: []string{"fixture-model", "other-model"}, keyOptional: true, keyURL: "https://example.invalid"}
	return remoteServiceStateInput{Service: sv, Setup: settings.Provider{ID: sv.id, Model: "fixture-model"}}
}

func remoteFixtureTest(err error) providerTest {
	return providerTest{res: providers.TestResult{Err: err, At: meetingFixtureNow, Latency: 412 * time.Millisecond}, detail: "redacted fixture detail"}
}

func remoteFixtureCaps() providers.GenCaps {
	return providers.GenCaps{
		ThinkModes:      []string{llm.ThinkDefault, llm.ThinkOff, llm.ThinkOn, llm.ThinkEffort, llm.ThinkBudget},
		Efforts:         []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"},
		MaxOutputTokens: true, Temperature: true, TopP: true, TopK: true, MinP: true, Seed: true,
		Logprobs: true, Stop: true, Penalties: true, ResponseFormat: true, JSONSchema: true, ExtraBody: true,
		Verbosity: true, KeepAlive: true, IncludeUsage: true, APIFormat: true,
	}
}

func remoteFixtureDialogInput(kind local.Kind) *remoteDialogStateInput {
	serviceInput := remoteFixtureServiceState(kind)
	dialog := &providerDialog{kind: kind, id: serviceInput.Service.id, exists: true}
	dialog.providerDialogLocalUI = providerDialogLocalUI{
		open: true, model: "fixture-model",
		gen: genForm{think: llm.ThinkOn, format: "", verbosity: "", usage: true, advanced: true, api: apiFormatTitle("")},
	}
	dialog.was = dialog.setup()
	saved := dialog.providerDialogLocalUI
	return &remoteDialogStateInput{
		Dialog: dialog, Saved: &saved, Service: serviceInput.Service, Model: "fixture-model", Capabilities: remoteFixtureCaps(),
		Mask: "••••", ChatEmail: "person@example.invalid",
	}
}

func remoteFixtureSpeechProblem(kind asr.Kind) *remoteDialogStateInput {
	input := remoteFixtureDialogInput(local.ASR)
	input.Dialog.test = remoteFixtureTest(&asr.Error{Kind: kind, Provider: providers.StepFunASR})
	return input
}

func remoteFixtureDebugInput(feature string) llmDebugStateInput {
	presets := []llmDebugPreset{{feature: hotkey.Transcribe}, {feature: hotkey.Translate}, {feature: hotkey.Rewrite}}
	return llmDebugStateInput{
		ModelNames: []string{"Fixture service · fixture-model"}, Presets: presets,
		SavedPrompt: "Fixture saved prompt",
		UI: &llmDebugState{llmDebugUI: llmDebugUI{model: "Fixture service · fixture-model", preset: feature, filled: feature, prompt: "Fixture edited prompt", input: "Fixture input", instruction: "Make it shorter"},
			res: &llmDebugResult{answer: "Fixture answer", took: 1200 * time.Millisecond, msgs: []compose.Message{{Role: "system", Content: "Fixture system"}, {Role: "user", Content: "Fixture input"}}}},
	}
}

func remoteFixtureDebugRunning(running bool) llmDebugStateInput {
	input := remoteFixtureDebugInput(hotkey.Transcribe)
	input.UI.running = running
	input.UI.res = nil
	return input
}

func remoteFixtureDebugError() llmDebugStateInput {
	input := remoteFixtureDebugInput(hotkey.Transcribe)
	input.UI.res = &llmDebugResult{err: "fixture run failed"}
	return input
}

func localizeRemotePageInput(input *remotePageStateInput, translator *i18n.Translator) {
	input.Cards.Services = append([]remoteServiceStateInput(nil), input.Cards.Services...)
	if input.BackBarText != "" {
		input.BackBarText = translator.T("The setup guide is paused at the %s step.", translator.T("Models"))
	}
	for index := range input.Cards.Services {
		if input.Cards.Services[index].Used != "" {
			input.Cards.Services[index].Used = translator.T("Used by %s", translator.T(input.Cards.Services[index].Used))
		}
	}
}

func localizeRemoteDialogInput(input *remoteDialogStateInput, translator *i18n.Translator) {
	dialog := input.Dialog
	if dialog.customModel != "" {
		dialog.model = otherModelFor(translator)
	}
	form := &dialog.gen
	form.think = thinkTitleFor(translator, remoteFixtureValue(form.think, []string{llm.ThinkDefault, llm.ThinkOff, llm.ThinkOn, llm.ThinkEffort, llm.ThinkBudget}, thinkTitle))
	form.format = formatTitleFor(translator, remoteFixtureValue(form.format, responseFormats, formatTitle))
	form.verbosity = verbosityTitleFor(translator, remoteFixtureValue(form.verbosity, verbosities, verbosityTitle))
}

func remoteFixtureValue(current string, values []string, titleOf func(string) string) string {
	for _, value := range values {
		if current == value {
			return value
		}
	}
	return valueOf(current, values, titleOf)
}

func localizeRemoteDebugInput(input *llmDebugStateInput, translator *i18n.Translator) {
	selected := input.UI
	var preset string
	if selected != nil {
		preset = selected.preset
	}
	for index := range input.Presets {
		feature := input.Presets[index].feature
		input.Presets[index].title = remoteDebugPresetTitle(translator, feature, input.Presets[index].group)
		if feature == preset {
			input.Selected = input.Presets[index]
		}
	}
	if input.Selected.title == "" && len(input.Presets) > 0 {
		input.Selected = input.Presets[0]
	}
	input.SavedPrompts = map[string]string{}
	for _, preset := range input.Presets {
		input.SavedPrompts[llmDebugPresetKey(preset)] = input.SavedPrompt + " · " + preset.title
	}
	if selected != nil {
		if len(input.ModelNames) > 0 {
			selected.model = input.ModelNames[0]
		}
		selected.preset = input.Selected.title
		selected.filled = input.Selected.title
	}
}

func remoteDebugPresetTitle(translator *i18n.Translator, feature, group string) string {
	if group != "" {
		return translator.T("App enhancement: %s", group)
	}
	switch feature {
	case hotkey.Translate:
		return translator.T("Translate")
	case hotkey.Rewrite:
		return translator.T("Rewrite")
	}
	return translator.T("Transcribe")
}
