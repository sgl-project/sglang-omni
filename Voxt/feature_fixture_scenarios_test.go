package main

import (
	"errors"
	"fmt"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type featureFixtureScenarioInput struct {
	Feature           string
	Settings          settings.Settings
	Local             featureUI
	ReadyASR          []local.Model
	ShownASR          []local.Model
	ReadyLLM          []local.Model
	HotkeysInstalled  bool
	HelperRecordsKeys bool
	EndWordStreaming  bool
	UsedBy            map[string][]string
	Snapshot          session.Snapshot
	Mine              bool
	Block             *sessionNotice
	Now               time.Time
}

func featureFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	add := func(name string, input featureFixtureScenarioInput) {
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "feature", feature: input})
	}
	add("feature-empty", featureFixtureEmptyInput())
	add("feature-loading", featureFixtureTryInput(hotkey.Transcribe, session.Recognizing, ""))
	add("feature-normal", featureFixtureBaseInput(hotkey.Transcribe))
	failed := featureFixtureTryInput(hotkey.Transcribe, session.Failed, session.ProblemInternal)
	add("feature-error", failed)
	for _, feature := range []string{hotkey.Transcribe, hotkey.Translate, hotkey.Rewrite} {
		add("feature-"+feature+"-normal", featureFixtureBaseInput(feature))
	}
	for _, state := range []session.State{
		session.Idle, session.Reading, session.Recording, session.Recognizing, session.Composing,
		session.Delivering, session.Done, session.Cancelled, session.Failed,
	} {
		add("feature-try-state-"+string(state), featureFixtureTryInput(hotkey.Transcribe, state, ""))
	}
	for _, code := range historyFixtureProblemCodes() {
		add("feature-problem-"+code, featureFixtureTryInput(hotkey.Transcribe, session.Failed, code))
	}
	for _, code := range []string{session.ProblemASR, session.ProblemLLM, session.ProblemTruncated} {
		input := featureFixtureTryInput(hotkey.Transcribe, session.Done, code)
		input.Snapshot.Problem.Degraded = true
		add("feature-problem-"+code+"-degraded", input)
	}

	shortcutProblems := featureFixtureBaseInput(hotkey.Transcribe)
	shortcutProblems.Settings.Features.Transcribe.Hotkeys = []settings.Hotkey{
		{Keys: "Control+Option+K+J"},
		{Keys: "RightOption+Shift", Trigger: settings.TriggerTap},
		{Keys: "RightOption+Shift", Trigger: settings.TriggerTap},
		{Keys: "Command+Space"},
		{Keys: "K"},
	}
	add("feature-shortcut-problems", shortcutProblems)
	customKeys := featureFixtureBaseInput(hotkey.Transcribe)
	customKeys.Local.rows = []keyRow{{custom: true, typing: true, edited: true, text: "Control+Option+K+J", sel: "Custom…"}}
	add("feature-shortcut-custom-error", customKeys)

	for size := range settings.CardSizeSteps {
		input := featureFixtureBaseInput(hotkey.Rewrite)
		input.Settings.Features.CardSizes = map[string]int{hotkey.Rewrite: size}
		add(fmt.Sprintf("feature-card-size-%d", size), input)
	}
	rules := featureFixtureBaseInput(hotkey.Transcribe)
	rules.Local.rules, rules.Local.preview, rules.Local.find = true, true, "meeting"
	add("feature-rules-preview", rules)
	targetWarning := featureFixtureBaseInput(hotkey.Translate)
	targetWarning.Settings.Features.Translate.Prompt = "Translate Chinese into English."
	targetWarning.Local.rules, targetWarning.Local.preview = true, true
	add("feature-translate-target-warning", targetWarning)
	noOtherLanguages := featureFixtureBaseInput(hotkey.Translate)
	noOtherLanguages.Settings.General.OtherLanguages = nil
	noOtherLanguages.Local.rules = true
	add("feature-rules-no-other-languages", noOtherLanguages)
	enhanceOff := featureFixtureBaseInput(hotkey.Transcribe)
	enhanceOff.Settings.Features.Transcribe.Enhance = false
	enhanceOff.Local.rules, enhanceOff.Local.preview = true, true
	add("feature-enhance-off", enhanceOff)

	noModels := featureFixtureBaseInput(hotkey.Transcribe)
	noModels.ReadyASR, noModels.ShownASR, noModels.ReadyLLM = nil, nil, nil
	add("feature-models-none", noModels)
	fallback := featureFixtureBaseInput(hotkey.Translate)
	fallback.ReadyLLM = append(fallback.ReadyLLM, local.Model{Kind: local.LLM, ID: "fixture-language-fallback", Name: "Fixture fallback language model"})
	fallback.Settings.Features.Translate.Fallback = "fixture-language-fallback"
	add("feature-models-fallback", fallback)
	endWord := featureFixtureBaseInput(hotkey.Transcribe)
	endWord.Settings.Features.EndWord = "fixture ending phrase"
	add("feature-end-word-custom", endWord)
	noAccessibility := featureFixtureBaseInput(hotkey.Transcribe)
	noAccessibility.HotkeysInstalled = false
	add("feature-no-accessibility", noAccessibility)
	blocked := featureFixtureBaseInput(hotkey.Transcribe)
	text, more, action := sessionProblemText(&session.Problem{Code: session.ProblemNoASR, Err: errors.New("fixture missing speech model")})
	blocked.Block = &sessionNotice{text: text, more: more, action: action}
	add("feature-blocked-start", blocked)
	return scenarios
}

func featureFixtureBaseInput(feature string) featureFixtureScenarioInput {
	speech := local.Model{Kind: local.ASR, ID: "fixture-speech", Name: "Fixture speech model", Stream: true}
	language := local.Model{Kind: local.LLM, ID: "fixture-language", Name: "Fixture language model"}
	return featureFixtureScenarioInput{
		Feature: feature, Settings: settings.Defaults(), ReadyASR: []local.Model{speech}, ShownASR: []local.Model{speech}, ReadyLLM: []local.Model{language},
		HotkeysInstalled: true, EndWordStreaming: true, Now: meetingFixtureNow,
	}
}

func featureFixtureEmptyInput() featureFixtureScenarioInput {
	input := featureFixtureBaseInput(hotkey.Transcribe)
	input.ReadyASR, input.ShownASR, input.ReadyLLM = nil, nil, nil
	input.Settings.Features.Transcribe.Hotkeys = nil
	return input
}

func featureFixtureTryInput(feature string, state session.State, problemCode string) featureFixtureScenarioInput {
	input := featureFixtureBaseInput(feature)
	snapshot := session.Snapshot{
		Rev: 1, ID: "fixture-try", Kind: featureKind(feature), State: state, Try: true,
		StartedAt: meetingFixtureNow.Add(-4 * time.Second), HeardAt: meetingFixtureNow.Add(-3 * time.Second),
	}
	switch state {
	case session.Recording:
		snapshot.Levels = []float64{0.1, 0.3, 0.6, 0.9, 0.5, 0.2, 0.4, 0.7, 0.3, 0.1}
		snapshot.Partial = "so the meeting moves to Thursday"
	case session.Composing:
		snapshot.Draft = "The meeting moves to Thursday at four."
	case session.Done:
		snapshot.StoppedAt = meetingFixtureNow
		snapshot.Raw = "um so the meeting moves to Thursday at three, no, at four"
		snapshot.Result = "The meeting moves to Thursday at four."
		snapshot.ASR = session.Model{Name: "Fixture speech model"}
		snapshot.LLM = session.Model{Name: "Fixture language model"}
		snapshot.Timings = session.Timings{Audio: 3200 * time.Millisecond, ASR: 800 * time.Millisecond, LLM: 1100 * time.Millisecond, Total: 5300 * time.Millisecond}
	}
	if problemCode != "" {
		snapshot.Problem = &session.Problem{Code: problemCode, Err: errors.New("fixture problem")}
	}
	input.Snapshot, input.Mine = snapshot, true
	return input
}

func buildFeatureFixtureViewState(translator *i18n.Translator, input featureFixtureScenarioInput) featureViewState {
	local := input.Local
	feature := input.Feature
	if !local.promptDirty {
		saved := promptOf(input.Settings, feature)
		local.prompt = saved
		if saved == "" {
			local.prompt = promptBuiltinFor(input.Settings, feature, translator)
		}
	}
	endWord := buildFeatureEndWordViewState(translator, featureEndWordStateInput{
		Current: input.Settings.Features.EndWord, Streaming: input.EndWordStreaming, Local: &local.endWord,
	})
	shortcuts := buildFeatureShortcutsViewState(translator, featureShortcutsStateInput{
		Feature: feature, Settings: input.Settings, Local: &local, HotkeysInstalled: input.HotkeysInstalled,
		HelperRecordsKeys: input.HelperRecordsKeys, UsedBy: input.UsedBy, EndWord: endWord,
	})
	models := buildFeatureModelsViewState(translator, featureModelsStateInput{
		Feature: feature, Settings: input.Settings, ReadyASR: input.ReadyASR, ShownASR: input.ShownASR, ReadyLLM: input.ReadyLLM, Local: &local,
	})
	prompt := buildFeaturePromptViewState(translator, featurePromptStateInput{Feature: feature, Settings: input.Settings, Local: &local})
	keys := ""
	if hotkeys := hotkey.HotkeysOf(input.Settings.Features, feature); len(hotkeys) > 0 {
		keys = uitext.HotkeyText(hotkeys[0].Keys)
	}
	tryInput := newTryCardStateInput(uitext.FeaturePrefix(feature), feature, input.Mine, buildTryIntro(translator, feature, keys), input.Block, input.Snapshot, input.Now)
	tryState := buildTryCardViewState(translator, tryInput)
	return buildFeatureViewState(translator, featurePageStateInput{
		Feature: feature, RefreshRecording: input.Snapshot.Try && input.Snapshot.State == session.Recording,
		Shortcuts: shortcuts, Models: models, Prompt: prompt, Try: tryState,
	})
}
