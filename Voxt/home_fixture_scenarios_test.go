package main

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func homeFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	add := func(name string, mutate func(*homeViewStateInput, *homeReadinessInput)) {
		input, readiness := homeFixtureInput(), homeFixtureReadiness()
		mutate(&input, &readiness)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "home", home: input, homeReadiness: readiness})
	}
	add("home-loading", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.Read, input.Loading, input.Recent, input.Today, input.ShowTry = false, true, nil, dayStats{byKind: map[store.Kind]int{}}, true
	})
	add("home-empty", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.Recent, input.Today, input.ShowTry = nil, dayStats{byKind: map[store.Kind]int{}}, true
	})
	add("home-normal", func(_ *homeViewStateInput, _ *homeReadinessInput) {})
	add("home-error", func(input *homeViewStateInput, _ *homeReadinessInput) { input.Error = "fixture read failed" })
	add("home-error-initial", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.Error, input.Recent, input.Today = "fixture read failed", nil, dayStats{byKind: map[store.Kind]int{}}
	})
	add("home-reloading", func(input *homeViewStateInput, _ *homeReadinessInput) { input.Loading = true })
	add("home-history-off", func(input *homeViewStateInput, _ *homeReadinessInput) { input.KeepHistory = false })
	add("home-history-off-empty", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.KeepHistory, input.Recent, input.Today, input.ShowTry = false, nil, dayStats{byKind: map[store.Kind]int{}}, true
	})
	add("home-hidden-features", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.Features.Hidden = []string{settings.HideTranslate, settings.HideRewrite}
	})
	add("home-recent-no-audio", func(input *homeViewStateInput, _ *homeReadinessInput) {
		for index := range input.Recent {
			input.Recent[index].Audio = ""
		}
	})
	for _, stage := range []session.State{session.Recognizing, session.Composing} {
		add("home-recent-retry-"+string(stage), func(input *homeViewStateInput, _ *homeReadinessInput) {
			input.Retry = entryRetry{id: input.Recent[0].ID, stage: stage}
		})
	}

	for _, state := range []struct {
		name  string
		value permission.State
	}{
		{"unknown", permission.Unknown}, {"notdetermined", permission.NotDetermined},
		{"denied", permission.Denied}, {"restricted", permission.Restricted}, {"granted", permission.Granted},
	} {
		add("home-microphone-"+state.name, func(_ *homeViewStateInput, readiness *homeReadinessInput) {
			readiness.Microphone = state.value
		})
	}
	add("home-permissions-unchecked", func(_ *homeViewStateInput, readiness *homeReadinessInput) {
		readiness.PermissionsChecked = false
	})
	add("home-access-missing", func(_ *homeViewStateInput, readiness *homeReadinessInput) {
		readiness.Accessibility = false
	})

	add("home-speech-missing", func(_ *homeViewStateInput, readiness *homeReadinessInput) {
		readiness.SpeechReady, readiness.SpeechModelID, readiness.SpeechModelName = false, "", ""
	})
	add("home-speech-on-disk", func(_ *homeViewStateInput, readiness *homeReadinessInput) {
		readiness.SpeechReady, readiness.SpeechStatus.OnDisk = false, true
	})
	add("home-speech-downloading", func(_ *homeViewStateInput, readiness *homeReadinessInput) {
		readiness.SpeechReady = false
		readiness.SpeechStatus.Downloading, readiness.SpeechStatus.Done, readiness.SpeechStatus.Total = true, 33, 100
	})
	add("home-speech-dictation-auto", func(_ *homeViewStateInput, readiness *homeReadinessInput) {
		readiness.SpeechReady, readiness.SpeechModelID, readiness.SpeechModelName = true, dictation.ModelID, "System dictation"
		readiness.SpeechExplicit = false
	})
	add("home-speech-dictation-explicit", func(_ *homeViewStateInput, readiness *homeReadinessInput) {
		readiness.SpeechReady, readiness.SpeechModelID, readiness.SpeechModelName = true, dictation.ModelID, "System dictation"
		readiness.SpeechExplicit = true
	})
	add("home-llm-missing", func(_ *homeViewStateInput, readiness *homeReadinessInput) {
		readiness.LanguageModelReady, readiness.LanguageModelName = false, ""
	})

	add("home-shortcut-missing", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.Features.Transcribe.Hotkeys = nil
	})
	for _, trigger := range []string{settings.TriggerDouble, settings.TriggerLong} {
		add("home-shortcut-"+trigger, func(input *homeViewStateInput, _ *homeReadinessInput) {
			input.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: "RightCommand", Trigger: trigger}}
		})
	}
	add("home-shortcut-invalid", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: "DefinitelyNotAKey", Trigger: settings.TriggerTap}}
	})
	add("home-shortcut-bare", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: "A", Trigger: settings.TriggerTap}}
	})
	for _, shortcut := range []struct {
		name string
		keys string
	}{
		{"spotlight", "Command+Space"}, {"finder-search", "Option+Command+Space"},
		{"input-source", "Control+Space"}, {"app-switcher", "Command+Tab"},
		{"window-switcher", "Command+`"}, {"editing", "Command+C"},
	} {
		add("home-shortcut-system-"+shortcut.name, func(input *homeViewStateInput, _ *homeReadinessInput) {
			input.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: shortcut.keys, Trigger: settings.TriggerTap}}
		})
	}
	add("home-shortcut-waiting", func(_ *homeViewStateInput, readiness *homeReadinessInput) {
		readiness.HotkeysInstalled, readiness.Accessibility = false, false
	})
	add("home-shortcut-carbon", func(input *homeViewStateInput, readiness *homeReadinessInput) {
		input.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: "Control+Option+A", Trigger: settings.TriggerTap}}
		readiness.HotkeysInstalled = false
	})

	for _, state := range []helper.State{helper.Stopped, helper.Starting, helper.Ready, helper.Failed} {
		add("home-helper-"+string(state), func(_ *homeViewStateInput, readiness *homeReadinessInput) {
			readiness.Helper = helper.Status{State: state}
		})
	}
	for _, reason := range []helper.FailReason{
		helper.FailMissing, helper.FailStart, helper.FailProtocol, helper.FailHandshake,
		helper.FailCrashed, helper.FailStuck, helper.FailUpdated,
	} {
		add("home-helper-reason-"+string(reason), func(_ *homeViewStateInput, readiness *homeReadinessInput) {
			readiness.Helper = helper.Status{State: helper.Failed, Reason: reason}
		})
	}
	return scenarios
}

func homeFixtureInput() homeViewStateInput {
	return homeViewStateInput{
		Read: true, KeepHistory: true, Features: settings.Defaults().Features,
		Today: dayStats{count: 7, chars: 12345, byKind: map[store.Kind]int{
			store.KindTranscribe: 4, store.KindTranslate: 2, store.KindRewrite: 1,
		}},
		Recent: homeFixtureEntries(), Now: meetingFixtureNow,
	}
}

func homeFixtureReadiness() homeReadinessInput {
	return homeReadinessInput{
		SpeechModelID: "fixture-speech", SpeechModelName: "Fixture speech model", SpeechReady: true,
		SpeechDownloadLeft: "2.5 GB", LanguageModelName: "Fixture language model", LanguageModelReady: true,
		PermissionsChecked: true, Microphone: permission.Granted, Accessibility: true,
		HotkeysInstalled: true, Helper: helper.Status{State: helper.Ready},
	}
}

func homeFixtureEntries() []store.Entry {
	first := historyFixtureBaseEntry()
	first.ID = "home-transcribe"
	first.Result = "  Weekly plan\nI'll send the new agenda tonight.  \n"
	first.RawText = "weekly plan\nI'll send the new agenda tonight"
	translate := first
	translate.ID, translate.Kind, translate.Result = "home-translate", store.KindTranslate, "晚上好"
	translate.RawText, translate.SourceText, translate.TargetLang = "good evening", "selected source", "zh-Hans"
	translate.Audio = ""
	rewrite := first
	rewrite.ID, rewrite.Kind, rewrite.Result = "home-rewrite", store.KindRewrite, "A formal answer"
	rewrite.TypedText, rewrite.SourceText, rewrite.PromptText = "make this formal", "an informal source", "Rewrite formally"
	second := first
	second.ID, second.CreatedAt, second.Result = "home-second", meetingFixtureNow.Add(-2*time.Hour), "Second result"
	second.RawText, second.Audio = second.Result, ""
	older := translate
	older.ID, older.CreatedAt, older.Result = "home-older", meetingFixtureNow.Add(-48*time.Hour), "Older result"
	older.RawText = older.Result
	return []store.Entry{first, translate, rewrite, second, older}
}
