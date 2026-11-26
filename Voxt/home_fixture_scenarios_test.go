package main

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/activity"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func homeFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	add := func(name string, mutate func(*homeViewStateInput, *homeReadinessInput)) {
		input, readiness := homeFixtureInput(), homeFixtureReadiness()
		mutate(&input, &readiness)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "home", home: input, homeReadiness: readiness})
	}
	add("home-loading", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.ActivityRead = false
	})
	add("home-empty", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.Activity, input.Usage = activity.Log{}, homeUsage{}
	})
	add("home-normal", func(_ *homeViewStateInput, _ *homeReadinessInput) {})
	add("home-error", func(input *homeViewStateInput, _ *homeReadinessInput) { input.Error = "fixture read failed" })
	add("home-history-off", func(input *homeViewStateInput, _ *homeReadinessInput) { input.KeepHistory = false })
	add("home-history-off-empty", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.KeepHistory = false
		input.Activity, input.Usage = activity.Log{}, homeUsage{}
	})
	add("home-hidden-features", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.Features.Hidden = []string{settings.HideTranslate, settings.HideRewrite}
	})
	// Looking back and Activity, once the activity log was read; folded,
	// Looking back keeps its title row.
	add("home-looking-back", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.Activity, input.ActivityRead = homeFixtureActivity(), true
	})
	add("home-looking-back-closed", func(input *homeViewStateInput, _ *homeReadinessInput) {
		input.Activity, input.ActivityRead, input.InsightsClosed = homeFixtureActivity(), true, true
	})

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
	// Accessibility allowed, the tap not in yet; with Accessibility missing, its step says the shortcut waits.
	// The microphone keeps the steps showing, the shortcut under them.
	add("home-shortcut-waiting", func(_ *homeViewStateInput, readiness *homeReadinessInput) {
		readiness.HotkeysInstalled, readiness.Accessibility = false, true
		readiness.Microphone = permission.NotDetermined
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
	var log activity.Log
	log.Add(meetingFixtureNow, activity.Result{Chars: 120, Spoken: 100, Audio: 30 * time.Second})
	log.Add(meetingFixtureNow.AddDate(0, 0, -10), activity.Result{Chars: 30})
	log.Add(meetingFixtureNow.AddDate(0, -9, 0), activity.Result{Chars: 1})
	return homeViewStateInput{
		KeepHistory: true, Features: settings.Defaults().Features, Now: meetingFixtureNow,
		Activity: log, ActivityRead: true,
		Usage: homeUsage{asrMS: 4000, asrCount: 2, lastModel: "fixture-speech"},
	}
}

// homeFixtureActivity is an activity log of two weeks before the
// fixtures' now, three results a day: 226 characters a minute.
func homeFixtureActivity() activity.Log {
	var log activity.Log
	for _, back := range []int{0, 1, 2, 4, 5, 8, 9, 10, 11, 13} {
		for range 3 {
			log.Add(meetingFixtureNow.AddDate(0, 0, -back), activity.Result{Chars: 74, Spoken: 68, Audio: 18 * time.Second})
		}
	}
	return log
}

func homeFixtureReadiness() homeReadinessInput {
	return homeReadinessInput{
		SpeechModelID: "fixture-speech", SpeechModelName: "Fixture speech model", SpeechReady: true,
		SpeechDownloadLeft: "2.5 GB", LanguageModelName: "Fixture language model", LanguageModelReady: true,
		PermissionsChecked: true, Microphone: permission.Granted, Accessibility: true,
		HotkeysInstalled: true, Helper: helper.Status{State: helper.Ready},
	}
}
