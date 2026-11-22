package main

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/activity"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func homeFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	add := func(name string, mutate func(*homeViewStateInput)) {
		input := homeFixtureInput()
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "home", home: input})
	}
	add("home-loading", func(input *homeViewStateInput) { input.ActivityRead = false })
	add("home-empty", func(input *homeViewStateInput) {
		input.Activity, input.Usage = activity.Log{}, homeUsage{}
	})
	add("home-normal", func(_ *homeViewStateInput) {})
	add("home-error", func(input *homeViewStateInput) { input.Error = "fixture read failed" })
	add("home-history-off", func(input *homeViewStateInput) { input.KeepHistory = false })
	add("home-history-off-empty", func(input *homeViewStateInput) {
		input.KeepHistory = false
		input.Activity, input.Usage = activity.Log{}, homeUsage{}
	})
	add("home-shortcut-missing", func(input *homeViewStateInput) {
		input.Features.Transcribe.Hotkeys = nil
	})
	for _, trigger := range []string{settings.TriggerDouble, settings.TriggerLong} {
		add("home-shortcut-"+trigger, func(input *homeViewStateInput) {
			input.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: "RightCommand", Trigger: trigger}}
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
