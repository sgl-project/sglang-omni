package main

import (
	"context"
	"errors"
	"net"
	"time"
)

func updatesFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	addSettings := func(name string, mutate func(*updatesViewStateInput)) {
		input := updatesViewStateInput{
			Available: true, Known: true, RunningVersion: "2.0.3", Now: meetingFixtureNow,
			Prefs: updatePrefs{Enabled: true, AutoCheck: true, LastCheck: meetingFixtureNow.Add(-2 * time.Hour)},
		}
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: "updates-about-" + name, kind: "updates-settings", updates: input})
	}
	addSettings("empty", func(input *updatesViewStateInput) { *input = updatesViewStateInput{} })
	addSettings("loading", func(input *updatesViewStateInput) { input.Known = false })
	addSettings("normal", func(*updatesViewStateInput) {})
	addSettings("disabled", func(input *updatesViewStateInput) { input.Prefs.Enabled = false })
	addSettings("never-checked", func(input *updatesViewStateInput) { input.Prefs.LastCheck = time.Time{} })
	addSettings("automatic-install", func(input *updatesViewStateInput) { input.Prefs.AutoInstall = true })
	addSettings("autocheck-off", func(input *updatesViewStateInput) {
		input.Prefs.AutoCheck, input.Prefs.AutoInstall = false, true
	})
	addSettings("check-only", func(input *updatesViewStateInput) { input.Prefs.CheckOnly = true })
	addSettings("installed", func(input *updatesViewStateInput) { input.BundleUpdated = true })
	addSettings("beta", func(input *updatesViewStateInput) {
		input.Beta, input.RunningVersion = true, "2.0.4-beta.1"
	})
	addSettings("beta-off", func(input *updatesViewStateInput) {
		input.Beta, input.RunningVersion = false, "2.0.4-beta.1"
	})

	addRelease := func(name string, mutate func(*releaseViewStateInput)) {
		input := releaseViewStateInput{
			Available: true, RunningVersion: "2.0.3",
			Found: release{Version: "2.0.4", Notes: "## What's new\n\n- Faster.", Download: "https://fixture.invalid/Voxt-2.0.4.dmg"},
		}
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: "updates-release-" + name, kind: "updates-release", release: input})
	}
	addRelease("checking", func(*releaseViewStateInput) {})
	addRelease("available", func(input *releaseViewStateInput) { input.Done, input.Newer = true, true })
	addRelease("available-no-notes", func(input *releaseViewStateInput) {
		input.Done, input.Newer, input.Found.Notes = true, true, ""
	})
	addRelease("up-to-date", func(input *releaseViewStateInput) { input.Done = true })
	for _, scenario := range []struct {
		name string
		err  error
	}{
		{name: "error-generic", err: errors.New("fixture release list failed")},
		{name: "error-rate-limit", err: errRateLimited},
		{name: "error-timeout", err: context.DeadlineExceeded},
		{name: "error-net-timeout", err: &net.DNSError{IsTimeout: true}},
	} {
		addRelease(scenario.name, func(input *releaseViewStateInput) {
			input.Done, input.Err = true, scenario.err
		})
	}
	return scenarios
}
