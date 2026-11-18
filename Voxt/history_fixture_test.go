package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func TestHistoryFixtureCoverage(t *testing.T) {
	for _, locale := range i18n.Supported() {
		checkHistoryFixtureValues(t, locale, readMeetingFixtures(t, locale))
	}
}

func checkHistoryFixtureValues(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	for _, name := range []string{
		"history-page-loading", "history-page-empty-all", "history-page-normal", "history-page-error",
		"history-settings-normal", "history-settings-loading", "history-settings-error",
		"history-access-off", "history-access-on",
	} {
		if _, ok := fixtures[name]; !ok {
			t.Errorf("%s: missing required history fixture %s", locale, name)
		}
	}
	kinds, statuses, deliveries := map[string]bool{}, map[string]bool{}, map[string]bool{}
	ruleSources, soundProcesses, retryStages := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, scenario := range meetingFixtureScenarios() {
		if scenario.historyProblem != "" {
			scenario.history = historyFixtureProblemInput(scenario.historyProblem)
		}
		for _, entry := range scenario.history.Entries {
			kinds[string(entry.Kind)], statuses[string(entry.Status)], deliveries[entry.Delivery] = true, true, true
			ruleSources[entry.RuleSource], soundProcesses[entry.DSP] = true, true
		}
		if stage := scenario.history.Retry.stage; stage != "" {
			retryStages[string(stage)] = true
		}
	}
	assertMeetingFixtureSet(t, locale+" history kinds", meetingTypedConstants(t, "internal/store/store.go", "Kind"), kinds)
	assertMeetingFixtureSet(t, locale+" history statuses", meetingTypedConstants(t, "internal/store/store.go", "Status"), statuses)
	assertMeetingFixtureSet(t, locale+" history deliveries", map[string]bool{"": true, "insert": true, "copy": true, "card": true}, deliveries)
	assertMeetingFixtureSet(t, locale+" history rule sources", map[string]bool{"": true, "url": true, "browserApp": true, "app": true}, ruleSources)
	assertMeetingFixtureSet(t, locale+" history sound processing", map[string]bool{"": true, "highpass": true, "level": true, "highpass+level": true}, soundProcesses)
	assertMeetingFixtureSet(t, locale+" history retry stages", map[string]bool{"recognizing": true, "composing": true}, retryStages)
	codes := map[string]bool{}
	for _, code := range historyFixtureProblemCodes() {
		codes[code] = true
		if _, ok := fixtures["history-problem-"+code]; !ok {
			t.Errorf("%s: missing fixture for problem code %s", locale, code)
		}
	}
	assertMeetingFixtureSet(t, locale+" history problem scenarios", meetingNamedConstants(t, "internal/session/session.go", "Problem"), codes)
}
