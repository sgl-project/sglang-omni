package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestHomeFixtureCoverage(t *testing.T) {
	for _, locale := range i18n.Supported() {
		checkHomeFixtureValues(t, locale, readMeetingFixtures(t, locale))
	}
}

func checkHomeFixtureValues(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	for _, name := range []string{
		"home-loading", "home-empty", "home-normal", "home-error", "home-history-off", "home-history-off-empty",
		"home-shortcut-missing", "home-shortcut-double", "home-shortcut-long",
	} {
		if _, ok := fixtures[name]; !ok {
			t.Errorf("%s: missing required home fixture %s", locale, name)
		}
	}
	for name, fixture := range fixtures {
		if fixture.Home != nil && fixture.Home.Activity == nil && name != "home-loading" {
			t.Errorf("%s: %s has no activity calendar", locale, name)
		}
	}
	triggers := map[string]bool{}
	for _, scenario := range meetingFixtureScenarios() {
		if scenario.kind != "home" {
			continue
		}
		if hotkeys := hotkey.HotkeysOf(scenario.home.Features, hotkey.Transcribe); len(hotkeys) > 0 {
			trigger := hotkeys[0].Trigger
			if trigger == "" {
				trigger = settings.TriggerTap
			}
			triggers[trigger] = true
		}
	}
	assertMeetingFixtureSet(t, locale+" home shortcut triggers", meetingNamedConstants(t, "internal/settings/settings.go", "Trigger"), triggers)
}
