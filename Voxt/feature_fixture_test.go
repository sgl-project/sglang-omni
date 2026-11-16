package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestFeatureFixtureCoverage(t *testing.T) {
	for _, locale := range i18n.Supported() {
		checkFeatureFixtureValues(t, locale, readMeetingFixtures(t, locale))
	}
}

func checkFeatureFixtureValues(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	for _, name := range []string{"feature-empty", "feature-loading", "feature-normal", "feature-error"} {
		if _, ok := fixtures[name]; !ok {
			t.Errorf("%s: missing required feature fixture %s", locale, name)
		}
	}
	features := map[string]bool{}
	states := map[string]bool{}
	problems := map[string]bool{}
	shortcutProblems := map[string]bool{}
	cardSizes := map[int]bool{}
	for name, fixture := range fixtures {
		feature := fixture.Feature
		if feature == nil {
			continue
		}
		features[feature.Feature] = true
		states[feature.Try.State] = true
		if feature.Try.ProblemCode != "" {
			problems[feature.Try.ProblemCode] = true
			if _, ok := fixtures["feature-problem-"+feature.Try.ProblemCode]; !ok {
				t.Errorf("%s: missing fixture for problem code %s", locale, feature.Try.ProblemCode)
			}
		}
		cardSizes[feature.Prompt.Card.Size] = true
		for _, row := range feature.Shortcuts.Rows {
			for _, code := range row.ProblemCodes {
				shortcutProblems[code] = true
			}
		}
		for _, row := range []*featureKeyRowViewState{feature.Shortcuts.Paste, feature.Shortcuts.Cancel} {
			if row == nil {
				continue
			}
			for _, code := range row.ProblemCodes {
				shortcutProblems[code] = true
			}
		}
		if name == "feature-shortcut-problems" && len(shortcutProblems) == 0 {
			t.Errorf("%s: shortcut problem fixture has no codes", locale)
		}
	}
	delete(states, "")
	assertMeetingFixtureSet(t, locale+" feature names", map[string]bool{hotkey.Transcribe: true, hotkey.Translate: true, hotkey.Rewrite: true}, features)
	assertMeetingFixtureSet(t, locale+" feature session states", meetingTypedConstants(t, "internal/session/session.go", "State"), states)
	assertMeetingFixtureSet(t, locale+" feature problem codes", meetingNamedConstants(t, "internal/session/session.go", "Problem"), problems)
	assertMeetingFixtureSet(t, locale+" feature shortcut problem codes", meetingNamedConstants(t, "internal/hotkey/check.go", "Problem"), shortcutProblems)
	for size := range settings.CardSizeSteps {
		if !cardSizes[size] {
			t.Errorf("%s: missing card size %d", locale, size)
		}
	}
}
