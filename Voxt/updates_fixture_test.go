package main

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func TestUpdatesFixtureCoverage(t *testing.T) {
	// Updates has no status enum. These names cover its display branches and
	// each release-check error branch.
	required := map[string]bool{}
	for _, name := range []string{
		"updates-about-empty", "updates-about-loading", "updates-about-normal", "updates-about-disabled",
		"updates-about-never-checked", "updates-about-automatic-install", "updates-about-autocheck-off",
		"updates-about-check-only", "updates-about-installed", "updates-about-beta", "updates-about-beta-off",
		"updates-release-checking", "updates-release-available", "updates-release-available-no-notes",
		"updates-release-up-to-date", "updates-release-error-generic", "updates-release-error-rate-limit",
		"updates-release-error-timeout",
	} {
		required[name] = true
	}
	for _, locale := range i18n.Supported() {
		fixtures := readMeetingFixtures(t, locale)
		actual := map[string]bool{}
		for name, fixture := range fixtures {
			if !strings.HasPrefix(name, "updates-") {
				continue
			}
			actual[name] = true
			switch fixture.Kind {
			case "updates-settings":
				if fixture.UpdatesSettings == nil || fixture.UpdatesRelease != nil {
					t.Errorf("%s/%s: wrong updates settings state", locale, name)
				}
			case "updates-release":
				if fixture.UpdatesRelease == nil || fixture.UpdatesSettings != nil {
					t.Errorf("%s/%s: wrong updates release state", locale, name)
				}
			default:
				t.Errorf("%s/%s: unexpected kind %q", locale, name, fixture.Kind)
			}
		}
		assertMeetingFixtureSet(t, locale+" updates scenarios", required, actual)
	}
}
