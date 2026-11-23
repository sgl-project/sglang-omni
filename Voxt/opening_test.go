package main

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// Upgrading a completed setup skipped the opening, including its first version with an animation.
func TestOpeningAfterVersionUpdate(t *testing.T) {
	for _, previous := range []settings.Setup{
		{Done: true},
		{Done: true, OpeningShown: true},
		{Done: true, OpeningShown: true, OpeningVersion: "2.1.0"},
	} {
		name := previous.OpeningVersion
		if name == "" {
			name = "legacy"
			if previous.OpeningShown {
				name = "legacy-opening"
			}
		}
		t.Run(name, func(t *testing.T) {
			directory := testDir(t)
			a, _, tester := newTestAppIn(t, directory)
			a.updateSettings(func(preferences *settings.Settings) { preferences.Setup = previous })
			for _, version := range []string{"2.1.1-beta.1", "2.1.1"} {
				a.version = version
				a.startOpening()
				a.maybeOnboard()
				frames(tester)
				if a.openingStarted.IsZero() || a.ob.open {
					t.Fatalf("version %s: an upgraded setup must show the opening without onboarding", version)
				}
				saved := a.lib.settings.Get().Setup
				if saved.OpeningVersion != version || !saved.OpeningShown || !saved.Done {
					t.Fatalf("version %s was not recorded independently of setup: %+v", version, saved)
				}
				a.openingStarted = time.Now().Add(-openingDuration)
				frames(tester)
				if !a.openingStarted.IsZero() || a.ob.open || has(tester, "Get started") {
					t.Fatalf("version %s did not return to the main interface", version)
				}
				restarted, _, nextTester := newTestAppIn(t, directory)
				restarted.version = version
				restarted.startOpening()
				restarted.maybeOnboard()
				frames(nextTester)
				if !restarted.openingStarted.IsZero() || restarted.ob.open {
					t.Fatalf("version %s replayed the opening after restarting", version)
				}
			}
		})
	}
}
