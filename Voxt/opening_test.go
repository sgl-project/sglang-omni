package main

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// The opening plays at each start of the app, the same version or not, and
// a completed setup returns to the main interface after it.
func TestOpeningPlaysAtEachStart(t *testing.T) {
	directory := testDir(t)
	for _, version := range []string{"2.1.3", "2.1.3", "2.1.4"} {
		a, _, tester := newTestAppIn(t, directory)
		a.version = version
		a.updateSettings(func(s *settings.Settings) { s.Setup.Done = true })
		a.startOpening(false)
		a.maybeOnboard()
		frames(tester)
		if a.openingStarted.IsZero() || a.ob.open {
			t.Fatalf("version %s: a start must show the opening without the guide", version)
		}
		a.openingStarted = time.Now().Add(-openingDuration)
		frames(tester)
		if !a.openingStarted.IsZero() || a.ob.open || has(tester, "Get started") {
			t.Fatalf("version %s: the opening did not return to the main interface", version)
		}
		a.life = &fakeWindow{} // closed with ⌘W, the app still running
		a.reopenWindow()       // the Dock icon
		frames(tester)
		if !a.openingStarted.IsZero() {
			t.Fatalf("version %s: reopening the window replayed the opening", version)
		}
	}
}

// A window that waits hidden at start (opened at login into the menu bar)
// skips the opening, which would play unseen.
func TestOpeningSkipsHiddenStart(t *testing.T) {
	a, _, tester := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) { s.Setup.Done = true })
	a.startOpening(true)
	frames(tester)
	if !a.openingStarted.IsZero() {
		t.Fatal("a hidden start played the opening")
	}
}
