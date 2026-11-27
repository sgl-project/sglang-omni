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
		a.openingStarted = time.Now().Add(-a.openingLength())
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

// With Settings → Appearance's opening off, a start goes straight to the
// main interface, and turning it on again plays it at the next start.
func TestOpeningOff(t *testing.T) {
	directory := testDir(t)
	a, _, tester := newTestAppIn(t, directory)
	if !a.prefs.Appearance.Opening {
		t.Fatal("the opening must be on by default")
	}
	a.updateSettings(func(s *settings.Settings) { s.Setup.Done = true; s.Appearance.Opening = false })
	a.startOpening(false)
	frames(tester)
	if !a.openingStarted.IsZero() {
		t.Fatal("a start played the opening that Settings turned off")
	}
	a.setOpening(true)
	a, _, _ = newTestAppIn(t, directory)
	a.startOpening(false)
	if a.openingStarted.IsZero() {
		t.Fatal("turned on again, the opening did not play at the next start")
	}
}

// While Reduce motion is on, the opening's switch shows off (it rests);
// the saved choice is kept and shows again once Reduce motion is off.
func TestOpeningSwitchFollowsReduceMotion(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.setReduceMotion(true)
	if a.appearanceSettingsModel().Opening || a.settingsForm.opening {
		t.Fatal("the opening's switch shows on while Reduce motion is on")
	}
	if !a.prefs.Appearance.Opening {
		t.Fatal("Reduce motion changed the saved choice")
	}
	a.setReduceMotion(false)
	if !a.appearanceSettingsModel().Opening || !a.settingsForm.opening {
		t.Fatal("the saved choice did not come back after Reduce motion")
	}
}

// The opening plays in the length Settings → Appearance chose (2 seconds
// unless chosen), along the same timeline; a length Voxt does not offer
// falls back to 2. The length's row shows only while the opening plays.
func TestOpeningLength(t *testing.T) {
	a, tt := pageTester(t, "/settings/appearance")
	if a.prefs.Appearance.OpeningMillis != settings.DefaultOpeningMillis || a.openingLength() != 2*time.Second {
		t.Fatalf("the opening lasts %v by default, want 2 s", a.openingLength())
	}
	if !has(tt, "Opening speed") {
		t.Error("no speed row while the opening plays")
	}
	// A click on the rabbit or the turtle goes to its end of the slider.
	for _, end := range []struct {
		label string
		ms    int
	}{{"Faster", 1000}, {"Slower", 3000}} {
		if err := tt.ClickControl(end.label); err != nil {
			t.Fatalf("click %q: %v", end.label, err)
		}
		frames(tt)
		if a.prefs.Appearance.OpeningMillis != end.ms {
			t.Errorf("%s: the opening lasts %d ms", end.label, a.prefs.Appearance.OpeningMillis)
		}
	}
	// Reduce motion folds the switch and the speed away; off again, both
	// come back as they were.
	a.setReduceMotion(true)
	frames(tt)
	if has(tt, "Opening speed") || a.prefs.Appearance.OpeningMillis != 3000 {
		t.Errorf("with Reduce motion on: speed row shown %v, length %d ms", has(tt, "Opening speed"), a.prefs.Appearance.OpeningMillis)
	}
	a.setReduceMotion(false)
	frames(tt)
	if !has(tt, "Opening speed") || a.settingsForm.openingSpeed != 0 {
		t.Errorf("Reduce motion off again: speed row shown %v, at stop %v, want the turtle's", has(tt, "Opening speed"), a.settingsForm.openingSpeed)
	}
	a.setOpeningLength(1000)
	a.openingStarted = time.Now().Add(-500 * time.Millisecond)
	if m := a.openingModel(false); m == nil || m.Elapsed < openingTimeline/2-100*time.Millisecond || m.Elapsed > openingTimeline/2+200*time.Millisecond {
		t.Errorf("half way through a 1-second opening: %+v, want half the timeline", m)
	}
	a.openingStarted = time.Now().Add(-time.Second)
	if a.openingModel(false) != nil {
		t.Error("a 1-second opening still plays after a second")
	}
	a.updateSettings(func(s *settings.Settings) { s.Appearance.OpeningMillis = 2500 })
	if a.prefs.Appearance.OpeningMillis != settings.DefaultOpeningMillis {
		t.Errorf("a length of 2.5 s was kept as %d ms", a.prefs.Appearance.OpeningMillis)
	}
	a.setOpening(false)
	frames(tt)
	if has(tt, "Opening speed") {
		t.Error("the length row shows with the opening off")
	}
}
