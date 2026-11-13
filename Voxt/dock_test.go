package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestDockSwitch: Settings → General → Show in the Dock leaves Voxt out of
// the Dock (Voxt 1.x's showInDock). Voxt stays reachable: hiding the Dock
// turns the menu bar icon on, and hiding the menu bar icon brings the
// Dock back.
func TestDockSwitch(t *testing.T) {
	var shown []bool
	old := setDockShown
	setDockShown = func(_ *app, on bool) { shown = append(shown, on) }
	t.Cleanup(func() { setDockShown = old })
	a, _, tt := newTestApp(t)
	a.navigate("/settings/general")
	frames(tt)
	if !has(tt, "Show in the Dock") || !a.settingsForm.dock {
		t.Fatalf("no Dock switch, or off: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.General.MenuBar = false })
	frames(tt)
	if err := tt.ClickControl("Show in the Dock"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	g := a.settings.Get().General
	if !g.HideDock || !g.MenuBar || !slices.Equal(shown, []bool{false}) {
		t.Fatalf("Dock off: %+v, shown %v", g, shown)
	}
	if err := tt.ClickControl("Show in the menu bar"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if g = a.settings.Get().General; g.HideDock || g.MenuBar || !slices.Equal(shown, []bool{false, true}) {
		t.Fatalf("menu bar off: %+v, shown %v", g, shown)
	}
}

// TestDockBeforeLaunch: main applies Show in the Dock before the app runs,
// with no window yet, so Voxt launches with the policy it keeps (#3918):
// one that starts in the Dock and turns accessory once launched may keep
// its Dock icon.
func TestDockBeforeLaunch(t *testing.T) {
	var got []mygo.ActivationPolicy
	old := setActivationPolicy
	setActivationPolicy = func(p mygo.ActivationPolicy) { got = append(got, p) }
	t.Cleanup(func() { setActivationPolicy = old })
	a := &app{shellState: newShellState()}
	a.prefs.General.HideDock = true
	a.applyDock()
	a.prefs.General.HideDock = false
	a.applyDock()
	if want := []mygo.ActivationPolicy{mygo.ActivationPolicyAccessory, mygo.ActivationPolicyRegular}; !slices.Equal(got, want) {
		t.Fatalf("policies %v, want %v", got, want)
	}
}
