package main

import (
	"slices"
	"testing"

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
