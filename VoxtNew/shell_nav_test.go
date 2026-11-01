package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

func TestAutoRail(t *testing.T) {
	a, _, tt := newTestApp(t)
	frames(tt)
	if a.railMode() {
		t.Fatal("rail at the default size")
	}
	tt.SetSize(windowMinWidth, windowMinHeight)
	frames(tt)
	if !a.railMode() || !has(tt, "Show the sidebar") {
		t.Fatal("no rail in a narrow window")
	}
	if a.prefs.Appearance.Rail || a.prefs.Appearance.RailChosen {
		t.Error("the automatic rail was saved")
	}
	// The user's choice wins, at any width.
	click(t, tt, "Show the sidebar")
	if a.railMode() || !a.prefs.Appearance.RailChosen {
		t.Fatalf("chose the sidebar: rail %v %+v", a.railMode(), a.prefs.Appearance)
	}
	tt.SetSize(windowWidth, windowHeight)
	frames(tt)
	click(t, tt, "Show icons only")
	if !a.railMode() {
		t.Fatal("chose the rail")
	}
	// Settings → Appearance: back to automatic.
	a.navigate("/settings/appearance")
	click(t, tt, "Automatic")
	if ap := a.settings.Get().Appearance; ap.Rail || ap.RailChosen || a.railMode() {
		t.Fatalf("automatic: %+v", ap)
	}
	click(t, tt, "Icons only")
	if ap := a.settings.Get().Appearance; !ap.Rail || !ap.RailChosen {
		t.Fatalf("icons only: %+v", ap)
	}
	click(t, tt, "Names and icons")
	if ap := a.settings.Get().Appearance; ap.Rail || !ap.RailChosen {
		t.Fatalf("names and icons: %+v", ap)
	}
	// An older document's rail stays a rail.
	if !(&app{shellState: shellState{prefs: settings.Settings{Appearance: settings.Appearance{Rail: true}}}}).railMode() {
		t.Error("older rail")
	}
}

func TestStartHidden(t *testing.T) {
	on := settings.General{MenuBar: true, CloseToTray: true}
	if startHidden(false, on) || !startHidden(true, on) {
		t.Error("menu bar and close to tray")
	}
	if startHidden(true, settings.General{MenuBar: false, CloseToTray: true}) || startHidden(true, settings.General{MenuBar: true}) {
		t.Error("a window that would not come back")
	}
}
