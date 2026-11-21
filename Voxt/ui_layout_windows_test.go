package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestDictionaryHeaderFitsWithWideSidebar(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowMinWidth, windowMinHeight)
	a.updateSettings(func(s *settings.Settings) {
		s.Appearance.Rail = false
		s.Appearance.RailChosen = true
		s.Appearance.SidebarWidth = settings.MaxSidebarWidth
		s.Appearance.Scale = 1.25
	})
	a.navigate("/dictionary")
	frames(tt)

	for _, label := range []string{"Use the dictionary", "Scan history…", "More actions", "Add term"} {
		r, ok := tt.Find(label)
		if !ok || r.W <= 0 || r.X+r.W > windowMinWidth {
			t.Errorf("%s is outside the window: %+v (found %v)", label, r, ok)
		}
	}
	click(t, tt, "Add term")
	if !a.dictUI.editOpen {
		t.Error("Add term did not open the editor")
	}
}
