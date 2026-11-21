package main

import (
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// Resizing never narrows the existing detail or consumes the history list.
func TestHistoryDetailResizeBounds(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	tt.SetSize(1100, 740)
	a.navigate("/history")
	a.histUI.sel = histInserted
	frames(tt)
	detailWidth := func() float32 {
		r, ok := tt.Find("Entry details")
		if !ok {
			t.Fatal("missing details")
		}
		return r.W
	}
	if got := detailWidth(); got != historyPanelW {
		t.Fatalf("initial width %g", got)
	}
	drag := func(delta float32) {
		t.Helper()
		r, ok := tt.Find("Resize entry details")
		if !ok {
			t.Fatal("missing divider")
		}
		x, y := r.X+r.W/2, r.Y+r.H/2
		tt.Advance(time.Second)
		tt.Press(x, y)
		tt.Move(x+delta, y)
		frames(tt)
		tt.Release(x+delta, y)
		frames(tt)
	}
	drag(-500)
	if got := detailWidth(); got != settings.MaxHistoryDetailWidth {
		t.Fatalf("maximum width %g", got)
	}
	if saved := a.settings.Get().Appearance.HistoryDetailWidth; saved != settings.MaxHistoryDetailWidth {
		t.Fatalf("saved width %g", saved)
	}
	a.histUI.panelWidth = 0
	frames(tt)
	if got := detailWidth(); got != settings.MaxHistoryDetailWidth {
		t.Fatalf("forgot saved width %g", got)
	}
	tt.SetSize(800, 740)
	frames(tt)
	if got := detailWidth(); got < historyPanelW || got > 300 {
		t.Fatalf("narrow window width %g", got)
	}
	tt.SetSize(1100, 740)
	frames(tt)
	if got := detailWidth(); got != settings.MaxHistoryDetailWidth {
		t.Fatalf("lost preferred width after resize: %g", got)
	}
	drag(500)
	if got := detailWidth(); got != historyPanelW {
		t.Fatalf("minimum width %g", got)
	}
	if saved := a.settings.Get().Appearance.HistoryDetailWidth; saved != 0 {
		t.Fatalf("default width saved as %g", saved)
	}
	tt.Key(0, ui.KeyLeft)
	frames(tt)
	if got := detailWidth(); got != historyPanelW+10 {
		t.Fatalf("keyboard widening: %g", got)
	}
	r, _ := tt.Find("Resize entry details")
	tt.Advance(time.Second)
	tt.ClickAt(r.X+r.W/2, r.Y+r.H/2)
	tt.ClickAt(r.X+r.W/2, r.Y+r.H/2)
	frames(tt)
	if got := detailWidth(); got != historyPanelW {
		t.Fatalf("double-click reset: %g", got)
	}
	a.setRail(false)
	tt.SetSize(680, 440)
	frames(tt)
	if has(tt, "Resize entry details") {
		t.Fatal("drawer has a resize handle")
	}
	if !has(tt, "Entry details") {
		t.Fatal("drawer disappeared")
	}
}

// Resetting the look or every setting also forgets the resized detail.
func TestResetRestoresHistoryDetailWidth(t *testing.T) {
	for _, reset := range []struct {
		name string
		fn   func(*app)
	}{
		{"resetLook", (*app).resetLook},
		{"resetSettings", (*app).resetSettings},
	} {
		t.Run(reset.name, func(t *testing.T) {
			a, _, tt := newTestApp(t)
			seedHistory(t, a)
			tt.SetSize(1100, 740)
			a.updateSettings(func(s *settings.Settings) { s.Appearance.HistoryDetailWidth = settings.MaxHistoryDetailWidth })
			a.navigate("/history")
			a.histUI.sel = histInserted
			frames(tt)
			detailWidth := func() float32 {
				r, ok := tt.Find("Entry details")
				if !ok {
					t.Fatal("missing details")
				}
				return r.W
			}
			if got := detailWidth(); got != settings.MaxHistoryDetailWidth {
				t.Fatalf("width before reset %g", got)
			}
			reset.fn(a)
			frames(tt)
			if got := detailWidth(); got != historyPanelW {
				t.Fatalf("width after reset %g", got)
			}
			if saved := a.settings.Get().Appearance.HistoryDetailWidth; saved != 0 {
				t.Fatalf("saved width after reset %g", saved)
			}
		})
	}
}

// Saved widths use unscaled DIPs so a larger interface keeps the same limits.
func TestHistoryDetailResizeScale(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	a.updateSettings(func(s *settings.Settings) { s.Appearance.Scale = 1.25 })
	tt.SetSize(1200, 800)
	a.navigate("/history")
	a.histUI.sel = histInserted
	frames(tt)
	r, ok := tt.Find("Entry details")
	if !ok || r.W != historyPanelW*1.25 {
		t.Fatalf("scaled initial width: %+v", r)
	}
	edge, ok := tt.Find("Resize entry details")
	if !ok {
		t.Fatal("missing scaled divider")
	}
	x, y := edge.X+edge.W/2, edge.Y+edge.H/2
	tt.Press(x, y)
	tt.Move(x-500, y)
	tt.Release(x-500, y)
	frames(tt)
	r, _ = tt.Find("Entry details")
	if r.W != settings.MaxHistoryDetailWidth*1.25 || a.settings.Get().Appearance.HistoryDetailWidth != settings.MaxHistoryDetailWidth {
		t.Fatalf("scaled width %g, saved %g", r.W, a.settings.Get().Appearance.HistoryDetailWidth)
	}
}
