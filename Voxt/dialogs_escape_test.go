package main

import (
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
)

// TestOneEscapeClosesDialogs: with the keyboard focus on a control that
// shows its tooltip (the × Close button, an ⓘ), one Escape closes the
// dialog. The tooltip took the first Escape and the dialog the second.
func TestOneEscapeClosesDialogs(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	for _, d := range []struct {
		name, path string
		open       func(a *app)
		isOpen     func(a *app) bool
	}{
		{"history settings", "/history", func(a *app) { a.histSet.open = true }, func(a *app) bool { return a.histSet.open }},
		{"setup guide", "/", (*app).openOnboarding, func(a *app) bool { return a.ob.open }},
		{"network", modelsPath, (*app).openNetwork, func(a *app) bool { return a.netOpen }},
		{"custom", remotePath, func(a *app) { a.openProviderDialog(local.LLM, llm.Custom) }, func(a *app) bool { return a.provDlg.open }},
		{"oMLX", remotePath, func(a *app) { a.openProviderDialog(local.LLM, "omlx") }, func(a *app) bool { return a.provDlg.open }},
		{"Ollama", remotePath, func(a *app) { a.openProviderDialog(local.LLM, "ollama") }, func(a *app) bool { return a.provDlg.open }},
	} {
		a, _, tt := newTestApp(t)
		tt.SetSize(windowWidth, 1000)
		a.navigate(d.path)
		d.open(a)
		frames(tt)
		if !d.isOpen(a) {
			t.Fatalf("%s: did not open", d.name)
		}
		// Tab until a tooltip adds its text to the window.
		n := len(tt.Texts())
		for i := 0; i < 30 && len(tt.Texts()) <= n; i++ {
			tt.Key(0, ui.KeyTab)
			frames(tt)
		}
		if len(tt.Texts()) <= n {
			t.Fatalf("%s: no control with a tooltip took the focus", d.name)
		}
		tt.Key(0, ui.KeyEscape)
		frames(tt)
		if d.isOpen(a) {
			t.Errorf("%s: the first Escape left it open", d.name)
		}
	}
}
