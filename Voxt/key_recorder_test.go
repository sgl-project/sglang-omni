package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestKeyRecorder: Custom… takes the keys as they are pressed, with the
// app's shortcuts off meanwhile; the keys are saved at once, show as
// macOS writes them, and still do once the page opens again. A key the
// grammar has no name for is refused; Tab moves the focus on.
func TestKeyRecorder(t *testing.T) {
	a, tt := trTester(t)
	saved := func() string { return a.lib.settings.Get().Features.Transcribe.Hotkeys[0].Keys }
	if err := tt.ClickControl("Keys of shortcut 1"); err != nil {
		t.Fatal(err)
	}
	click(t, tt, "Custom…")
	if !tt.Focused("Custom keys of shortcut 1") || !a.keysListening {
		t.Fatalf("the recorder has no focus (%v) or the shortcuts stay on (%v)", tt.Focused("Custom keys of shortcut 1"), a.keysListening)
	}
	if p := a.hotkeyPlan(); len(p.Bindings) == 0 {
		t.Fatal("the settings' plan is empty") // only what is sent is
	}

	tt.Key(ui.Alt, ui.KeySpace)
	frames(tt)
	if saved() != "Option+Space" || !hasExact(tt, "⌥Space") {
		t.Fatalf("pressed ⌥Space: saved %q, shown %q", saved(), tt.Texts())
	}
	tt.Key(ui.Ctrl|ui.Shift|ui.Super, ui.KeyK)
	frames(tt)
	if saved() != "Control+Shift+Command+K" || !hasExact(tt, "⌃⇧⌘K") {
		t.Fatalf("pressed ⌃⇧⌘K: saved %q", saved())
	}

	tt.Key(0, ui.KeyContextMenu)
	frames(tt)
	if saved() != "Control+Shift+Command+K" || !has(tt, "Voxt cannot use that key. Press another, or type the keys instead.") {
		t.Errorf("an unknown key: saved %q, %q", saved(), tt.Texts())
	}
	tt.Key(0, ui.KeyTab)
	frames(tt)
	if saved() != "Control+Shift+Command+K" || tt.Focused("Custom keys of shortcut 1") {
		t.Errorf("Tab was recorded (%q) or kept the focus", saved())
	}
	if a.keysListening {
		t.Error("the shortcuts stay off after the recorder lost the focus")
	}

	a.navigate("/translate")
	a.navigate("/transcribe")
	if !hasExact(tt, "⌃⇧⌘K") {
		t.Errorf("opened again: %q", tt.Texts())
	}
}

// TestKeyRecorderHidden: the shortcuts come back when the page with the
// focused recorder goes away.
func TestKeyRecorderHidden(t *testing.T) {
	a, tt := trTester(t)
	row := &a.fui(hotkey.Transcribe).rows[0]
	row.sel = "Custom…"
	a.chooseKeys(hotkey.Transcribe, 0, row, nil)
	frames(tt)
	if !a.keysListening {
		t.Fatal("not listening")
	}
	a.navigate("/home")
	frames(tt)
	if a.keysListening {
		t.Error("the shortcuts stay off on Home")
	}
}

// TestCancelKeysEsc: Cancel transcription's select offers Esc, by its
// name, and shows it so once chosen.
func TestCancelKeysEsc(t *testing.T) {
	a, tt := trTester(t)
	click(t, tt, "Keys to cancel recording")
	click(t, tt, "Off")
	if k := a.lib.settings.Get().Features.Cancel.Keys; k != "" {
		t.Fatalf("turned off: %q", k)
	}
	opts, byLabel := keyOptions("", true, settings.DefaultCancel)
	if i := slices.Index(opts, "Esc"); i != 1 || byLabel["Esc"] != "Escape" {
		t.Fatalf("options %q", opts)
	}
	click(t, tt, "Keys to cancel recording")
	click(t, tt, "Esc")
	frames(tt)
	if k := a.lib.settings.Get().Features.Cancel.Keys; k != "Escape" || a.fui(hotkey.Transcribe).cancel.sel != "Esc" || !hasExact(tt, "Esc") {
		t.Errorf("chose Esc: saved %q, shown %q", k, a.fui(hotkey.Transcribe).cancel.sel)
	}
}
