package main

import (
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
)

// TestKeyRecorder: clicking a shortcut takes the keys as they are pressed, with the
// app's shortcuts off meanwhile; the keys are saved at once, show as
// macOS writes them, and still do once the page opens again. A key the
// grammar has no name for is refused; Tab moves the focus on.
func TestKeyRecorder(t *testing.T) {
	a, tt := trTester(t)
	saved := func() string { return a.lib.settings.Get().Features.Transcribe.Hotkeys[0].Keys }
	if err := tt.ClickControl("Keys of shortcut 1"); err != nil {
		t.Fatal(err)
	}
	if !tt.Focused("Keys of shortcut 1") || !a.keysListening {
		t.Fatalf("the recorder has no focus (%v) or the shortcuts stay on (%v)", tt.Focused("Keys of shortcut 1"), a.keysListening)
	}
	if p := a.hotkeyPlan(); len(p.Bindings) == 0 {
		t.Fatal("the settings' plan is empty") // only what is sent is
	}

	tt.Key(ui.Alt, ui.KeySpace)
	frames(tt)
	if saved() != "Option+Space" || !hasExact(tt, "⌥Space") {
		t.Fatalf("pressed ⌥Space: saved %q, shown %q", saved(), tt.Texts())
	}
	click(t, tt, "Keys of shortcut 1")
	tt.Key(ui.Ctrl|ui.Shift|ui.Super, ui.KeyK)
	frames(tt)
	if saved() != "Control+Shift+Command+K" || !hasExact(tt, "⌃⇧⌘K") {
		t.Fatalf("pressed ⌃⇧⌘K: saved %q", saved())
	}

	click(t, tt, "Keys of shortcut 1")
	tt.Key(0, ui.KeyContextMenu)
	frames(tt)
	if saved() != "Control+Shift+Command+K" || !has(tt, "Voxt cannot use that key. Press another, or type the keys instead.") {
		t.Errorf("an unknown key: saved %q, %q", saved(), tt.Texts())
	}
	tt.Key(0, ui.KeyTab)
	frames(tt)
	if saved() != "Control+Shift+Command+K" || tt.Focused("Keys of shortcut 1") {
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

// TestKeyRecorderTakesWindowShortcuts: the window's own shortcuts (⌘K
// opens the palette, ⌘, Settings, ⌘1 a page, ⌘F a search) are keys to
// record while a recorder has the focus, and work again once it has not.
func TestKeyRecorderTakesWindowShortcuts(t *testing.T) {
	a, tt := trTester(t)
	saved := func() string { return a.prefs.Features.Transcribe.Hotkeys[0].Keys }
	cmd := "Command"
	if ui.Cmd == ui.Ctrl {
		cmd = "Control"
	}
	for _, k := range []struct {
		key  ui.Key
		name string
	}{{ui.KeyK, "K"}, {ui.KeyComma, ","}, {ui.Key1, "1"}, {ui.KeyF, "F"}} {
		click(t, tt, "Keys of shortcut 1")
		tt.Key(ui.Cmd, k.key)
		frames(tt)
		if want := cmd + "+" + k.name; saved() != want || a.paletteOpen || a.settingsOpen || a.fui("transcribe").rules {
			t.Fatalf("pressed %s: saved %q, palette %v, settings %v, rules %v", want, saved(), a.paletteOpen, a.settingsOpen, a.fui("transcribe").rules)
		}
	}
	a.navigate("/settings/shortcuts")
	scrollToText(tt, "Cancel recording Esc")
	click(t, tt, "Cancel recording Esc")
	tt.Key(ui.Cmd, ui.KeyF)
	frames(tt)
	if k := a.prefs.Features.Cancel.Keys; k != cmd+"+F" || !a.settingsOpen || a.settingsForm.searchFocus {
		t.Fatalf("⌘F in Settings: saved %q, open %v, search focused %v", k, a.settingsOpen, a.settingsForm.searchFocus)
	}
	tt.Key(0, ui.KeyTab)
	frames(tt)
	tt.Key(ui.Cmd, ui.KeyK)
	frames(tt)
	if !a.paletteOpen && !a.settingsOpen {
		t.Error("⌘K did nothing once the recorder lost the focus")
	}
}

// TestKeyRecorderHidden: the shortcuts come back when the page with the
// focused recorder goes away.
func TestKeyRecorderHidden(t *testing.T) {
	a, tt := trTester(t)
	click(t, tt, "Keys of shortcut 1")
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

// TestCancelKeysEsc: Escape can be recorded as the cancel shortcut, also
// in Settings, where it does not close the dialog.
func TestCancelKeysEsc(t *testing.T) {
	a, tt := trTester(t)
	click(t, tt, "Keys to cancel recording")
	tt.Key(ui.Ctrl, ui.KeyK)
	frames(tt)
	if k := a.lib.settings.Get().Features.Cancel.Keys; k != "Control+K" {
		t.Fatalf("recorded %q", k)
	}
	click(t, tt, "Keys to cancel recording")
	tt.Key(0, ui.KeyEscape)
	frames(tt)
	if k := a.lib.settings.Get().Features.Cancel.Keys; k != "Escape" || !hasExact(tt, "Esc") {
		t.Errorf("recorded %q", k)
	}
	a.navigate("/settings/shortcuts")
	scrollToText(tt, "Cancel recording Esc")
	click(t, tt, "Cancel recording Esc")
	tt.Key(0, ui.KeyEscape)
	frames(tt)
	if !a.settingsOpen || a.keysListening || a.prefs.Features.Cancel.Keys != "Escape" {
		t.Fatal("Escape closed Settings instead of being recorded")
	}
}

// TestShortcutCaptureCancel keeps the saved binding and releases keyboard capture.
func TestShortcutCaptureCancel(t *testing.T) {
	a, tt := trTester(t)
	saved := a.prefs.Features.Transcribe.Hotkeys[0].Keys
	if a.keysListening {
		t.Fatal("opening the page started capture")
	}
	click(t, tt, "Keys of shortcut 1")
	tt.Key(0, ui.KeyContextMenu)
	frames(tt)
	click(t, tt, "Cancel")
	if a.keysListening || a.prefs.Features.Transcribe.Hotkeys[0].Keys != saved {
		t.Fatal("Cancel changed the binding or kept capture active")
	}
	click(t, tt, "Keys of shortcut 1")
	click(t, tt, "Type instead")
	click(t, tt, "Keys of shortcut 1")
	tt.Key(ui.Cmd, ui.KeyA)
	tt.Type("Control+Shift+K")
	click(t, tt, "Cancel")
	if a.prefs.Features.Transcribe.Hotkeys[0].Keys != saved {
		t.Fatal("Cancel saved an unsubmitted typed shortcut")
	}
	click(t, tt, "Keys of shortcut 1")
	a.navigate("/home")
	frames(tt)
	a.navigate("/transcribe")
	frames(tt)
	if a.keysListening {
		t.Fatal("returning to the page resumed capture")
	}
}

func TestKeyRecorderUIInputEquivalence(t *testing.T) {
	modifiers := []struct {
		flag ui.Modifiers
		name string
	}{{ui.Ctrl, "Control"}, {ui.Alt, "Option"}, {ui.Shift, "Shift"}, {ui.Super, "Command"}}
	checks := 0
	for key, keyName := range uiKeyNames {
		for bits := 0; bits < 1<<len(modifiers); bits++ {
			event := ui.InputEvent{Key: key}
			var parts []string
			for index, modifier := range modifiers {
				if bits&(1<<index) != 0 {
					event.Mods |= modifier.flag
					parts = append(parts, modifier.name)
				}
			}
			joined := strings.Join(append(parts, keyName), "+")
			chord, err := hotkey.Parse(joined)
			if err != nil {
				t.Fatalf("Parse(%q): %v", joined, err)
			}
			got, ok := pressedKeys(event)
			if !ok || got != chord.String() {
				t.Fatalf("pressedKeys(%+v) = %q, %v; Parse(%q).String() = %q", event, got, ok, joined, chord.String())
			}
			checks++
		}
	}
	for _, event := range []ui.InputEvent{
		{Key: ui.KeyContextMenu, PhysicalKey: ui.KeyK},
		{Key: ui.KeyK, PhysicalKey: ui.KeyUnknown},
	} {
		got, ok := pressedKeys(event)
		if !ok || got != "K" {
			t.Fatalf("physical-key fallback (%+v) = %q, %v", event, got, ok)
		}
		checks++
	}
	if got, ok := pressedKeys(ui.InputEvent{Key: ui.KeyContextMenu, PhysicalKey: ui.KeyUnknown}); ok || got != "" {
		t.Fatalf("unknown key = %q, %v", got, ok)
	}
	checks++
	t.Logf("%d UI-input checks matched hotkey.Parse canonicalization", checks)
}
