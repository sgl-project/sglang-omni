package main

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func TestTranscribeHotkeysEdit(t *testing.T) {
	a, tt := trTester(t)
	saved := func() []settings.Hotkey { return a.lib.settings.Get().Features.Transcribe.Hotkeys }
	a.updateSettings(func(s *settings.Settings) { s.Features.Note = settings.Hotkey{} }) // its Right ⌘ (TestTranslatePage)

	// Add one: the first preset no feature uses (Right ⌥, Right ⌥ ⇧, ⌃ Right ⌥ are taken).
	click(t, tt, "Add a shortcut")
	if hs := saved(); len(hs) != 2 || hs[1].Keys != "RightCommand" || hs[1].Trigger != settings.TriggerTap {
		t.Fatalf("added %+v", hs)
	}
	// Choose a preset through the select's state, as the Select does.
	a.fui(hotkey.Transcribe).rows[1].sel = uitext.HotkeyText("RightShift")
	a.chooseFeatureShortcut(hotkey.Transcribe, 1, "RightShift")
	frames(tt)
	if hs := saved(); hs[1].Keys != "RightShift" {
		t.Fatalf("preset not saved: %+v", hs)
	}

	// Custom keys: invalid ones show why and are not saved; valid ones are
	// saved in canonical spelling.
	click(t, tt, "Keys of shortcut 2")
	click(t, tt, "Custom…")
	frames(tt)
	if !a.fui(hotkey.Transcribe).rows[1].custom || a.fui(hotkey.Transcribe).rows[1].text != "RightShift" {
		t.Fatalf("custom row %+v", a.fui(hotkey.Transcribe).rows[1])
	}
	tt.Key(0, ui.KeyTab)
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	click(t, tt, "Custom keys of shortcut 2")
	for range len("RightShift") {
		tt.Key(0, ui.KeyBackspace)
	}
	tt.Type("ctrl+opt+K+J")
	if !has(tt, "Only one key besides the modifiers.") {
		t.Errorf("no parse error: %q", tt.Texts())
	}
	if hs := saved(); hs[1].Keys != "RightShift" {
		t.Errorf("invalid keys saved: %+v", hs)
	}
	tt.Key(0, ui.KeyBackspace)
	tt.Key(0, ui.KeyBackspace)
	frames(tt)
	if hs := saved(); hs[1].Keys != "RightShift" || !has(tt, "Press Return to use ⌃⌥K.") {
		t.Errorf("saved before Return: %+v", hs)
	}
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	if hs := saved(); hs[1].Keys != "Control+Option+K" {
		t.Errorf("custom keys %+v", hs)
	}
	if !a.fui(hotkey.Transcribe).rows[1].custom {
		t.Error("the custom field closed while typing")
	}

	// Remove the first.
	click(t, tt, "Remove shortcut 1")
	if hs := saved(); len(hs) != 1 || hs[0].Keys != "Control+Option+K" {
		t.Fatalf("after remove %+v", hs)
	}
	if len(a.fui(hotkey.Transcribe).rows) != 1 || a.fui(hotkey.Transcribe).rows[0].custom {
		t.Errorf("rows not set again: %+v", a.fui(hotkey.Transcribe).rows)
	}
	// What the helper gets follows (the plan is built from the settings).
	if p := a.hotkeyPlan(); !slices.ContainsFunc(p.Bindings, func(b proto.HotkeyBinding) bool { return b.ID == hotkey.ID(hotkey.Transcribe, 0) }) {
		t.Errorf("plan %+v", p.Bindings)
	}
}

// TestTranscribeMouseButton: Custom… refuses Mouse1 (the left button); a
// typed side button is saved, shows by its name in the try card, and goes
// to the helper as its button number.
func TestTranscribeMouseButton(t *testing.T) {
	a, tt := trTester(t)
	if err := tt.ClickControl("Keys of shortcut 1"); err != nil {
		t.Fatal(err)
	}
	click(t, tt, "Custom…")
	tt.Key(0, ui.KeyTab)
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	click(t, tt, "Custom keys of shortcut 1")
	for range len(settings.DefaultTranscribe) {
		tt.Key(0, ui.KeyBackspace)
	}
	tt.Type("mouse1")
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	if hs := a.lib.settings.Get().Features.Transcribe.Hotkeys; hs[0].Keys != settings.DefaultTranscribe {
		t.Fatalf("Mouse1 saved %+v", hs)
	}
	if !has(tt, "Unknown key. Type the modifiers first, then one key, as in Fn, RightCommand, Control+Option+Space or Fn+F5.") {
		t.Errorf("Mouse1: no unknown-key warning in %q", tt.Texts())
	}
	for range len("mouse1") {
		tt.Key(0, ui.KeyBackspace)
	}
	tt.Type("mouse4")
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	if hs := a.lib.settings.Get().Features.Transcribe.Hotkeys; hs[0].Keys != "Mouse4" {
		t.Fatalf("saved %+v", hs)
	}
	if b, _ := json.Marshal(a.hotkeyPlan().Bindings[0]); string(b) != `{"id":"transcribe/0","modifiers":{},"button":3,"behavior":"tap"}` {
		t.Errorf("sent the helper %s", b)
	}
	if !has(tt, "Press Mouse button 4 or Start, then say a sentence or two. The result shows here and is not typed anywhere.") {
		t.Errorf("the try card does not name the button: %q", tt.Texts())
	}
}

func TestTranscribeTrigger(t *testing.T) {
	a, tt := trTester(t)
	click(t, tt, "Double tap")
	if hs := a.lib.settings.Get().Features.Transcribe.Hotkeys; hs[0].Trigger != settings.TriggerDouble {
		t.Fatalf("trigger %+v", hs)
	}
	click(t, tt, "Long press")
	if hs := a.lib.settings.Get().Features.Transcribe.Hotkeys; hs[0].Trigger != settings.TriggerLong {
		t.Fatalf("trigger %+v", hs)
	}
}

func TestTranscribeHotkeyConflicts(t *testing.T) {
	a, tt := trTester(t)
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Transcribe.Hotkeys = []settings.Hotkey{
			{Keys: "RightOption+Shift", Trigger: settings.TriggerTap}, // Translate's
			{Keys: "RightOption+Shift", Trigger: settings.TriggerDouble},
			{Keys: "Command+Space"},
			{Keys: "K"},
		}
		s.Features.PasteLast = settings.Hotkey{Keys: "Command+Space"}
	})
	for _, s := range []string{
		"Translate has the same keys and trigger, so only one of them works.",
		"The same keys as shortcut 1.",
		"macOS uses these keys for Spotlight.",
		"A single key would be taken from every app. Add ⌃, ⌥, ⇧, ⌘ or fn.",
		"Paste the last result has the same keys and trigger, so only one of them works.",
		"Transcribe has the same keys and trigger, so only one of them works.",
	} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, tt.Texts())
		}
	}
}

func TestTranscribePasteLast(t *testing.T) {
	a, tt := trTester(t)
	click(t, tt, "Keys to paste the last result")
	click(t, tt, "Off")
	if k := a.lib.settings.Get().Features.PasteLast.Keys; k != "" {
		t.Fatalf("paste-last %q", k)
	}
	click(t, tt, "Keys to paste the last result")
	click(t, tt, "Right ⌘")
	if k := a.lib.settings.Get().Features.PasteLast; k.Keys != "RightCommand" || k.Trigger != settings.TriggerTap {
		t.Fatalf("paste-last %+v", k)
	}
}

func TestTranscribeReturnFinishesSetting(t *testing.T) {
	a, tt := trTester(t)
	const title = "Return finishes dictation"
	if !a.lib.settings.Get().Features.Transcribe.ReturnFinishes || !has(tt, title) {
		t.Fatalf("default or setting row: %v, %q", a.lib.settings.Get().Features.Transcribe.ReturnFinishes, firstTexts(tt))
	}
	clickSwitch(t, tt, title)
	if a.lib.settings.Get().Features.Transcribe.ReturnFinishes {
		t.Error("the switch did not turn Return finishing off")
	}
	a.updateSettings(func(s *settings.Settings) { s.General.Language = "zh-Hans" })
	frames(tt)
	if !has(tt, "按回车完成转录") {
		t.Errorf("zh-Hans setting: %q", firstTexts(tt))
	}
}
