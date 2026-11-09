package main

// Tests of the settings sections: Overlay style, Shortcuts, General's
// other languages, and the dialog's size (Audio: settings_audio_test.go;
// the sidebar's automatic rail: shell_nav_test.go; the menu bar:
// tray_test.go).

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func TestOverlayStyleSettings(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/overlay")
	if !has(tt, "Preview") {
		t.Fatalf("section: %q", firstTexts(tt))
	}
	if _, ok := tt.Find("Preview of the overlay"); !ok {
		t.Error("no preview")
	}
	click(t, tt, "Top centre")
	if o := a.settings.Get().Overlay; o.Position != "top" || o.Opacity != settings.DefaultOpacity || o.Margin != 96 || o.ShowTime {
		t.Fatalf("top: %+v", o)
	}
	// A slider's change goes to the overlay and the settings.
	a.settingsForm.ovStyle.opacity = 0.73
	a.overlayEdited()
	if o := a.settings.Get().Overlay; o.Opacity != 0.73 || o.Position != "top" {
		t.Fatalf("opacity: %+v", o)
	}
	// The pill has no buttons; the time shows when it is chosen.
	if _, ok := tt.Find("Stop"); ok {
		t.Error("the preview has a Stop button")
	}
	if has(tt, "0:07") {
		t.Error("the time shows by default")
	}
	if _, ok := tt.Find("Show the recording time"); !ok {
		t.Error("no time switch")
	}
	a.settingsForm.ovStyle.showTime = true
	a.overlayEdited()
	if o := a.settings.Get().Overlay; !o.ShowTime || !has(tt, "0:07") {
		t.Errorf("show time: %+v", o)
	}
	// A change saved elsewhere reaches the form.
	a.updateSettings(func(s *settings.Settings) { s.Overlay.Radius = 8 })
	if a.settingsForm.ovStyle.radius != 8 {
		t.Errorf("form radius %v", a.settingsForm.ovStyle.radius)
	}
}

func TestShortcutsSettings(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/shortcuts")
	for _, s := range []string{"Transcribe", "Translate", "Rewrite", "Paste the last result", "Right ⌥ ⇧", "⌃⌥V"} {
		if !has(tt, s) {
			t.Errorf("no %q: %q", s, firstTexts(tt))
		}
	}
	// A conflict shows on both rows.
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Translate.Hotkeys = []settings.Hotkey{{Keys: "RightOption", Trigger: settings.TriggerTap}}
	})
	if !has(tt, "Transcribe has the same keys and trigger: only one of them works.") || !has(tt, "Translate has the same keys and trigger: only one of them works.") {
		t.Errorf("conflict: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Rewrite.Hotkeys = nil })
	if !has(tt, "None") {
		t.Error("a feature without shortcut")
	}
	click(t, tt, "Paste the last result ⌃⌥V")
	if a.settingsOpen || a.shownPath() != "/transcribe" {
		t.Fatalf("row click: %v %s", a.settingsOpen, a.shownPath())
	}
	a.navigate("/settings/shortcuts")
	click(t, tt, "Translate Right ⌥")
	if a.shownPath() != "/translate" {
		t.Fatalf("Translate row: %s", a.shownPath())
	}
}

// TestShortcutPreset: a preset gives its keys to the shortcuts the user
// did not change, fn's comes with Auralink's and macOS's warnings, and a
// preset with nothing left to change says so.
func TestShortcutPreset(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/shortcuts")
	const auralink = "Auralink uses fn too: while it runs, both apps react to these keys."
	const globe = "macOS may react to fn too (the Globe key). If it gets in the way, set it to Do Nothing in System Settings → Keyboard."
	if !hasExact(tt, "Right ⌥ (default)") || has(tt, auralink) {
		t.Fatalf("defaults: %q", firstTexts(tt))
	}
	if !slices.Contains(tt.AccessLabels(), "Shortcut preset") {
		t.Errorf("no control named Shortcut preset: %q", tt.AccessLabels())
	}
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Rewrite.Hotkeys = []settings.Hotkey{{Keys: "RightCommand", Trigger: settings.TriggerTap}}
	})
	click(t, tt, "Shortcut preset")
	if hasExact(tt, "Custom") {
		t.Errorf("Custom offered while the shortcuts follow a preset: %q", tt.Texts())
	}
	click(t, tt, "fn (Auralink's keys)")
	f := a.settings.Get().Features
	if f.Transcribe.Hotkeys[0].Keys != "Fn" || f.Translate.Hotkeys[0].Keys != "Fn+Shift" ||
		f.PasteLast.Keys != "Control+Command+V" || f.Rewrite.Hotkeys[0].Keys != "RightCommand" {
		t.Fatalf("fn: %+v", f)
	}
	if !hasExact(tt, "fn (Auralink's keys)") || !has(tt, auralink) || !has(tt, globe) || !has(tt, "fn ⇧") {
		t.Errorf("fn shown: %q", tt.Texts())
	}
	// Every shortcut the user's: Custom, and a preset changes none.
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Transcribe.Hotkeys, s.Features.Translate.Hotkeys = nil, nil
		s.Features.PasteLast = settings.Hotkey{Keys: "F6", Trigger: settings.TriggerTap}
	})
	if !hasExact(tt, "Custom") || has(tt, auralink) {
		t.Fatalf("custom: %q", firstTexts(tt))
	}
	click(t, tt, "Shortcut preset")
	if n := countText(tt, "Custom"); n != 2 { // the select's and its item's
		t.Errorf("Custom shows %d times with the list open: %q", n, tt.Texts())
	}
	click(t, tt, "Right ⌥ (default)")
	if !has(tt, "You changed every shortcut yourself: the preset keeps them.") || a.settings.Get().Features.PasteLast.Keys != "F6" {
		t.Errorf("nothing to change: %q, %+v", tt.Texts(), a.settings.Get().Features.PasteLast)
	}
}

// countText is how many texts on screen are s.
func countText(tt *ui.Tester, s string) int {
	n := 0
	for _, x := range tt.Texts() {
		if x == s {
			n++
		}
	}
	return n
}

func TestOtherLanguages(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/general")
	click(t, tt, "Add a language")
	if err := tt.ChooseMenuItem(uitext.LangLabel("ja")); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if err := tt.Click("Add a language"); err != nil {
		t.Fatal(err)
	}
	if slices.Contains(tt.Menu(), uitext.LangLabel("en")) || slices.Contains(tt.Menu(), uitext.LangLabel("ja")) {
		t.Errorf("the menu offers the main language or a chosen one: %q", tt.Menu())
	}
	if err := tt.ChooseMenuItem(uitext.LangLabel("zh")); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if got := a.settings.Get().General.OtherLanguages; !slices.Equal(got, []string{"ja", "zh"}) {
		t.Fatalf("added: %q", got)
	}
	click(t, tt, "Remove "+uitext.LangLabel("ja"))
	if got := a.settings.Get().General.OtherLanguages; !slices.Equal(got, []string{"zh"}) {
		t.Fatalf("removed: %q", got)
	}
	// Choosing it as the main language takes it off the others.
	a.setMainLanguage("zh") // the select's choice
	if got := a.settings.Get().General.MainLanguage; got != "zh" {
		t.Fatalf("main language %q", got)
	}
	if got := a.settings.Get().General.OtherLanguages; len(got) != 0 {
		t.Errorf("main language still among the others: %q", got)
	}
	if got := withoutLanguage([]string{"zh-Hans", "en", "x-unknown"}, "zh"); !slices.Equal(got, []string{"en", "x-unknown"}) {
		t.Errorf("withoutLanguage: %q", got)
	}
	if got := withoutLanguage([]string{"x-unknown", "en"}, "y-unknown"); !slices.Equal(got, []string{"x-unknown", "en"}) {
		t.Errorf("withoutLanguage of an unknown code: %q", got)
	}
}

// TestSettingsDialogSize: about 720×460 in the default window, a margin
// around it in the smallest, settingsMaxW × settingsMaxH in a large one.
func TestSettingsDialogSize(t *testing.T) {
	th := ui.LightTheme()
	for _, c := range []struct{ w, h, dw, dh float32 }{
		{windowWidth, windowHeight, 720, 457.6},
		{windowMinWidth, windowMinHeight, 612, 387.2},
		{1600, 1000, settingsMaxW, settingsMaxH},
	} {
		dw, dh := settingsDialogSize(th, c.w, c.h)
		if math.Abs(float64(dw-c.dw)) > 0.5 || math.Abs(float64(dh-c.dh)) > 0.5 {
			t.Errorf("%v×%v: %v×%v, want %v×%v", c.w, c.h, dw, dh, c.dw, c.dh)
		}
	}
}

// TestHideFeatures: unchecking Translate in Settings → General takes its
// page out of the sidebar and ⌘K and its shortcuts out of the event tap;
// a hidden App enhancement's scene rules are off; checking them brings
// everything back. Voxt 1.x could hide unused features; Voxt 2
// always showed them (#3644).
func TestHideFeatures(t *testing.T) {
	if settings.HideTranslate != hotkey.Translate || settings.HideRewrite != hotkey.Rewrite {
		t.Fatal("Features.Hidden's ids are not internal/hotkey's")
	}
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	state := func() (sidebar, palette, keys, scenes bool) {
		a.navigate("/home")
		frames(tt)
		sidebar = hasExact(tt, "Translate")
		defer func() {
			a.navigate("/settings/general")
			frames(tt)
			if r, ok := tt.Find("Shown features"); ok { // the section's foot, in view
				tt.Scroll(windowWidth/2, 1200, 0, r.Y-600)
				frames(tt)
			}
		}()
		for _, cmd := range a.paletteItems() {
			palette = palette || cmd.id == "go/translate"
		}
		for _, b := range a.hotkeyPlan().Bindings {
			keys = keys || strings.HasPrefix(b.ID, hotkey.Translate+"/")
		}
		_, scenes = a.sceneConfig().Rules()
		return
	}
	if sidebar, palette, keys, scenes := state(); !sidebar || !palette || !keys || !scenes {
		t.Fatalf("shown: sidebar %v, palette %v, shortcuts %v, scenes %v", sidebar, palette, keys, scenes)
	}
	if err := tt.ClickControl("Show Translate"); err != nil {
		t.Fatal(err)
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Hidden = append(s.Features.Hidden, settings.HideAppEnhancement) })
	if h := a.settings.Get().Features.Hidden; !slices.Equal(h, []string{settings.HideTranslate, settings.HideAppEnhancement}) {
		t.Fatalf("hidden %v", h)
	}
	if sidebar, palette, keys, scenes := state(); sidebar || palette || keys || scenes {
		t.Fatalf("hidden: sidebar %v, palette %v, shortcuts %v, scenes %v", sidebar, palette, keys, scenes)
	}
	if err := tt.ClickControl("Show Translate"); err != nil {
		t.Fatal(err)
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Hidden = nil })
	if sidebar, palette, keys, scenes := state(); !sidebar || !palette || !keys || !scenes || len(a.settings.Get().Features.Hidden) != 0 {
		t.Fatalf("shown again: sidebar %v, palette %v, shortcuts %v, scenes %v", sidebar, palette, keys, scenes)
	}
}
