package main

import (
	"encoding/json"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// trTester is the test app on the Transcribe page, in a window tall
// enough to show all of it.
func trTester(t *testing.T) (*app, *ui.Tester) {
	t.Helper()
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	a.navigate("/transcribe")
	frames(tt)
	return a, tt
}

// readyModels installs a speech model and a language model (fakes).
func readyModels(a *app) {
	a.local = readyLocal()
	a.checkModels() // as installing does: the models' card goes
}

func TestTranscribePageRenders(t *testing.T) {
	a, tt := trTester(t)
	for _, s := range []string{
		"Transcribe", "Shortcuts", "Right ⌥", "Tap", "Double tap", "Long press", "Add a shortcut",
		"Paste the last result", "⌃⌥V",
		"Models", "No speech model is installed: install one to transcribe.", "Open Speech models",
		"Clean-up", "Clean up with a language model", "Basic clean-up", "Structure",
		"Your rules", "Not set",
		"Try it", "Start",
	} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, tt.Texts())
		}
	}
	// No event tap in tests: the Accessibility notice, and Right ⌥ is marked.
	if !has(tt, "Open Permissions") {
		t.Error("no Accessibility notice")
	}
	if _, ok := tt.Find("Works only while Voxt may use Accessibility."); !ok {
		t.Error("Right ⌥ is not marked as needing Accessibility")
	}
	click(t, tt, "Open Permissions")
	if a.shownPath() != "/settings/permissions" {
		t.Errorf("went to %s", a.shownPath())
	}
}

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
	a.chooseKeys(hotkey.Transcribe, 1, &a.fui(hotkey.Transcribe).rows[1], func() map[string]string { _, m := keyOptions("RightCommand", false); return m }())
	frames(tt)
	if hs := saved(); hs[1].Keys != "RightShift" {
		t.Fatalf("preset not saved: %+v", hs)
	}

	// Custom keys: invalid ones show why and are not saved; valid ones are
	// saved in canonical spelling.
	a.fui(hotkey.Transcribe).rows[1].sel = "Custom…"
	a.chooseKeys(hotkey.Transcribe, 1, &a.fui(hotkey.Transcribe).rows[1], nil)
	frames(tt)
	if !a.fui(hotkey.Transcribe).rows[1].custom || a.fui(hotkey.Transcribe).rows[1].text != "RightShift" {
		t.Fatalf("custom row %+v", a.fui(hotkey.Transcribe).rows[1])
	}
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

// TestTranscribeMouseButton: Custom… says that mouse buttons work too, also
// when it refuses Mouse1 (the left button); a typed side button is saved,
// shows by its name in the try card, and goes to the helper as its button
// number.
func TestTranscribeMouseButton(t *testing.T) {
	const hint = "Mouse buttons work too: Mouse3 is the middle button, Mouse4 and up the side buttons, with modifiers or without (Option+Mouse4)."
	a, tt := trTester(t)
	if err := tt.ClickControl("Keys of shortcut 1"); err != nil {
		t.Fatal(err)
	}
	click(t, tt, "Custom…")
	if !has(tt, hint) {
		t.Errorf("no mouse hint in %q", tt.Texts())
	}
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
	if !has(tt, "A key is unknown. Write the modifiers first, then one key: Fn, RightCommand, Control+Option+Space, Fn+F5.") || !has(tt, hint) {
		t.Errorf("Mouse1: no unknown-key warning or no mouse hint in %q", tt.Texts())
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
	if !has(tt, "Press Mouse button 4 or Start and say a sentence or two: the result shows here and is not typed anywhere.") {
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
		"Translate has the same keys and trigger: only one of them works.",
		"The same keys as shortcut 1.",
		"macOS uses these keys for Spotlight.",
		"A key alone would be taken from every app: add ⌃, ⌥, ⇧, ⌘ or fn.",
		"Paste the last result has the same keys and trigger: only one of them works.",
		"Transcribe has the same keys and trigger: only one of them works.",
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

func TestTranscribeServicesAndSwitches(t *testing.T) {
	a, tt := trTester(t)
	readyModels(a)
	a.local.(*fakeLocal).MoreLLM = []local.Model{{Kind: local.LLM, ID: models.LLMFast, Name: "Qwen3.5 2B (4bit)"}}
	frames(tt)
	if has(tt, "No speech model is installed: install one to transcribe.") {
		t.Error("notice with a model installed")
	}
	click(t, tt, "Language model for Transcribe")
	click(t, tt, "Qwen3.5 2B (4bit)")
	if id := a.lib.settings.Get().Features.Transcribe.LLM; id != models.LLMFast {
		t.Errorf("LLM %q", id)
	}
	click(t, tt, "Language model for Transcribe")
	click(t, tt, "Automatic (now: "+fakeLLMModel.Name+")")
	if id := a.lib.settings.Get().Features.Transcribe.LLM; id != "" {
		t.Errorf("LLM %q", id)
	}

	// The switches write back; with Enhance off the layers are greyed.
	clickSwitch(t, tt, "Structure")
	if !a.lib.settings.Get().Features.Transcribe.Structure {
		t.Error("structure not saved")
	}
	clickSwitch(t, tt, "Clean up with a language model")
	if a.lib.settings.Get().Features.Transcribe.Enhance {
		t.Error("enhance not saved")
	}
	clickSwitch(t, tt, "Basic clean-up")
	if !a.lib.settings.Get().Features.Transcribe.Cleanup {
		t.Error("clean-up changed while disabled")
	}
}

// TestAutomaticNamesTheModelInUse: the bug was a model menu reading
// "Automatic: X" above "X", two entries nobody could tell apart. The
// Automatic entry says it uses X now, X is listed once, and the
// Translate page's target language reads the same way.
func TestAutomaticNamesTheModelInUse(t *testing.T) {
	a, tt := trTester(t)
	readyModels(a)
	a.local.(*fakeLocal).MoreLLM = []local.Model{{Kind: local.LLM, ID: models.LLMFast, Name: "Qwen3.5 2B (4bit)"}}
	check := func(path, menu, name string) {
		t.Helper()
		a.navigate(path)
		click(t, tt, menu)
		texts := tt.Texts()
		auto := "Automatic (now: " + name + ")"
		if !slices.Contains(texts, auto) || slices.Contains(texts, "Automatic: "+name) || strings.Count("\x00"+strings.Join(texts, "\x00")+"\x00", "\x00"+name+"\x00") != 1 {
			t.Errorf("%s: want %q and %q once: %q", menu, auto, name, texts)
		}
		tt.Key(0, ui.KeyEscape)
		frames(tt)
	}
	check("/transcribe", "Speech model for Transcribe", fakeASRModel.Name)
	check("/transcribe", "Language model for Transcribe", fakeLLMModel.Name)
	check("/translate", "Target language", "Simplified Chinese")
}

// TestTranslateFallbackModel:Translate's Models section picks a
// fallback model, None by default, among the language models set up; a
// fallback no longer set up reads None.
func TestTranslateFallbackModel(t *testing.T) {
	a, tt := trTester(t)
	readyModels(a)
	a.local.(*fakeLocal).MoreLLM = []local.Model{{Kind: local.LLM, ID: models.LLMFast, Name: "Qwen3.5 2B (4bit)"}}
	a.navigate("/translate")
	frames(tt)
	if !has(tt, "Fallback model") || !has(tt, "None") || !slices.Contains(tt.AccessLabels(), "Fallback language model for Translate") {
		t.Fatalf("no fallback row: %q", tt.AccessLabels())
	}
	click(t, tt, "Fallback language model for Translate")
	click(t, tt, "Qwen3.5 2B (4bit)")
	if id := a.lib.settings.Get().Features.Translate.Fallback; id != models.LLMFast {
		t.Errorf("fallback %q", id)
	}
	click(t, tt, "Fallback language model for Translate")
	click(t, tt, "None")
	if id := a.lib.settings.Get().Features.Translate.Fallback; id != "" {
		t.Errorf("fallback %q", id)
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Translate.Fallback = "gone" })
	frames(tt)
	if !has(tt, "None") {
		t.Errorf("a fallback not set up: %q", tt.Texts())
	}
}

// TestTranscribeRules: Your rules start folded, saying how long they are
// (long rules pushed the rest of the page down); open, the field saves
// what is typed, the preview fills its variables, Reset clears it, a
// change elsewhere shows, and long rules scroll in a field at most 320
// high.
func TestTranscribeRules(t *testing.T) {
	a, tt := trTester(t)
	if has(tt, "{{USER_MAIN_LANGUAGE}}") || has(tt, "Show the full prompt") {
		t.Fatal("Your rules are open at first")
	}
	click(t, tt, "Your rules")
	click(t, tt, "Your rules for Transcribe")
	tt.Type("Write {{USER_MAIN_LANGUAGE}} numbers as digits.")
	frames(tt)
	if !a.fui(hotkey.Transcribe).promptDirty {
		t.Fatal("not dirty after typing")
	}
	// Leaving the field saves it.
	click(t, tt, "Show the full prompt")
	if p := a.lib.settings.Get().Features.Transcribe.Prompt; p != "Write {{USER_MAIN_LANGUAGE}} numbers as digits." {
		t.Fatalf("prompt %q", p)
	}
	// The preview has the rules, with the variable filled in.
	if !a.fui(hotkey.Transcribe).preview || !strings.Contains(strings.Join(tt.Texts(), "\n"), "Write English numbers as digits.") {
		t.Errorf("preview: %q", tt.Texts())
	}
	click(t, tt, "Reset your rules")
	if p := a.lib.settings.Get().Features.Transcribe.Prompt; p != "" || a.fui(hotkey.Transcribe).prompt != "" {
		t.Fatalf("reset: %q %q", p, a.fui(hotkey.Transcribe).prompt)
	}
	// A change elsewhere shows.
	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.Prompt = "Be brief." })
	frames(tt)
	if a.fui(hotkey.Transcribe).prompt != "Be brief." || !has(tt, "9 characters") {
		t.Errorf("not set from the settings: %q", a.fui(hotkey.Transcribe).prompt)
	}
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Transcribe.Prompt = strings.Repeat("Keep every product name as it is written.\n", 400)
	})
	frames(tt)
	if r, ok := tt.Find("Your rules for Transcribe"); !ok || r.H > 330 {
		t.Errorf("the rules' field: found %v, %.0f high", ok, r.H)
	}
}

// clickSwitch clicks the switch of the setting row titled title: at
// the row's right end (the switch has the title as its label too, and a
// click on a label finds the title's text first).
func clickSwitch(t *testing.T, tt *ui.Tester, title string) {
	t.Helper()
	frames(tt)
	r, ok := tt.Find(title)
	if !ok {
		t.Fatalf("no %q", title)
	}
	tt.ClickAt(float32(windowWidth-24-18), r.Y+r.H/2)
	frames(tt)
}

// trySnapshot is a Try session's snapshot in state st.
func trySnapshot(a *app, st session.State) session.Snapshot {
	now := time.Now()
	s := session.Snapshot{Rev: a.sess.Rev + 1, ID: "try-1", Kind: store.KindTranscribe, State: st, Try: true,
		StartedAt: now.Add(-4 * time.Second), HeardAt: now.Add(-3 * time.Second)}
	switch st {
	case session.Recording:
		s.Levels = []float64{0.1, 0.3, 0.6, 0.9, 0.5, 0.2, 0.4, 0.7, 0.3, 0.1}
		s.Partial = "so the meeting moves to Thursday"
	case session.Done:
		s.StoppedAt = now
		s.Raw = "um so the meeting moves to Thursday at three, no, at four"
		s.Result = "The meeting moves to Thursday at four."
		s.ASR = session.Model{Name: "Doubao (Volcengine)"}
		s.LLM = session.Model{Name: "DeepSeek"}
		s.Timings = session.Timings{Audio: 3200 * time.Millisecond, ASR: 800 * time.Millisecond, LLM: 1100 * time.Millisecond, Total: 5300 * time.Millisecond}
	case session.Failed:
		s.Problem = &session.Problem{Code: session.ProblemASR, Err: &asr.Error{Kind: asr.KindServer, Provider: "omni-asr"}}
	}
	return s
}

func TestTranscribeTry(t *testing.T) {
	a, tt := trTester(t)
	// Nothing set up: Start says what is missing, on the page.
	click(t, tt, "Start a try")
	if a.sess.State.Live() || !has(tt, "No speech model is installed.") {
		t.Fatalf("blocked try: %+v %q", a.sess, tt.Texts())
	}
	click(t, tt, "Open Speech models")
	if a.shownPath() != modelsPath {
		t.Fatalf("went to %s", a.shownPath())
	}
	a.navigate("/transcribe")

	a.sessionChanged(trySnapshot(a, session.Recording))
	for _, s := range []string{"so the meeting moves to Thursday", "Stop", "Cancel", "0:03"} {
		if !has(tt, s) {
			t.Errorf("recording: no %q in %q", s, tt.Texts())
		}
	}
	if _, ok := tt.Find("Input level"); !ok {
		t.Error("no level bars")
	}
	if a.ov.shown {
		t.Error("the overlay shows a try")
	}
	s := trySnapshot(a, session.Composing)
	a.sessionChanged(s)
	if !has(tt, "Cleaning up…") {
		t.Errorf("composing: %q", tt.Texts())
	}
	a.sessionChanged(trySnapshot(a, session.Done))
	for _, s := range []string{"The meeting moves to Thursday at four.", "As recognised", "um so the meeting moves to Thursday at three, no, at four",
		"Recording 3.2 s · Recognition 0.8 s · Clean-up 1.1 s · Total 5.3 s · Doubao (Volcengine), DeepSeek", "Try again"} {
		if !has(tt, s) {
			t.Errorf("done: no %q in %q", s, tt.Texts())
		}
	}
	click(t, tt, "Copy the result")
	if tt.Clipboard() != "The meeting moves to Thursday at four." {
		t.Errorf("clipboard %q", tt.Clipboard())
	}
	a.sessionChanged(trySnapshot(a, session.Failed))
	if !has(tt, "Speech recognition failed: The speech model stopped or did not answer. Try again in a moment.") {
		t.Errorf("failed: %q", tt.Texts())
	}
	// A session that is not a try does not show here.
	n := trySnapshot(a, session.Recording)
	n.ID, n.Try = "real", false
	a.sessionChanged(n)
	if has(tt, "so the meeting moves to Thursday") || !has(tt, "A transcription is running.") {
		t.Errorf("a shortcut's session shows: %q", tt.Texts())
	}
}

// transcribeShots saves the Transcribe page's Try area while it records
// and once done, scrolled to the end of the page (TestSnapshots).
func transcribeShots(a *app, tt *ui.Tester, save func(string)) {
	a.navigate("/transcribe")
	for _, st := range []session.State{session.Recording, session.Done} {
		a.sessionChanged(trySnapshot(a, st))
		frames(tt)
		w, h := snapshotSize()
		for range 5 {
			tt.Scroll(float32(w)-100, float32(h)/2, 0, 2000)
			frames(tt)
		}
		save("page-transcribe-try-" + string(st))
	}
	a.sessionChanged(session.Snapshot{Rev: a.sess.Rev + 1})
}

// TestTranscribeFullPage renders the whole feature pages (a tall window)
// with a done try and the preview open, for looking at them:
//
//	VOXTNEW_SNAPSHOTS=/tmp/shots go test -run TestTranscribeFullPage .
func TestTranscribeFullPage(t *testing.T) {
	out := os.Getenv("VOXTNEW_SNAPSHOTS")
	if out == "" {
		t.Skip("set VOXTNEW_SNAPSHOTS to a directory")
	}
	os.MkdirAll(out, 0o755)
	w, _ := snapshotSize()
	for _, feature := range []string{hotkey.Transcribe, hotkey.Translate, hotkey.Rewrite} {
		for _, dark := range []bool{false, true} {
			a, tt := trTester(t)
			a.navigate("/" + feature)
			tt.SetSize(w, 2600)
			tt.SetDark(dark)
			readyModels(a)
			if feature == hotkey.Transcribe {
				a.updateSettings(func(s *settings.Settings) {
					s.Features.Transcribe.Hotkeys = append(s.Features.Transcribe.Hotkeys, settings.Hotkey{Keys: "Fn+Shift", Trigger: settings.TriggerDouble}, settings.Hotkey{Keys: "K"})
				})
			}
			frames(tt)
			if feature == hotkey.Transcribe {
				a.fui(feature).rows[1].custom = true
			}
			a.fui(feature).rules, a.fui(feature).preview = true, true
			s := trySnapshot(a, session.Done)
			s.Kind = featureKind(feature)
			switch feature {
			case hotkey.Translate:
				s.Raw, s.Result, s.TargetLanguage = "嗯 会议改到周四下午四点", "The meeting moves to Thursday at 4 pm.", "en"
			case hotkey.Rewrite:
				s.Raw, s.Title, s.Result = "write a short note saying I will be late", "Running late", "Sorry, I'm running about ten minutes late. Start without me."
			}
			a.sessionChanged(s)
			frames(tt)
			name := "page-" + feature + "-full-light.png"
			if dark {
				name = "page-" + feature + "-full-dark.png"
			}
			f, err := os.Create(filepath.Join(out, name))
			if err != nil {
				t.Fatal(err)
			}
			png.Encode(f, tt.Image())
			f.Close()
		}
	}
}

// TestSectionTitleAdd: "Add a shortcut" and App enhancement's "Add apps…"
// sit on their section title's line, on its centre line, with their
// label ending on the card's right edge (its last control + the card's
// padding and border), at the default and the smallest window, in English
// and Chinese.
func TestSectionTitleAdd(t *testing.T) {
	for _, size := range [][2]int{{windowWidth, windowHeight}, {windowMinWidth, windowMinHeight}} {
		for _, lang := range []string{"en", "zh-Hans"} {
			a, _, tt := newTestApp(t)
			tt.SetSize(size[0], size[1])
			if lang != "en" {
				a.updateSettings(func(s *settings.Settings) { s.General.Language = lang })
			}
			check := func(title, add, last string) {
				t.Helper()
				frames(tt)
				tr, ok1 := tt.Find(i18n.T(title))
				br, ok2 := tt.Find(i18n.T(add))
				lr, ok3 := tt.Find(last)
				if !ok1 || !ok2 || !ok3 {
					t.Fatalf("%v %s: %q %v, %q %v, %q %v", size, lang, title, ok1, add, ok2, last, ok3)
				}
				if d := (tr.Y + tr.H/2) - (br.Y + br.H/2); d < -0.5 || d > 0.5 {
					t.Errorf("%v %s: %q is %.1f off %q's centre line", size, lang, add, d, title)
				}
				if tr.H > 20 {
					t.Errorf("%v %s: %q wraps (%.1f high)", size, lang, title, tr.H)
				}
				edge := lr.X + lr.W + float32(sp4) + 1
				if d := br.X + br.W - titleAddHang(&ui.Theme{FontSize: baseFontSize}) - edge; d < -0.5 || d > 0.5 {
					t.Errorf("%v %s: %q ends %.1f from the card's edge", size, lang, add, d)
				}
				if br.X+br.W > float32(size[0]) || br.X < tr.X+tr.W {
					t.Errorf("%v %s: %q at %+v, title %+v", size, lang, add, br, tr)
				}
			}
			for _, p := range []string{"/transcribe", "/translate", "/rewrite"} {
				a.navigate(p)
				check("Shortcuts", "Add a shortcut", i18n.T("Remove shortcut %d", 1))
			}
			a.setRules(sampleRules())
			a.navigate("/app-enhancement")
			a.aePage.sel = "g-chat"
			tt.SetSize(size[0], 1600) // the Apps section, below the prompt
			check("Apps", "Add apps…", i18n.T("Remove %s", "Slack"))
		}
	}
}

// Your rules start folded, so open the section to see its two-stage
// caption: the rules act after recognition, the speech hints before it.
func TestRulesExplainTwoStages(t *testing.T) {
	_, tt := trTester(t)
	click(t, tt, "Your rules")
	frames(tt)
	if !strings.Contains(strings.Join(tt.Texts(), "\n"), "Rules act after recognition, on the text the speech model produced.") {
		t.Errorf("no two-stage caption in %q", tt.Texts())
	}
}
