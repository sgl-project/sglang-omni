package main

import (
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
)

func TestPromptSaveUsesFullSystem(t *testing.T) {
	a, tt := pageTester(t, "/translate")
	click(t, tt, "Your rules")
	click(t, tt, "System prompt for Translate")
	custom := "CUSTOM PROMPT LINE"
	a.fui(hotkey.Translate).prompt = custom
	a.fui(hotkey.Translate).promptDirty = true
	a.savePrompt(hotkey.Translate)
	click(t, tt, "Example user message")
	if p := a.lib.settings.Get().Features.Translate.Prompt; p != custom {
		t.Fatalf("saved %q", p)
	}
	out, err := a.composedPreview(hotkey.Translate)
	if err != nil {
		t.Fatal(err)
	}
	if out.System != custom {
		t.Fatalf("composed system %q", out.System)
	}
}

func TestPromptResetRestoresBuiltin(t *testing.T) {
	a, tt := pageTester(t, "/translate")
	click(t, tt, "Your rules")
	builtin := a.promptBuiltin(hotkey.Translate)
	if strings.TrimSpace(builtin) == "" {
		t.Fatal("no builtin")
	}
	click(t, tt, "System prompt for Translate")
	a.fui(hotkey.Translate).prompt = "TEMP OVERRIDE"
	a.fui(hotkey.Translate).promptDirty = true
	a.savePrompt(hotkey.Translate)
	click(t, tt, "Reset to default prompt")
	if got := a.lib.settings.Get().Features.Translate.Prompt; got != "" {
		t.Fatalf("settings %q", got)
	}
	if strings.TrimSpace(a.fui(hotkey.Translate).prompt) != strings.TrimSpace(builtin) {
		t.Fatalf("ui %q", a.fui(hotkey.Translate).prompt)
	}
}

// TestRulesFieldGrows: the rules field is as high as its text, wrapped
// at its width, up to promptMaxLines. It was a few lines high with a
// scroll bar of its own, whatever the rules' length.
func TestRulesFieldGrows(t *testing.T) {
	a, tt := pageTester(t, "/transcribe")
	click(t, tt, "Your rules")
	const label = "System prompt for Transcribe"
	u := a.fui(hotkey.Transcribe)
	u.prompt, u.promptDirty = "x", true
	frames(tt)
	short, _ := tt.Find(label)
	u.prompt, u.promptDirty = strings.Repeat("Keep product names in English and write numbers as digits. ", 60), true
	frames(tt)
	long, _ := tt.Find(label)
	if long.H < 4*short.H {
		t.Errorf("a long prompt's field is %.0f high, a one-letter one's %.0f", long.H, short.H)
	}
}

// TestRulesFieldLetsThePageScroll: a scroll that starts on the page goes
// on scrolling the page when the rules field comes under the pointer, as
// on macOS; the field scrolls a long prompt only for a scroll that starts
// over it. The field took the page's scroll over as soon as it reached
// the pointer (MyGo patch 0011).
func TestRulesFieldLetsThePageScroll(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/transcribe")
	frames(tt)
	const label = "System prompt for Transcribe"
	u := a.fui(hotkey.Transcribe)
	u.prompt, u.promptDirty = strings.Repeat("Keep product names in English and write numbers as digits.\n", 200), true
	x, y := float32(windowWidth/2), float32(windowHeight/2)
	tt.Move(x, y)
	if title, ok := tt.Find("Your rules"); ok { // the section at the top, so that it opens in view
		tt.Scroll(x, y, 0, title.Y-60)
		frames(tt)
		tt.Advance(time.Second)
	}
	click(t, tt, "Your rules")
	field, ok := tt.Find(label)
	if !ok {
		t.Fatalf("no rules field: %q", firstTexts(tt))
	}
	tt.Scroll(x, y, 0, field.Y-y-20) // the page: the field's top just under the pointer
	frames(tt)
	before, _ := tt.Find(label)
	for range 4 { // the same gesture, the field under the pointer now
		tt.Scroll(x, y, 0, 60)
	}
	frames(tt)
	after, _ := tt.Find(label)
	if before.Y-after.Y < 200 {
		t.Fatalf("the page stopped when the field came under the pointer: the field from %.0f to %.0f", before.Y, after.Y)
	}
	tt.Advance(time.Second)
	tt.Scroll(x, y, 0, 60) // a new gesture over the field: the field's
	frames(tt)
	if again, _ := tt.Find(label); again.Y != after.Y {
		t.Errorf("a scroll that starts over the field moved the page: the field from %.0f to %.0f", after.Y, again.Y)
	}
}

// TestRulesFind: ⌘F on a feature page opens Your rules with Find
// focused; Find counts the matches, ignoring case, and Return steps
// through them, back to the first after the last.
func TestRulesFind(t *testing.T) {
	a, tt := pageTester(t, "/transcribe")
	u := a.fui(hotkey.Transcribe)
	u.prompt, u.promptDirty = "Use dLLM terms.\nNever expand DLLM.\nplain line", true
	tt.Key(ui.Cmd, ui.KeyF)
	frames(tt)
	if !u.rules || !tt.Focused("Find in the prompt") {
		t.Fatalf("after ⌘F: rules open %v, Find focused %v", u.rules, tt.Focused("Find in the prompt"))
	}
	tt.Type("dllm")
	if !has(tt, "2 matches") {
		t.Fatalf("no count: %q", firstTexts(tt))
	}
	for _, want := range []string{"1 of 2", "2 of 2", "1 of 2"} {
		tt.Key(0, ui.KeyEnter)
		if !has(tt, want) {
			t.Errorf("after Return: no %q (found %d)", want, u.found)
		}
	}
	if got := findFolded(u.prompt, "DLLM"); !slices.Equal(got, [][2]int{{4, 8}, {29, 33}}) {
		t.Errorf("matches %v", got)
	}
}
