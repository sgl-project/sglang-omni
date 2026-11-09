package main

import (
	"slices"
	"strings"
	"testing"

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
