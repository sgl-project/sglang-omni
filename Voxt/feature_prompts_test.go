package main

import (
	"strings"
	"testing"

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
