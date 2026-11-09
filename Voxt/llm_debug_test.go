package main

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestLLMDebugRuns: the debug page sends Transcribe's request for typed
// text with the prompt as edited there, shows the answer and the
// messages sent, and saves nothing until the prompt is used.
func TestLLMDebugRuns(t *testing.T) {
	srv, requests := mockOpenAI(t, "Hello there.")
	useKeys(t, &memKeys{m: map[string]string{"llm." + llm.Custom: "sk-mock-123"}})
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400) // note (luojiaxuan): tall enough to keep Run on screen where text measures taller (Linux CI)
	a.updateSettings(func(s *settings.Settings) {
		s.Providers.LLM = []settings.Provider{{ID: llm.Custom, Endpoint: srv.URL + "/v1", Model: "mock-model"}}
	})
	a.readUnreadKeys()
	a.navigate(llmDebugPath)
	frames(tt)
	a.llmDbg.prompt = "Fix the punctuation. NO-OTHER-RULES"
	a.llmDbg.input = "um hello there"
	frames(tt)
	click(t, tt, "Run")
	frames(tt)
	got := requests()
	if len(got) != 1 || !strings.Contains(got[0], "NO-OTHER-RULES") || !strings.Contains(got[0], "um hello there") {
		t.Fatalf("requests %q, result %+v", got, a.llmDbg.res)
	}
	if !has(tt, "Hello there.") || !has(tt, "Messages sent") || !has(tt, "system") {
		t.Fatalf("page %q", tt.Texts())
	}
	if a.prefs.Features.Transcribe.Prompt != "" {
		t.Fatal("saved before Use this prompt")
	}
	click(t, tt, "Use this prompt")
	frames(tt)
	if a.prefs.Features.Transcribe.Prompt != "Fix the punctuation. NO-OTHER-RULES" {
		t.Errorf("saved %q", a.prefs.Features.Transcribe.Prompt)
	}
}
