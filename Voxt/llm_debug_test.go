package main

import (
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"

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

// TestLLMDebugPromptGrows: the debug page's prompt field is as high as
// its text, however long, under the page's one scroll bar, as Your rules
// on a feature's page; it was 160 high with a scroll bar of its own
// inside the scrolling page (11-09: a 25 000-character prompt). The page
// says the prompt is the feature's Your rules.
func TestLLMDebugPromptGrows(t *testing.T) {
	srv, _ := mockOpenAI(t, "Hello there.")
	useKeys(t, &memKeys{m: map[string]string{"llm." + llm.Custom: "sk-mock-123"}})
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 9000) // room for the whole field
	a.updateSettings(func(s *settings.Settings) {
		s.Providers.LLM = []settings.Provider{{ID: llm.Custom, Endpoint: srv.URL + "/v1", Model: "mock-model"}}
	})
	a.readUnreadKeys()
	a.navigate(llmDebugPath)
	frames(tt)
	a.llmDbg.prompt = "x"
	frames(tt)
	short, ok := tt.Find("System prompt")
	a.llmDbg.prompt = strings.Repeat("Keep product names in English.\n", 200)
	frames(tt)
	long, _ := tt.Find("System prompt")
	if line := short.H / llmDebugPromptLines; !ok || long.H < 150*line {
		t.Errorf("a 200-line prompt's field is %.0f high: %.0f lines", long.H, long.H/line)
	}
}

// TestLLMDebugButtonsFit: the way to the debug page, at the foot of
// Language models, and the debug page's Open Language models are buttons
// on the left, under the page's caption; they filled the column, their
// labels centred, and did not look clickable (#3644).
func TestLLMDebugButtonsFit(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	for _, c := range []struct{ path, caption, label string }{
		{remotePath, "Voxt sends the language model the recognised or selected text, never your audio.", "Try prompts on a model…"},
		{llmDebugPath, "Try a prompt on your own text. Nothing is saved until you click Use this prompt.", "Open Language models"},
	} {
		a.navigate(c.path)
		frames(tt)
		cap, ok1 := tt.Find(c.caption)
		r, ok2 := tt.Find(c.label)
		if !ok1 || !ok2 || r.X-cap.X > 32 || r.W > cap.W/2 {
			t.Errorf("%s: %q at %+v, the caption at %+v", c.path, c.label, r, cap)
		}
	}
}

// TestLLMDebugCloses: Esc or × leaves the debug page for Language
// models; it had neither, so with no model set up its only way out was
// Open Language models (#3781). Esc in an open menu closes only the menu.
func TestLLMDebugCloses(t *testing.T) {
	srv, _ := mockOpenAI(t, "Hello there.")
	useKeys(t, &memKeys{m: map[string]string{"llm." + llm.Custom: "sk-mock-123"}})
	a, _, tt := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) {
		s.Providers.LLM = []settings.Provider{{ID: llm.Custom, Endpoint: srv.URL + "/v1", Model: "mock-model"}}
	})
	a.readUnreadKeys()
	a.navigate(llmDebugPath)
	rewrites := func() int { return strings.Count("\x00"+strings.Join(tt.Texts(), "\x00")+"\x00", "\x00Rewrite\x00") }
	click(t, tt, "Request")
	if rewrites() != 2 { // the sidebar's and the menu's
		t.Fatalf("menu not open: %q", tt.Texts())
	}
	tt.Key(0, ui.KeyEscape)
	frames(tt)
	if p := a.router.Path(); p != llmDebugPath || rewrites() != 1 {
		t.Fatalf("Esc in the menu: at %q, %q", p, tt.Texts())
	}
	tt.Key(0, ui.KeyEscape)
	frames(tt)
	if p := a.router.Path(); p != remotePath {
		t.Fatalf("Esc: at %q", p)
	}
	a.navigate(llmDebugPath)
	frames(tt)
	click(t, tt, "Close")
	frames(tt)
	if p := a.router.Path(); p != remotePath {
		t.Fatalf("Close: at %q", p)
	}
}
