package main

import (
	"errors"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestAboutModels: Settings → About's self-check says whether each kind
// of model has its runtime and a model installed, the diagnostics list
// the models installed and loaded, and the page names no other app.
func TestAboutModels(t *testing.T) {
	a, _, tt := newTestApp(t)
	oldOmni := omniLocate
	t.Cleanup(func() { omniLocate = oldOmni })
	omniLocate = func() (string, error) { return "", errors.New("no runtime") }
	status := func() map[string]string {
		out := map[string]string{}
		for _, c := range a.checks {
			out[c.id] = c.status + " " + c.detail
		}
		return out
	}
	a.runSelfCheck()
	if s := status(); s["speech"] != "fail Its runtime is missing from this copy of Voxt." || !strings.HasPrefix(s["language"], "warn No model set up") {
		t.Errorf("no runtimes: %v", s)
	}

	omniLocate = func() (string, error) { return "/x/qwen3_asr_server", nil }
	a.runSelfCheck()
	if s := status(); !strings.HasPrefix(s["speech"], "warn No model installed") || !strings.HasPrefix(s["language"], "warn No model set up") {
		t.Errorf("no models: %v", s)
	}
	seedModels(t, a, models.ASRFast, models.VADSilero, models.LLMTranslate) // a retired language model: not used
	a.updateSettings(func(s *settings.Settings) { s.Providers.LLM = []settings.Provider{{ID: "ollama"}} })
	a.runSelfCheck()
	if s := status(); s["speech"] != "pass Qwen3-ASR 0.6B (4bit)" || s["language"] != "pass Ollama · qwen3" {
		t.Errorf("installed: %v", s)
	}

	a.modelMem = map[string]int64{models.ASRFast: 743 << 20, models.LLMTranslate: 1300 << 20}
	d := a.diagnosticsText()
	for _, want := range []string{"Models installed: asr.fast, vad.silero, llm.translate", "Models loaded: asr.fast 743 MB, llm.translate 1.3 GB"} {
		if !strings.Contains(d, want) {
			t.Errorf("no %q in the diagnostics:\n%s", want, d)
		}
	}
	a.navigate("/settings/about")
	frames(tt)
	if has(tt, "Auralink") || has(tt, "AudioSmith") {
		t.Errorf("about: %q", firstTexts(tt))
	}
}
