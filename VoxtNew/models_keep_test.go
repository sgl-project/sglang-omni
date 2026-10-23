package main

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/llmrttest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// TestKeepIdle: Automatic keeps each runtime's own idle time; the other
// choices set one for both; "until VoxtNew quits" is never.
func TestKeepIdle(t *testing.T) {
	for _, c := range []struct {
		v                string
		speech, language time.Duration
	}{
		{settings.KeepAuto, 10 * time.Minute, 5 * time.Minute},
		{"unknown", 10 * time.Minute, 5 * time.Minute},
		{settings.Keep5Min, 5 * time.Minute, 5 * time.Minute},
		{settings.Keep15Min, 15 * time.Minute, 15 * time.Minute},
		{settings.Keep30Min, 30 * time.Minute, 30 * time.Minute},
		{settings.Keep1Hour, time.Hour, time.Hour},
		{settings.KeepUntilQuit, -1, -1},
	} {
		s, l := keepIdle(settings.Models{KeepLoaded: c.v})
		if s != c.speech || l != c.language {
			t.Errorf("%q: speech %v, language %v; want %v, %v", c.v, s, l, c.speech, c.language)
		}
	}
	if len(keepLabels()) != len(settings.KeepChoices) {
		t.Errorf("%d labels for %d choices", len(keepLabels()), len(settings.KeepChoices))
	}
}

// TestKeepLoadedRow: the Local models page's Advanced section is folded;
// open, its select shows the choice and saves another; a choice made
// elsewhere shows too.
func TestKeepLoadedRow(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600) // the section under every card
	a.navigate(modelsPath)
	if !hasExact(tt, "Advanced") || has(tt, "Keep models loaded") {
		t.Fatalf("the section is not folded: %q", tt.Texts())
	}
	click(t, tt, "Advanced")
	if !hasExact(tt, "Keep models loaded") || !hasExact(tt, "Automatic") {
		t.Fatalf("open: %q", tt.Texts())
	}
	click(t, tt, "Automatic")
	click(t, tt, "Until VoxtNew quits")
	if got := a.settings.Get().Models.KeepLoaded; got != settings.KeepUntilQuit {
		t.Errorf("saved %q", got)
	}
	if !hasExact(tt, "Until VoxtNew quits") || hasExact(tt, "Automatic") {
		t.Errorf("after the choice: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.Models.KeepLoaded = settings.Keep1Hour })
	if !hasExact(tt, "1 hour") {
		t.Errorf("a choice made elsewhere: %q", tt.Texts())
	}
	click(t, tt, "Advanced")
	if has(tt, "Keep models loaded") {
		t.Error("the section did not fold")
	}
}

// TestKeepLoadedLive: the runtimes start with the saved choice, a change
// reaches the running ones at once, and with "until VoxtNew quits"
// another app's memory request (unloadIdleModel, as the family's
// TopicMemory and Reserve call it) still unloads the idle models, whose
// entries said idle since when.
func TestKeepLoadedLive(t *testing.T) {
	fakeOmni(t, omnitest.Opts{})
	a, _, _ := newTestApp(t)
	t.Setenv(llmrt.DevEnv, "")
	a.updateSettings(func(s *settings.Settings) { s.Models.KeepLoaded = settings.KeepUntilQuit })
	seedModels(t, a, models.ASRFast, models.VADSilero, models.LLMFast)
	self, _ := os.Executable()
	_, idle := keepIdle(a.keepSetting())
	a.llmrt = llmrt.New(llmrt.Config{Bin: self, Env: append(llmrttest.Env(""), omnitest.EnvFake+"="), IdleTimeout: idle,
		OnModel: a.modelEvents(workerLLM), BeforeLoad: a.beforeLoad()})

	if _, _, err := a.local.ASR(context.Background(), models.ASRFast); err != nil { // warms the servers
		t.Fatal(err)
	}
	chat, _, err := a.local.LLM(context.Background(), models.LLMFast)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := chat.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}, Skill: llm.SkillRewrite}); err != nil {
		t.Fatal(err)
	}
	if d := a.omniRT.IdleTimeout(); d != -1 {
		t.Errorf("the speech runtime started with %v", d)
	}
	for _, id := range []string{models.ASRFast, models.VADSilero, models.LLMFast} {
		eventually(t, id+" idle", func() bool { l, ok := a.loadedModel(t, id); return ok && l.IdleSince != nil })
	}

	// A change reaches both runtimes, and back.
	a.updateSettings(func(s *settings.Settings) { s.Models.KeepLoaded = settings.Keep30Min })
	eventually(t, "30 minutes", func() bool {
		return a.omniRT.IdleTimeout() == 30*time.Minute && a.llmrt.IdleTimeout() == 30*time.Minute
	})
	a.updateSettings(func(s *settings.Settings) { s.Models.KeepLoaded = settings.KeepAuto })
	eventually(t, "automatic", func() bool {
		return a.omniRT.IdleTimeout() == omni.DefaultIdleTimeout && a.llmrt.IdleTimeout() == llmrt.DefaultIdleTimeout
	})
	a.updateSettings(func(s *settings.Settings) { s.Models.KeepLoaded = settings.KeepUntilQuit })
	eventually(t, "until VoxtNew quits", func() bool { return a.omniRT.IdleTimeout() == -1 && a.llmrt.IdleTimeout() == -1 })
	// Still loaded, and idle: eventually, as a warm's lease may come
	// after its server's launch said it idle (omni's Warm: the launch
	// reports before the warm leases and lets go).
	for _, id := range []string{models.ASRFast, models.VADSilero, models.LLMFast} {
		eventually(t, id+" idle after the changes", func() bool { l, ok := a.loadedModel(t, id); return ok && l.IdleSince != nil })
	}

	// The family's memory request wins over the setting.
	a.unloadIdleModel(models.ASRFast)
	a.unloadIdleModel(models.LLMFast)
	eventually(t, "the unloads", func() bool {
		_, asr := a.loadedModel(t, models.ASRFast)
		_, lm := a.loadedModel(t, models.LLMFast)
		return !asr && !lm
	})
	if _, ok := a.loadedModel(t, models.VADSilero); !ok {
		t.Error("Silero went too")
	}
	if st := a.llmrt.Status(); st.State != llmrt.Stopped {
		t.Errorf("the language process %+v", st)
	}
}
