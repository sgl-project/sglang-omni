package main

import (
	"context"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestKeepIdle: Automatic keeps the speech servers' own idle time; the
// other choices set another; "until VoxtNew quits" is never.
func TestKeepIdle(t *testing.T) {
	for _, c := range []struct {
		v    string
		idle time.Duration
	}{
		{settings.KeepAuto, 10 * time.Minute},
		{"unknown", 10 * time.Minute},
		{settings.Keep5Min, 5 * time.Minute},
		{settings.Keep15Min, 15 * time.Minute},
		{settings.Keep30Min, 30 * time.Minute},
		{settings.Keep1Hour, time.Hour},
		{settings.KeepUntilQuit, -1},
	} {
		if got := keepIdle(settings.Models{KeepLoaded: c.v}); got != c.idle {
			t.Errorf("%q: %v, want %v", c.v, got, c.idle)
		}
	}
	if len(keepLabels()) != len(settings.KeepChoices) {
		t.Errorf("%d labels for %d choices", len(keepLabels()), len(settings.KeepChoices))
	}
}

// TestKeepLoadedRow: the Speech models page's Advanced section is folded;
// open, its select shows the choice and saves another; a choice made
// elsewhere shows too.
func TestKeepLoadedRow(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 4800) // the section under every card
	a.navigate(modelsPath)
	if !hasExact(tt, "Advanced") || has(tt, "Keep models loaded") {
		t.Fatalf("the section is not folded: %q", tt.Texts())
	}
	click(t, tt, "Advanced")
	if !hasExact(tt, "Keep models loaded") || !hasExact(tt, "Automatic") {
		t.Fatalf("open: %q", tt.Texts())
	}
	click(t, tt, "Automatic")
	click(t, tt, "Until Voxt quits")
	if got := a.settings.Get().Models.KeepLoaded; got != settings.KeepUntilQuit {
		t.Errorf("saved %q", got)
	}
	if !hasExact(tt, "Until Voxt quits") || hasExact(tt, "Automatic") {
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

// TestKeepLoadedLive: the speech servers start with the saved choice, a
// change reaches the running ones at once, and with "until VoxtNew quits"
// another app's memory request (unloadIdleModel, as the family's
// TopicMemory and Reserve call it) still unloads the idle models, whose
// entries said idle since when.
func TestKeepLoadedLive(t *testing.T) {
	fakeOmni(t, omnitest.Opts{})
	events := omniEvents(t)
	a, _, _ := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) { s.Models.KeepLoaded = settings.KeepUntilQuit })
	seedModels(t, a, models.ASRFast, models.VADSilero)

	if _, _, err := a.local.ASR(context.Background(), models.ASRFast); err != nil { // warms the servers
		t.Fatal(err)
	}
	if d := a.omniRT.IdleTimeout(); d != -1 {
		t.Errorf("the speech runtime started with %v", d)
	}
	a.waitIdle(t, events, models.ASRFast, models.VADSilero)

	// A change reaches the servers (at once: the test app applies
	// settings in the caller), and back.
	for _, c := range []struct {
		keep string
		idle time.Duration
	}{
		{settings.Keep30Min, 30 * time.Minute},
		{settings.KeepAuto, omni.DefaultIdleTimeout},
		{settings.KeepUntilQuit, -1},
	} {
		a.updateSettings(func(s *settings.Settings) { s.Models.KeepLoaded = c.keep })
		if got := a.omniRT.IdleTimeout(); got != c.idle {
			t.Errorf("%s: %v, want %v", keepLabel(c.keep), got, c.idle)
		}
	}
	// Still loaded, and idle: at once, or at the end of a warm's lease
	// that came after its server's launch said it idle (omni's Warm: the
	// launch reports before the warm leases and lets go).
	a.waitIdle(t, events, models.ASRFast, models.VADSilero)

	// The family's memory request wins over the setting.
	a.unloadIdleModel(models.ASRFast)
	waitUnloaded(t, events, models.ASRFast)
	if _, ok := a.loadedModel(t, models.ASRFast); ok {
		t.Errorf("%s is still in the file after its unload", models.ASRFast)
	}
	if _, ok := a.loadedModel(t, models.VADSilero); !ok {
		t.Error("Silero went too")
	}
}
