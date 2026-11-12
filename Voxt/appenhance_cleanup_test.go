package main

import (
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestAppEnhancementTranscribeCleanup(t *testing.T) {
	for _, lang := range []string{"en", "zh-Hans"} {
		t.Run(lang, func(t *testing.T) {
			dir := testDir(t)
			a, _, tt := newTestAppIn(t, dir)
			tt.SetSize(windowWidth, 1600)
			a.setRules(sampleRules())
			a.updateSettings(func(s *settings.Settings) { s.General.Language = lang })
			a.aePage.sel = "g-chat"
			a.navigate("/app-enhancement")
			frames(tt)
			global := a.settings.Get().Features.Transcribe.Enhance
			defaultLabel := i18n.X("clean-up override", "Use default")
			if !hasExact(tt, defaultLabel) {
				t.Fatalf("existing group does not follow the global setting: %q", tt.Texts())
			}
			choose := func(index int, want appctx.Cleanup) {
				t.Helper()
				clickControl(t, tt, i18n.T("Transcribe clean-up"))
				for _, option := range []string{defaultLabel, i18n.T("On"), i18n.T("Off")} {
					if !hasExact(tt, option) {
						t.Fatalf("missing clean-up choice %q: %q", option, tt.Texts())
					}
				}
				tt.Key(0, ui.KeyHome)
				for range index {
					tt.Key(0, ui.KeyDown)
				}
				tt.Key(0, ui.KeyEnter)
				frames(tt)
				if g, _ := storedRules(t, a).Group("g-chat"); g.TranscribeCleanup != want {
					t.Fatalf("saved clean-up = %q, want %q", g.TranscribeCleanup, want)
				}
				if a.settings.Get().Features.Transcribe.Enhance != global {
					t.Fatal("group override changed global clean-up")
				}
				if g, _ := storedRules(t, a).Group("g-code"); g.TranscribeCleanup != appctx.CleanupDefault {
					t.Fatal("group override changed another group")
				}
			}
			choose(1, appctx.CleanupOn)
			choose(2, appctx.CleanupOff)
			choose(0, appctx.CleanupDefault)
			choose(2, appctx.CleanupOff)

			// Removing Transcribe hides its override without erasing it.
			click(t, tt, i18n.T("Use the group for %s", i18n.T("Transcribe")))
			if hasExact(tt, i18n.T("Transcribe clean-up")) {
				t.Fatal("clean-up shown for a group that excludes Transcribe")
			}
			if g, _ := storedRules(t, a).Group("g-chat"); g.Features.Transcribe || g.TranscribeCleanup != appctx.CleanupOff {
				t.Fatalf("hidden override not retained: %+v", g)
			}
			a.aePage.sel = "g-code"
			if !hasExact(tt, defaultLabel) {
				t.Fatal("another group inherited the previous group's selection")
			}
			a.aePage.sel = "g-chat"
			frames(tt)
			click(t, tt, i18n.T("Use the group for %s", i18n.T("Transcribe")))
			if !hasExact(tt, i18n.T("Off")) {
				t.Fatal("re-enabled Transcribe did not restore its clean-up choice")
			}

			// A new app instance reads the choice from the saved rules document.
			b, _, bt := newTestAppIn(t, dir)
			bt.SetSize(windowWidth, 1600)
			b.aePage.sel = "g-chat"
			b.navigate("/app-enhancement")
			if g, _ := b.aeRules.Group("g-chat"); g.TranscribeCleanup != appctx.CleanupOff || !g.Features.Transcribe {
				t.Fatalf("reloaded group: %+v", g)
			}
			if !hasExact(bt, i18n.T("Off")) {
				t.Fatal("reloaded choice is not shown")
			}
		})
	}
}
