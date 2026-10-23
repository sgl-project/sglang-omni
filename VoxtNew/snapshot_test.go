package main

import (
	"context"
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// TestSnapshots renders each page, the settings sections and the shell's
// overlays to PNG with MyGo's renderer, for looking at the layout without
// a Mac screen, light and dark, in English (<name>-light.png) and Chinese
// (<name>-light-zh-Hans.png), at 2×. Set VOXTNEW_SNAPSHOTS to a
// directory:
//
//	VOXTNEW_SNAPSHOTS=/tmp/shots go test -run TestSnapshots .
//
// VOXTNEW_SNAPSHOTS_SIZE=680x440 renders at the window's smallest size
// (snapshotSize).
func TestSnapshots(t *testing.T) { runSnapshots(t, "VOXTNEW_SNAPSHOTS", snapshotsMain) }

func snapshotsMain(t *testing.T, out string) {
	w, h := snapshotSize()
	for _, run := range []struct {
		lang string
		dark bool
	}{{"en", false}, {"en", true}, {"zh-Hans", false}, {"zh-Hans", true}} {
		a, fs, tt := newTestApp(t)
		tt.SetSize(w, h)
		if run.lang != "en" {
			a.updateSettings(func(s *settings.Settings) { s.General.Language = run.lang })
		}
		tt.SetDark(run.dark)
		tt.SetScale(2)
		a.version = "0.1.0"
		// The try sessions here mute nothing: the mute's log lines come
		// from its goroutine, in no fixed order with the others, and the
		// logs panel is a snapshot (mute_link_test.go tests the mute).
		a.mute.Close(time.Second)
		seedHistory(t, a)      // history_page_test.go
		seedWriteHistory(t, a) // history_page_test.go
		suffix := "light"
		if run.dark {
			suffix = "dark"
		}
		if run.lang != "en" {
			suffix += "-" + run.lang
		}
		save := func(name string) {
			t.Helper()
			a.toasts = nil
			frames(tt)
			f, err := os.Create(filepath.Join(out, name+"-"+suffix+".png"))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			if err := png.Encode(f, tt.Image()); err != nil {
				t.Fatal(err)
			}
		}
		// Every page of the sidebar.
		for _, p := range pages {
			a.navigate(p.path)
			name := strings.TrimPrefix(p.path, "/")
			if p.path == "/logs" {
				at := time.Date(2026, 10, 15, 9, 30, 0, 0, time.Local)
				for i, l := range []core.LogEntry{
					{Source: "app", Message: "VoxtNew 0.1.0 started"},
					{Source: helper.LogSource, Message: "fake helper ready"},
					{Source: helper.LogSource, Level: core.SeverityWarning, Message: "AirPods Pro gave no audio within 1.2 s: recording from the default input"},
					{Source: "app", Level: core.SeverityError, Message: "settings: could not save: disk full"},
				} {
					l.Time = at.Add(time.Duration(i) * time.Second)
					if l.Level == "" {
						l.Level = core.SeverityInfo
					}
					a.logs.Append(l)
				}
			}
			save("page-" + name)
		}
		snapshotHistoryDetail(a, save)   // history_page_test.go
		historySettingsShots(a, save)    // history_settings_test.go
		transcribeShots(a, tt, save)     // the Try area's states (feature_page_test.go)
		appEnhanceShots(a, fs, tt, save) // appenhance_page_test.go
		dictShots(t, a, save)            // dictionary_page_test.go
		// The settings dialog, section by section.
		a.updateSettings(func(s *settings.Settings) {
			s.General.OtherLanguages = []string{"zh", "en", "ja"} // the main language's is not shown
			s.Audio.Devices = []string{"AirPodsPro:input", "USB-Mic-1234"}
			s.Audio.DeviceNames = map[string]string{"USB-Mic-1234": "Blue Yeti"}
		})
		for _, s := range settingsSections() {
			a.navigate("/settings/" + s.id)
			if s.id == "storage" {
				frames(tt)    // the section measures the folders when it first builds
				pinStorage(a) // snapshot_env_test.go
			}
			if s.id == "about" {
				a.sys = &sysInfo{OS: "macOS 26.7", Chip: "Apple M2", Memory: 16 << 30, Cores: 8, Arch: "arm64"}
			}
			save("settings-" + s.id)
			if s.id == "about" { // Updates, under the fold (updates.go)
				scrollToText(tt, i18n.T("Updates"))
				save("settings-about-updates")
				// Beta updates, on by default in a beta.
				ver := a.version
				a.version = "0.1.0-beta.1"
				scrollToText(tt, i18n.T("Beta updates"))
				save("settings-about-beta")
				off := false // turned off in a beta: no going back
				a.prefs.Updates.Beta = &off
				frames(tt)
				save("settings-about-beta-off")
				a.prefs.Updates.Beta = nil
				a.version = ver
				// The self-check: both runtimes in the app, no model yet
				// (settings_about_models.go).
				oldOmni, oldLLM := omniLocate, llmLocate
				omniLocate = func() (string, error) { return "VoxtNew.app/Contents/Resources/omni/bin/qwen3_asr_server", nil }
				llmLocate = func() string { return "VoxtNew.app/Contents/Resources/llm/voxtnew-llm" }
				a.runSelfCheck()
				omniLocate, llmLocate = oldOmni, oldLLM
				scrollToText(tt, i18n.T("Self-check"))
				save("settings-about-check")
				a.checks = nil
			}
			if s.id == "audio" { // the sounds and the output's mute, under the fold
				scrollToText(tt, i18n.T("Mute system audio while recording"))
				save("settings-audio-sounds")
				tt.Click(i18n.T("Start sound")) // the sound picker's panel (settings_soundpicker.go)
				frames(tt)
				save("settings-audio-sounds-menu")
				tt.Key(0, ui.KeyEscape)
				frames(tt)
			}
			if s.id == "overlay" { // its preview, under the controls
				w, h := snapshotSize()
				tt.Scroll(float32(w)*0.7, float32(h)*0.6, 0, 600)
				save("settings-overlay-preview")
			}
		}
		a.settingsOpen = false

		// The shell's overlays.
		a.navigate("/home")
		a.openPalette()
		save("palette")
		a.paletteOpen = false
		a.sheetOpen = true
		save("shortcuts")
		a.sheetOpen = false
		a.logsPanel = true
		save("logs-panel")
		a.logsPanel = false

		// The foot of the sidebar (shell_notices.go): no notification,
		// one, three stacked; then the rail's dot and its popover.
		a.clearNotices()
		save("sidebar-notices-0")
		a.problemDismissed = nil
		a.checkModels()
		save("sidebar-notices-1")
		a.notify(core.SeveritySuccess, i18n.T("Imported: %d new, %d updated.", 12, 3), "", "")
		fs.set(helper.Status{State: helper.Failed, Reason: helper.FailCrashed, Err: "exit status 2"})
		save("sidebar-notices-3")
		a.setRail(true)
		save("rail-notices-dot")
		a.noticesOpen = true
		save("rail-notices")
		a.noticesOpen = false
		a.setRail(false)
		fs.set(helper.Status{State: helper.Failed, Reason: helper.FailMissing, Err: "voxtnew-helper was not found"})
		a.navigate("/settings/permissions")
		save("permissions-no-helper")
		a.settingsOpen = false
		a.setRail(true)
		a.navigate("/home")
		save("rail")
		a.setRail(false)

		// An update installed while VoxtNew runs (updates.go): its card,
		// and Settings → About.
		a.clearNotices()
		fs.set(helper.Status{State: helper.Failed, Reason: helper.FailUpdated, Err: "VoxtNew was updated"})
		save("sidebar-notices-update")
		a.navigate("/settings/about")
		scrollToText(tt, i18n.T("Updates"))
		save("settings-about-updated")
		a.settingsOpen = false
		a.bundleUpdated = false
		a.setProblem(problemUpdate, "", nil)
		fs.Start(context.Background())

		// The recording overlay, state by state (overlay_test.go).
		overlayShots(t, a, out, suffix, run.dark)

		// Last, as they set services up: the setup guide, step by step
		// (onboarding_test.go), then Home ready (home_page_test.go).
		modelShots(t, a, tt, save) // models_page_test.go
		onboardingShots(a, tt, save)
		homeShots(a, tt, save)
		familyShots(t, a, tt, save) // family_ui_test.go: other ActNow apps' shortcuts
	}
}

// scrollToText scrolls the page or the settings until s is near the top.
func scrollToText(tt *ui.Tester, s string) {
	frames(tt)
	if r, ok := tt.Find(s); ok {
		w, h := snapshotSize()
		tt.Scroll(float32(w)*0.6, float32(h)*0.5, 0, r.Y-float32(h)*0.2)
	}
}

// snapshotSize is the window's size for snapshots: its default size, or
// its smallest with VOXTNEW_SNAPSHOTS_SIZE=680x440 (or "min").
func snapshotSize() (w, h int) {
	switch os.Getenv("VOXTNEW_SNAPSHOTS_SIZE") {
	case "min", fmt.Sprintf("%dx%d", windowMinWidth, windowMinHeight):
		return windowMinWidth, windowMinHeight
	}
	return windowWidth, windowHeight
}
