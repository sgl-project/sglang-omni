package main

import (
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// TestSnapshotsShortcuts saves the feature pages' Shortcuts sections (and
// App enhancement's Apps) at the default and the smallest window, Chinese
// dark and English light, and the sections with nothing in them.
func TestSnapshotsShortcuts(t *testing.T) { runSnapshots(t, "VOXTNEW_SNAPSHOTS", snapshotsShortcuts) }

func snapshotsShortcuts(t *testing.T, out string) {
	for _, size := range [][2]int{{windowWidth, windowHeight}, {windowMinWidth, windowMinHeight}} {
		for _, run := range []struct {
			lang string
			dark bool
		}{{"zh-Hans", true}, {"en", false}} {
			a, _, tt := newTestApp(t)
			tt.SetSize(size[0], size[1])
			if run.lang != "en" {
				a.updateSettings(func(s *settings.Settings) { s.General.Language = run.lang })
			}
			tt.SetDark(run.dark)
			tt.SetScale(2)
			a.version = "0.1.0"
			readyModels(a) // no service notices above the section
			suffix := fmt.Sprintf("%dx%d-light", size[0], size[1])
			if run.dark {
				suffix = fmt.Sprintf("%dx%d-dark", size[0], size[1])
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
			// Transcribe with two shortcuts, Translate and Rewrite with their own.
			a.updateSettings(func(s *settings.Settings) {
				s.Features.Transcribe.Hotkeys = append(s.Features.Transcribe.Hotkeys, settings.Hotkey{Keys: "RightCommand", Trigger: settings.TriggerLong})
			})
			for _, p := range []string{"/transcribe", "/translate", "/rewrite"} {
				a.navigate(p)
				save("shortcuts-" + filepath.Base(p))
			}
			// No shortcut.
			a.updateSettings(func(s *settings.Settings) { s.Features.Translate.Hotkeys = nil })
			a.navigate("/translate")
			save("shortcuts-translate-none")

			// App enhancement: a group's apps, then none.
			a.setRules(sampleRules())
			a.navigate("/app-enhancement")
			for _, g := range []struct{ id, name string }{{"g-chat", "ae-apps"}, {"g-code", "ae-apps-none"}} {
				a.aePage.sel = g.id
				frames(tt)
				if r, ok := tt.Find(i18n.T("Apps")); ok {
					tt.Scroll(float32(size[0])*0.6, float32(size[1])*0.5, 0, r.Y-float32(size[1])*0.3)
				}
				save(g.name)
			}
		}
	}
}
