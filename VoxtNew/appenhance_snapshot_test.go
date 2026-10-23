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

// TestSnapshotsAppEnhance saves the App enhancement page at the default
// and the smallest window, Chinese dark and English light: the page (its
// header), its scene prompt and apps, and the Add
// apps and Browsers dialogs, with the fake helper's icons.
func TestSnapshotsAppEnhance(t *testing.T) {
	runSnapshots(t, "VOXTNEW_SNAPSHOTS", snapshotsAppEnhance)
}

func snapshotsAppEnhance(t *testing.T, out string) {
	for _, size := range [][2]int{{windowWidth, windowHeight}, {windowMinWidth, windowMinHeight}} {
		for _, run := range []struct {
			lang string
			dark bool
		}{{"zh-Hans", true}, {"en", false}} {
			a, fs, tt := newTestApp(t)
			tt.SetSize(size[0], size[1])
			if run.lang != "en" {
				a.updateSettings(func(s *settings.Settings) { s.General.Language = run.lang })
			}
			tt.SetDark(run.dark)
			tt.SetScale(2)
			a.version = "0.1.0"
			readyModels(a)
			a.clearNotices()
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
			scrollTo := func(label string) {
				frames(tt)
				if r, ok := tt.Find(i18n.T(label)); ok {
					tt.Scroll(float32(size[0])*0.6, float32(size[1])*0.5, 0, r.Y-float32(size[1])*0.25)
				}
			}
			browserURLs(fs, map[string]string{chromeApp.BundleID: "https://github.com/hehehai/voxtnew/pulls"})
			snapshotIcons(fs, os.Getenv("VOXTNEW_SNAPSHOT_ICONS"))
			a.setRules(sampleRules())
			a.urlReads["com.google.Chrome"] = urlRead{ok: true}
			a.urlReads["com.apple.Safari"] = urlRead{err: "no web area"}
			a.helperNotified(appActivated(slackApp))
			a.aePage.sel = "g-chat"
			a.navigate("/app-enhancement")
			save("ae-page")
			scrollTo("Scene prompt")
			save("ae-scene")
			scrollTo("Apps")
			save("ae-apps")
			a.navigate("/home")
			a.navigate("/app-enhancement")
			a.aePage.appsOpen = true
			a.loadApps()
			save("ae-apps-dialog")
			a.aePage.appsOpen = false
			a.aePage.browsersOpen = true
			save("ae-browsers-dialog")
			a.aePage.browsersOpen = false
		}
	}
}
