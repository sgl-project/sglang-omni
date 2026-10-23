package main

import (
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/permission"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// TestHomeSnapshots renders Home in its three states (first use and not
// ready, first use and ready, a returning user), at the window's default
// and smallest sizes, Chinese dark and English light:
//
//	VOXTNEW_HOME_SHOTS=/tmp/home go test -run TestHomeSnapshots .
//
// Files: home-<state>-<w>x<h>-<lang>-<theme>.png.
func TestHomeSnapshots(t *testing.T) { runSnapshots(t, "VOXTNEW_HOME_SHOTS", snapshotsHome) }

func snapshotsHome(t *testing.T, out string) {
	for _, size := range [][2]int{{windowWidth, windowHeight}, {windowMinWidth, windowMinHeight}} {
		for _, run := range []struct {
			lang string
			dark bool
		}{{"zh-Hans", true}, {"en", false}} {
			for _, state := range []string{"first-missing", "first-ready", "returning"} {
				a, _, tt := newTestApp(t)
				tt.SetSize(size[0], size[1])
				tt.SetDark(run.dark)
				tt.SetScale(2)
				a.version = "0.1.0"
				if run.lang != "en" {
					a.updateSettings(func(s *settings.Settings) { s.General.Language = run.lang })
				}
				switch state {
				case "first-missing":
					// A speech model, no language model; Accessibility
					// not allowed yet.
					a.local = &fakeLocal{ASRP: &fakeTranscriptions{}}
					a.perms.mic, a.perms.access, a.perms.checked = permission.Granted, false, true
				case "first-ready":
					readyModels(a)
					allowAll(a)
				case "returning":
					readyModels(a)
					allowAll(a)
					seedHistory(t, a)
					seedWriteHistory(t, a)
					a.markTried()
				}
				a.clearNotices()
				a.navigate("/home")
				a.historyChanged()
				frames(tt)
				theme := "light"
				if run.dark {
					theme = "dark"
				}
				name := fmt.Sprintf("home-%s-%dx%d-%s-%s.png", state, size[0], size[1], run.lang, theme)
				f, err := os.Create(filepath.Join(out, name))
				if err != nil {
					t.Fatal(err)
				}
				if err := png.Encode(f, tt.Image()); err != nil {
					t.Fatal(err)
				}
				f.Close()
			}
		}
	}
}
