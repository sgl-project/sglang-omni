package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestFeaturesMovedToSettings: the feature pages have one home in Settings;
// their old paths, deep links and palette commands open the same sections.
func TestFeaturesMovedToSettings(t *testing.T) {
	a, _, tt := newTestApp(t)
	for _, tc := range []struct {
		feature, control string
	}{
		{"transcribe", "Clean up with a language model"},
		{"translate", "Target language for Translate"},
		{"rewrite", "Clean up the instruction"},
	} {
		t.Run(tc.feature, func(t *testing.T) {
			oldPath := "/" + tc.feature
			newPath := "/settings/" + tc.feature
			if slices.ContainsFunc(pages, func(p navPage) bool { return p.path == oldPath }) {
				t.Errorf("%s is still a page of the sidebar", oldPath)
			}
			for _, open := range []func(){
				func() { a.navigate(oldPath) },
				func() { a.openURL("voxtnew://" + tc.feature) },
				func() { a.runCommandID("go/settings/" + tc.feature) },
			} {
				a.settingsOpen = false
				a.navigate("/home")
				open()
				frames(tt)
				if got := a.shownPath(); got != newPath {
					t.Errorf("shown %s, want %s", got, newPath)
				}
				if a.router.Path() != "/home" {
					t.Errorf("page under the dialog %s", a.router.Path())
				}
				if !has(tt, tc.control) || has(tt, "There is no page at") {
					t.Errorf("section does not show %q: %q", tc.control, firstTexts(tt))
				}
			}
		})
	}
}

func TestFeatureShortcutsSwitchSettingsSections(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/transcribe")
	frames(tt)
	for _, tc := range []struct {
		key  ui.Key
		want string
	}{
		{ui.Key4, "/settings/translate"},
		{ui.Key5, "/settings/rewrite"},
		{ui.Key3, "/settings/transcribe"},
	} {
		tt.Key(ui.Cmd, tc.key)
		frames(tt)
		if got := a.shownPath(); got != tc.want {
			t.Errorf("shortcut went to %s, want %s", got, tc.want)
		}
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Hidden = []string{settings.HideTranslate} })
	frames(tt)
	tt.Key(ui.Cmd, ui.Key4)
	frames(tt)
	if got := a.shownPath(); got != "/settings/transcribe" {
		t.Errorf("shortcut opened a hidden feature at %s", got)
	}
}
