package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// TestSettingsSearchFindsRows: Find in settings finds a row by its name
// even when the section's name does not match, and in Chinese by the
// English name too (#3788: "Dock" found nothing in General).
func TestSettingsSearchFindsRows(t *testing.T) {
	a, _, tt := newTestApp(t)
	t.Cleanup(func() { i18n.SetLocale("en") })
	a.navigate("/settings/general")
	frames(tt)
	for _, c := range []struct{ locale, query, want string }{
		{"en", "Dock", "Show in the Dock"},
		{"en", "clipboard", "Also copy the result to the clipboard"},
		{"zh-Hans", "Dock", "在程序坞中显示"},
		{"zh-Hans", "dock", "在程序坞中显示"},
		{"zh-Hans", "程序坞", "在程序坞中显示"},
		{"zh-Hans", "菜单栏", "在菜单栏显示"},
	} {
		i18n.SetLocale(c.locale)
		a.settingsForm.search = c.query
		if !has(tt, c.want) {
			t.Errorf("%s %q: no %q in %q", c.locale, c.query, c.want, tt.Texts())
		}
	}
	a.settingsForm.search = "Dock"
	if has(tt, "在菜单栏显示") {
		t.Errorf(`zh-Hans "Dock" shows rows that do not match: %q`, tt.Texts())
	}
	// A section with nothing that matches says so, and shows nothing else.
	i18n.SetLocale("en")
	a.navigate("/settings/storage")
	if !has(tt, "Nothing in this section matches your search.") || has(tt, "Used by Voxt") {
		t.Errorf(`storage "Dock": %q`, tt.Texts())
	}
	// The section's name shows it whole.
	a.settingsForm.search = "storage"
	if !has(tt, "Used by Voxt") {
		t.Errorf(`storage "storage": %q`, tt.Texts())
	}
}
