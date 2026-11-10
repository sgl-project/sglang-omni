package main

import "testing"

// TestSettingsSearchFindsRows: Find in settings shows the rows whose
// label holds the words typed, on any section, whatever the section's
// name (#3790), and says so, alone, when no row matches.
func TestSettingsSearchFindsRows(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/general")
	const none = "Nothing in this section matches your search."
	for _, c := range []struct{ query, row string }{
		{"clipboard", "Also copy the result to the clipboard"},
		{"Dock", "Show in the Dock"},
		{"show in the dock", "Show in the Dock"},
		{"menu", "Show in the menu bar"},
	} {
		a.settingsForm.search = c.query
		frames(tt)
		if !has(tt, c.row) || has(tt, none) {
			t.Errorf("search %q: no row %q: %q", c.query, c.row, tt.Texts())
		}
	}
	a.settingsForm.search = "clipboard"
	frames(tt)
	if has(tt, "Show in the Dock") {
		t.Errorf("search clipboard still shows the Dock row")
	}
	a.settingsForm.search = "zzqx"
	frames(tt)
	if !has(tt, none) || has(tt, "Show in the Dock") {
		t.Errorf("search zzqx: %q", tt.Texts())
	}
	// Shortcuts' table is no row: it goes when nothing matches.
	a.navigate("/settings/shortcuts")
	frames(tt)
	if !has(tt, none) || has(tt, "Paste the last result") {
		t.Errorf("Shortcuts, search zzqx: %q", tt.Texts())
	}
}
