package main

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/egoist/mygo"
)

// TestNameApp: the app is Voxt, and its data, logs and cache stay in the
// VoxtNew folders, where testers' libraries are.
func TestNameApp(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("XDG_CACHE_HOME", "")
	paths := []mygo.PathName{mygo.PathUserData, mygo.PathLogs, mygo.PathCache}
	old := mygo.App.Name()
	t.Cleanup(func() {
		for _, p := range paths {
			mygo.App.SetPath(p, "")
		}
		mygo.App.SetName(old)
	})
	if err := nameApp(); err != nil {
		t.Fatal(err)
	}
	if n := mygo.App.Name(); n != "Voxt" {
		t.Errorf("named %q", n)
	}
	for _, p := range paths {
		dir, err := mygo.App.Path(p)
		parts := strings.Split(dir, string(filepath.Separator))
		if err != nil || !slices.Contains(parts, "VoxtNew") || slices.Contains(parts, "Voxt") {
			t.Errorf("%s: %q %v", p, dir, err)
		}
	}
}
