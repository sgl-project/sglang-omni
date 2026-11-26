package main

// VOXTNEW_SMOKE_LIBRARY: the library transfer in the packaged app, as two
// Macs do it (scripts/library-e2e.sh). "export" seeds a term and the dark
// theme, then exports through Settings → Storage's action to
// VOXTNEW_SMOKE_LIBRARY_FILE; "import" seeds another term and imports that
// file, which asks for a restart; "check" is that restart (runSmoke
// resolves the data folder as main does) and reads what the library holds.

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/datadir"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const (
	smokeTermA = "Voxt library from A"
	smokeTermB = "Voxt library from B"
)

func smokeLibrary(a *app, mode, file string, pump func(time.Duration, func() bool) bool, say func(string, ...any)) bool {
	ctx := context.Background()
	dict := a.lib.store.Dictionary()
	put := func(term string) bool {
		if err := dict.Put(ctx, store.Term{ID: term, Term: term, Enabled: true, CreatedAt: time.Now()}); err != nil {
			say("FAIL adding %q: %v", term, err)
			return false
		}
		return true
	}
	toasted := func(prefix string) bool {
		return slices.ContainsFunc(a.toasts, func(s string) bool { return strings.HasPrefix(s, prefix) })
	}
	switch mode {
	case "export":
		if !put(smokeTermA) {
			return false
		}
		if err := a.lib.settings.Update(ctx, func(s *settings.Settings) { s.Appearance.Theme = settings.ThemeDark }); err != nil {
			say("FAIL the theme: %v", err)
			return false
		}
		chooseExportFile = func(_ *app, _, _ string, then func(string)) { then(file) }
		a.exportLibrary()
		ok := pump(time.Minute, func() bool { return len(a.toasts) > 0 }) && toasted("Exported")
		say("library export: %v", a.toasts)
		return ok
	case "import":
		if !put(smokeTermB) {
			return false
		}
		restarted := false
		chooseLibraryFile = func(_ *app, then func(string)) { then(file) }
		appRelaunch = func() { restarted = true }
		a.importLibrary()
		pump(time.Minute, func() bool { return restarted || len(a.toasts) > 0 })
		say("library import: restart %t %v", restarted, a.toasts)
		return restarted
	case "check":
		terms, err := dict.List(ctx, "")
		if err != nil {
			say("FAIL the dictionary: %v", err)
			return false
		}
		var names []string
		for _, t := range terms {
			names = append(names, t.Term)
		}
		theme := a.lib.settings.Get().Appearance.Theme
		_, kept := os.Stat(filepath.Join(a.dataDir(), datadir.BeforeImport))
		say("library check: terms %q, theme %s, replaced library kept %t", names, theme, kept == nil)
		return slices.Contains(names, smokeTermA) && !slices.Contains(names, smokeTermB) && theme == settings.ThemeDark && kept == nil
	}
	say("FAIL VOXTNEW_SMOKE_LIBRARY=%q: export, import or check", mode)
	return false
}
