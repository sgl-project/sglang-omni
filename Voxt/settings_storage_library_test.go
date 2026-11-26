package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/datadir"
)

// Settings → Storage: Export… writes the library to the chosen file;
// Import… asks first, then stages the chosen file and restarts.
func TestStorageLibraryExportImport(t *testing.T) {
	a, _, tt := newTestApp(t)
	file := filepath.Join(t.TempDir(), "Voxt.voxtlibrary")
	oldSave, oldOpen, oldRelaunch := chooseExportFile, chooseLibraryFile, appRelaunch
	chooseExportFile = func(_ *app, _, _ string, then func(string)) { then(file) }
	chooseLibraryFile = func(_ *app, then func(string)) { then(file) }
	relaunched := false
	appRelaunch = func() { relaunched = true }
	defer func() { chooseExportFile, chooseLibraryFile, appRelaunch = oldSave, oldOpen, oldRelaunch }()
	a.navigate("/settings/storage")
	frames(tt)

	click(t, tt, "Export…")
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("no exported file: %v; toasts %q", err, a.toasts)
	}
	click(t, tt, "Import…")
	frames(tt)
	if relaunched {
		t.Fatal("restarted before the question")
	}
	click(t, tt, "Choose file…")
	if !relaunched {
		t.Fatalf("did not restart to import; toasts %q", a.toasts)
	}
	if _, err := os.Stat(filepath.Join(a.dataDir(), datadir.ImportFile)); err != nil {
		t.Errorf("no import staged: %v", err)
	}
}
