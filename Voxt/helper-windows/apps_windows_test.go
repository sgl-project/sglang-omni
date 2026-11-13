//go:build windows

package main

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

func TestStableAppIDNormalizesPathCase(t *testing.T) {
	first := stableAppID(`C:\Program Files\Example\App.exe`)
	second := stableAppID(`c:\program files\example\APP.EXE`)
	if first == "" || first != second {
		t.Fatalf("stable IDs differ for one executable: %q %q", first, second)
	}
	if other := stableAppID(`C:\Program Files\Example\Other.exe`); other == first {
		t.Fatal("different executables share an ID")
	}
}

func TestMergeWindowsAppsDeduplicatesAndSorts(t *testing.T) {
	running := []proto.AppInfo{{BundleID: "Win32.App", Name: "zeta", Running: true, PID: 42}}
	installed := []proto.AppInfo{
		{BundleID: "win32.app", Name: "ignored", Path: `C:\Apps\App.exe`},
		{BundleID: "win32.other", Name: "Alpha", Path: `C:\Apps\Other.exe`},
	}
	merged := mergeWindowsApps(running, installed)
	if len(merged) != 2 || merged[0].Name != "Alpha" || merged[1].Name != "zeta" {
		t.Fatalf("merged order = %+v", merged)
	}
	if !merged[1].Running || merged[1].PID != 42 || merged[1].Path != `C:\Apps\App.exe` {
		t.Fatalf("running app did not inherit its install path: %+v", merged[1])
	}
}

func TestInstalledExecutablePath(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "app.exe")
	if err := os.WriteFile(path, []byte("exe"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VOXT_TEST_ICON_DIR", directory)
	for _, command := range []string{`"` + path + `" /silent`, `%VOXT_TEST_ICON_DIR%\app.exe`} {
		if got, found := installedExecutablePath(command); !found || got != path {
			t.Fatalf("installedExecutablePath(%q) = %q, %v; want %q", command, got, found, path)
		}
	}
}

func TestMakeAppIconsOrderLimitsAndMissing(t *testing.T) {
	path := filepath.Join(t.TempDir(), "app.exe")
	if err := os.WriteFile(path, []byte("exe"), 0o600); err != nil {
		t.Fatal(err)
	}
	app := appInfoForPath(path, false, 0, "App")
	params := proto.AppIconsParams{Size: 300, Apps: []proto.AppRef{
		{BundleID: "missing"}, {BundleID: app.BundleID},
	}}
	var extractedSize int
	result, err := makeAppIcons(params, []proto.AppInfo{app}, func(gotPath string, size int) ([]byte, error) {
		if gotPath != path {
			return nil, errors.New("wrong path")
		}
		extractedSize = size
		return []byte("png"), nil
	})
	if err != nil || len(result.Icons) != 2 || result.Icons[0].PNG != nil || string(result.Icons[1].PNG) != "png" {
		t.Fatalf("icons = %+v, %v", result, err)
	}
	if extractedSize != proto.MaxIconSize || result.Icons[1].BundleID != app.BundleID {
		t.Fatalf("size %d, response order %+v", extractedSize, result.Icons)
	}
	if iconSide(0) != proto.IconSize || iconSide(-1) != proto.IconSize || iconSide(32) != 32 {
		t.Fatalf("unexpected icon sizes: %d %d %d", iconSide(0), iconSide(-1), iconSide(32))
	}
	params.Apps = make([]proto.AppRef, proto.MaxIcons+1)
	if _, err := makeAppIcons(params, nil, nil); err == nil {
		t.Fatal("an oversized icon batch was accepted")
	}
}

func TestRGBAFromBGRA(t *testing.T) {
	converted, err := rgbaFromBGRA(1, 1, []byte{10, 20, 30, 40})
	if err != nil {
		t.Fatal(err)
	}
	if got := converted.Pix; !reflect.DeepEqual(got, []byte{30, 20, 10, 40}) {
		t.Fatalf("converted pixels = %v", got)
	}
	if _, err := rgbaFromBGRA(2, 2, []byte{0}); err == nil {
		t.Fatal("a short pixel buffer was accepted")
	}
}
