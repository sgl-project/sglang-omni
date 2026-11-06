package appbundle

import (
	"os"
	"path/filepath"
	"testing"
)

// plist is an Info.plist as mygo build writes it: a nested dict before the
// version, whose keys must not be taken for the top dict's.
const plist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleName</key>
	<string>VoxtNew</string>
	<key>CFBundleURLTypes</key>
	<array>
		<dict>
			<key>CFBundleShortVersionString</key>
			<string>nested</string>
		</dict>
	</array>
	<key>LSMinimumSystemVersion</key>
	<string>15.0</string>
	<key>CFBundleShortVersionString</key>
	<string>0.1.1</string>
	<key>CFBundleVersion</key>
	<string>0.1.1</string>
</dict>
</plist>
`

func TestShortVersion(t *testing.T) {
	v, err := ShortVersion([]byte(plist))
	if err != nil || v != "0.1.1" {
		t.Fatalf("ShortVersion = %q, %v; want 0.1.1", v, err)
	}
	if _, err := ShortVersion([]byte(`<plist><dict><key>CFBundleName</key><string>x</string></dict></plist>`)); err != ErrNoVersion {
		t.Errorf("no version: %v", err)
	}
	if _, err := ShortVersion([]byte(`<plist><dict><key>`)); err == nil {
		t.Error("a broken plist read")
	}
}

// bundle makes Contents/Info.plist with version and returns
// Contents/Resources.
func bundle(t *testing.T, version string) string {
	t.Helper()
	contents := filepath.Join(t.TempDir(), "VoxtNew.app", "Contents")
	res := filepath.Join(contents, "Resources")
	if err := os.MkdirAll(res, 0o755); err != nil {
		t.Fatal(err)
	}
	p := `<plist version="1.0"><dict><key>CFBundleShortVersionString</key><string>` + version + `</string></dict></plist>`
	if err := os.WriteFile(filepath.Join(contents, "Info.plist"), []byte(p), 0o644); err != nil {
		t.Fatal(err)
	}
	return res
}

func TestUpdated(t *testing.T) {
	res := bundle(t, "0.1.1")
	if v, err := Version(res); err != nil || v != "0.1.1" {
		t.Fatalf("Version = %q, %v", v, err)
	}
	for _, c := range []struct {
		running string
		want    bool
	}{
		{"0.1.1", false}, // the version running
		{"0.1.0", true},  // an update was installed
		{"0.2.0", true},  // an older one (a rollback by hand)
		{"", false},      // a development build
	} {
		if got := Updated(res, c.running); got != c.want {
			t.Errorf("Updated(%q) = %v, want %v", c.running, got, c.want)
		}
	}
	// Outside a bundle (go run): no Info.plist.
	dir := t.TempDir()
	if _, err := Version(dir); !os.IsNotExist(err) {
		t.Errorf("no bundle: %v", err)
	}
	if Updated(dir, "0.1.0") {
		t.Error("no bundle reads as updated")
	}
}
