package conformance

import (
	"context"
	"os"
	"path/filepath"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/registry"
	"github.com/hehehai/actnowkit/shortcuts"
)

// Bundle ids the scenarios use. They are test variants of family apps
// (spec §6.1), so that no real installation on the test Mac interferes:
// LaunchServices knows none of them.
const (
	AppUnderTest = "dev.hehehai.dicta.conformance"      // the app under test
	SecondApp    = "dev.hehehai.voxtnew.conformance"    // a second instance of the app under test
	PeerSmith    = "dev.hehehai.audiosmith.conformance" // peers, played by actnowkit
	PeerAuralink = "dev.hehehai.auralink.conformance"
	PeerGone     = "dev.hehehai.pik.conformance" // a peer that gets uninstalled
)

// FakeApp writes a minimal <dir>/<name>.app whose Info.plist names
// bundleID, and returns its path.
func FakeApp(dir, name, bundleID string) (string, error) {
	app := filepath.Join(dir, name+".app")
	if err := os.MkdirAll(filepath.Join(app, "Contents", "MacOS"), 0o755); err != nil {
		return "", err
	}
	plist := `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleIdentifier</key>
	<string>` + bundleID + `</string>
	<key>CFBundleName</key>
	<string>` + name + `</string>
</dict>
</plist>
`
	return app, os.WriteFile(filepath.Join(app, "Contents", "Info.plist"), []byte(plist), 0o644)
}

// Register makes a peer installed in the spec's sense (§6.3): a fake .app
// under the fixture and a manifest pointing at it, with shortcuts if
// given. It returns the app's path.
func (fx *Fixture) Register(bundleID string, sc ...shortcuts.Shortcut) string {
	fx.T.Helper()
	a, _, err := family.ParseBundleID(bundleID)
	if err != nil {
		fx.T.Fatalf("%s: %v", bundleID, err)
	}
	path, err := FakeApp(filepath.Join(filepath.Dir(fx.Family.Root), "Applications", bundleID), a.Name, bundleID)
	must(fx.T, err)
	m := registry.Manifest{
		App: a.ID, BundleID: bundleID, Name: a.Name, Version: "1.0.0", Path: path,
		Capabilities: []string{registry.CapModelsHF}, Models: &registry.ModelsInfo{Root: fx.Hub},
		Shortcuts: sc, Links: map[string]string{registry.LinkShortcuts: a.URLScheme + "://settings"},
	}
	must(fx.T, registry.Write(fx.Family, m))
	fx.Apps[bundleID] = models.AppState{BundleID: bundleID, AppID: a.ID, Name: a.Name, Installed: true, ModelsRoot: fx.Hub}
	// A peer on the spec v1.0.9 writes its claims file at launch, even
	// empty (models.Store.Heal): without it, deletions are unsure.
	if _, err := fx.Peer(bundleID).Heal(context.Background()); err != nil {
		fx.T.Fatalf("%s: %v", bundleID, err)
	}
	return path
}

// Uninstall deletes a registered peer's .app, as dragging it to the
// Trash does: its manifest and claims stay (spec §7.7).
func (fx *Fixture) Uninstall(bundleID string) {
	fx.T.Helper()
	m, err := registry.Read(fx.Family, bundleID)
	must(fx.T, err)
	must(fx.T, os.RemoveAll(m.Path))
	st := fx.Apps[bundleID]
	st.Installed = false
	fx.Apps[bundleID] = st
}

// FixtureSystem is discovery's view of the machine for a fixture: no app
// is known to LaunchServices or running, Info.plist files are read from
// disk. Adapters of the app under test may use it in place of the real
// system on a developer's Mac.
type FixtureSystem struct{}

func (FixtureSystem) AppPaths(string) []string                { return nil }
func (FixtureSystem) Running(string) []registry.Process       { return nil }
func (FixtureSystem) BundleID(appPath string) (string, error) { return registry.BundleIDAt(appPath) }
