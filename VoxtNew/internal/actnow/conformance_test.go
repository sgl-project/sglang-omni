package actnow

// actnowkit's conformance scenarios for VoxtNew's shared settings and
// shortcuts (family spec §16.1 scenarios 5, 7, 9): the adapter drives a
// Family over the fixture with VoxtNew's own settings document, as the
// app does (family.go).

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/conformance"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

func TestFamilyConformance(t *testing.T) {
	conformance.Run(t, func(t testing.TB, fx *conformance.Fixture, bundleID string) any {
		f := Open(Config{BundleID: bundleID, Version: "0.1.0", Dir: fx.Family, Hub: fx.Hub, System: conformance.FixtureSystem{}, Logf: t.Logf}, settings.Defaults())
		return &adapter{fx: fx, f: f, prefs: settings.Defaults()}
	})
}

type adapter struct {
	fx    *conformance.Fixture
	f     *Family
	prefs settings.Settings
}

func (a *adapter) BundleID() string { return a.f.BundleID() }

// Set is the user changing a shared setting in VoxtNew: its own value,
// written to the family while it follows (familySettingsChanged).
func (a *adapter) Set(ctx context.Context, key, value string) error {
	SetOwn(&a.prefs, key, value)
	return a.f.Set(ctx, a.prefs, key, value)
}

func (a *adapter) StopFollowing(_ context.Context, key string) error {
	a.f.ReadShared()
	a.f.Follow(&a.prefs, key, false)
	return nil
}

// Effective re-reads the family's file, as on activation.
func (a *adapter) Effective(_ context.Context, key string) (string, error) {
	a.f.ReadShared()
	a.f.Mirror(&a.prefs)
	return a.f.Effective(a.prefs, key), nil
}

// Declare makes sc VoxtNew's Transcribe bindings and writes the manifest
// (a fixture app's path: tests run outside a bundle).
func (a *adapter) Declare(_ context.Context, sc []shortcuts.Shortcut) error {
	a.prefs.Features.Transcribe.Hotkeys = nil
	a.prefs.Features.Translate.Hotkeys, a.prefs.Features.Rewrite.Hotkeys = nil, nil
	a.prefs.Features.PasteLast = settings.Hotkey{}
	for _, s := range sc {
		trigger := settings.TriggerTap
		switch shortcuts.NormalizeTrigger(s.Trigger) {
		case shortcuts.Hold:
			trigger = settings.TriggerLong
		case shortcuts.Double:
			trigger = settings.TriggerDouble
		}
		a.prefs.Features.Transcribe.Hotkeys = append(a.prefs.Features.Transcribe.Hotkeys, settings.Hotkey{Keys: s.Keys, Trigger: trigger})
	}
	StampKeys(&a.prefs, time.Now())
	m := a.f.Manifest(a.prefs)
	m.Path = filepath.Join(filepath.Dir(a.fx.Family.Root), "Applications", a.f.BundleID(), "VoxtNew.app")
	return a.f.WriteManifest(m)
}

func (a *adapter) Conflicts(context.Context) ([]string, error) {
	a.f.Discover()
	var out []string
	for _, c := range a.f.Conflicts(a.prefs) {
		out = append(out, c.Other.BundleID)
	}
	return out, nil
}
