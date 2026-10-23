package actnow

import (
	"context"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/conformance"
	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/registry"
	famsettings "github.com/hehehai/actnowkit/settings"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

func open(t *testing.T, fx *conformance.Fixture, prefs settings.Settings) *Family {
	t.Helper()
	return Open(Config{BundleID: "dev.hehehai.voxtnew", Version: "0.1.0-beta.1", Dir: fx.Family, Hub: fx.Hub,
		System: conformance.FixtureSystem{}, Logf: t.Logf}, prefs)
}

// TestManifest: VoxtNew's manifest has its capabilities, links, model
// library and shortcuts in the family's notation, written once until it
// changes.
func TestManifest(t *testing.T) {
	fx := conformance.New(t)
	prefs := settings.Defaults()
	prefs.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: "RightOption", Trigger: settings.TriggerTap}, {Keys: "Option+Space", Trigger: settings.TriggerLong}}
	prefs.Features.Translate.Hotkeys = []settings.Hotkey{{Keys: "RightOption+Shift"}}
	prefs.Features.Rewrite.Hotkeys = []settings.Hotkey{{Keys: "RightOption+Control", Trigger: settings.TriggerDouble}, {Keys: "Hyper+?"}}
	prefs.Features.PasteLast = settings.Hotkey{Keys: "Control+Option+V"}
	at := time.Date(2026, 10, 17, 8, 0, 0, 0, time.UTC)
	if !StampKeys(&prefs, at) || StampKeys(&prefs, at.Add(time.Hour)) {
		t.Fatal("stamps")
	}
	f := open(t, fx, prefs)
	m := f.Manifest(prefs)
	m.Path = "/Applications/VoxtNew.app"
	if err := f.WriteManifest(m); err != nil {
		t.Fatal(err)
	}
	got, err := registry.Read(fx.Family, "dev.hehehai.voxtnew")
	if err != nil {
		t.Fatal(err)
	}
	if got.App != AppID || got.Variant != family.Release || got.Name != "VoxtNew" || got.Version != "0.1.0-beta.1" || got.URLScheme != "voxtnew" ||
		got.Models == nil || got.Models.Root != fx.Hub || !slices.Equal(got.Capabilities, []string{"models.hf", "asr.local", "llm.local", "dictation"}) ||
		got.Links["shortcuts"] != "voxtnew://settings/shortcuts" {
		t.Errorf("manifest %+v", got)
	}
	type sc struct{ id, keys, trigger string }
	var have []sc
	for _, s := range got.Shortcuts {
		have = append(have, sc{s.ID, s.Keys, s.Trigger})
		if s.EnabledAt == nil || !s.EnabledAt.Equal(at) || !s.Enabled || s.Mechanism != shortcuts.EventTap || s.Exclusive != (s.ID == "pasteLast") {
			t.Errorf("shortcut %+v", s)
		}
	}
	want := []sc{{"transcribe", "RightOption", "tap"}, {"transcribe.2", "Option+Space", "hold"}, {"translate", "RightOption+Shift", "tap"},
		{"rewrite", "Control+RightOption", "double"}, {"pasteLast", "Control+Option+V", "press"}}
	if !slices.Equal(have, want) {
		t.Errorf("shortcuts %v, want %v", have, want)
	}
	// Unchanged: not written again.
	fi, _ := os.Stat(fx.Family.AppFile("dev.hehehai.voxtnew"))
	os.Chtimes(fx.Family.AppFile("dev.hehehai.voxtnew"), fi.ModTime().Add(-time.Hour), fi.ModTime().Add(-time.Hour))
	if err := f.WriteManifest(m); err != nil {
		t.Fatal(err)
	}
	if fi2, _ := os.Stat(fx.Family.AppFile("dev.hehehai.voxtnew")); !fi2.ModTime().Before(fi.ModTime()) {
		t.Error("written again")
	}
	// New keys: a new time.
	prefs.Features.PasteLast.Keys = "Control+Option+B"
	if !StampKeys(&prefs, at.Add(time.Hour)) || !prefs.Family.KeysSince["pasteLast"].Since.Equal(at.Add(time.Hour)) {
		t.Errorf("%+v", prefs.Family.KeysSince)
	}
	// Outside a bundle: no manifest.
	m.Path = ""
	m.Version = "2"
	if err := f.WriteManifest(m); err != nil {
		t.Fatal(err)
	}
	if got, _ := registry.Read(fx.Family, "dev.hehehai.voxtnew"); got.Version != "0.1.0-beta.1" {
		t.Error("written without a path")
	}
}

// TestConflicts: Auralink's Fn against VoxtNew's Fn (its old defaults)
// is a conflict; the later of the two prompts.
func TestConflicts(t *testing.T) {
	fx := conformance.New(t)
	fx.Register("dev.hehehai.auralink", shortcuts.DefaultsOf("auralink")...)
	prefs := settings.Defaults()
	prefs.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: "Fn", Trigger: settings.TriggerTap}}
	StampKeys(&prefs, time.Now().Add(time.Hour))
	f := open(t, fx, prefs)
	f.Discover()
	cs := f.Conflicts(prefs)
	if len(cs) == 0 || cs[0].Mine.ID != "transcribe" || cs[0].Other.Name != "Auralink" || !cs[0].Prompt {
		t.Fatalf("%+v", cs)
	}
	if hits := f.UsedBy("Control+Command+V"); len(hits) != 1 || hits[0].Shortcut.ID != "pasteLast" {
		t.Errorf("used by %+v", hits)
	}
	if AppName("dev.hehehai.dicta.dev") != "Dicta (dev)" || AppName("com.example") != "com.example" {
		t.Error("names")
	}
}

// TestSharedSettings: at the first start, VoxtNew's changed language goes
// to the family; a family value on a default follows; a family value on
// a changed one is a conflict, kept, noticed once; Mirror and Follow.
func TestSharedSettings(t *testing.T) {
	fx := conformance.New(t)
	ctx := context.Background()
	if _, err := famsettings.Set(ctx, fx.Family, "dev.hehehai.dicta", KeyProxy, "http://127.0.0.1:7897"); err != nil {
		t.Fatal(err)
	}
	if _, err := famsettings.Set(ctx, fx.Family, "dev.hehehai.dicta", KeyHFEndpoints, "mirror"); err != nil {
		t.Fatal(err)
	}
	prefs := settings.Defaults()
	prefs.General.Language = "zh-Hans"
	prefs.Network.HFEndpoints = "official"
	f := open(t, fx, prefs)
	if err := f.Migrate(ctx, &prefs); err != nil {
		t.Fatal(err)
	}
	if v, _ := f.Shared().Get(KeyLanguage); v != "zh-Hans" {
		t.Errorf("language %q", v)
	}
	if !prefs.Family.Follows(KeyLanguage) || !prefs.Family.Follows(KeyProxy) || prefs.Family.Follows(KeyHFEndpoints) ||
		!slices.Equal(prefs.Family.Conflicts, []string{KeyHFEndpoints}) {
		t.Fatalf("%+v", prefs.Family)
	}
	if f.Effective(prefs, KeyProxy) != "http://127.0.0.1:7897" || f.Effective(prefs, KeyHFEndpoints) != "official" {
		t.Errorf("effective %q %q", f.Effective(prefs, KeyProxy), f.Effective(prefs, KeyHFEndpoints))
	}
	if !f.Mirror(&prefs) || prefs.Network.Proxy != "http://127.0.0.1:7897" || prefs.Network.HFEndpoints != "official" {
		t.Errorf("mirror %+v", prefs.Network)
	}
	// "Use theirs".
	f.Follow(&prefs, KeyHFEndpoints, true)
	if prefs.Network.HFEndpoints != "mirror" || len(prefs.Family.Conflicts) != 0 {
		t.Errorf("use theirs %+v %+v", prefs.Network, prefs.Family)
	}
	// Another app changes the language: VoxtNew takes it.
	if _, err := famsettings.Set(ctx, fx.Family, "dev.hehehai.dicta", KeyLanguage, "en"); err != nil {
		t.Fatal(err)
	}
	if !f.ReadShared() || !f.Mirror(&prefs) || prefs.General.Language != "en" {
		t.Errorf("language %q", prefs.General.Language)
	}
	// VoxtNew's own change while following goes to the family.
	SetOwn(&prefs, KeyLanguage, "zh-Hans")
	if err := f.Set(ctx, prefs, KeyLanguage, "zh-Hans"); err != nil {
		t.Fatal(err)
	}
	if v, _ := f.Shared().Get(KeyLanguage); v != "zh-Hans" {
		t.Errorf("language %q", v)
	}
}

// TestReadOnly: a family directory of a newer spec is only read.
func TestReadOnly(t *testing.T) {
	fx := conformance.New(t)
	data := []byte(`{"schema":"actnow.family/1","created":"2026-10-20T08:00:00Z","createdBy":"x","specVersion":9}`)
	if err := os.WriteFile(fx.Family.FamilyFile(), data, 0o600); err != nil {
		t.Fatal(err)
	}
	prefs := settings.Defaults()
	f := open(t, fx, prefs)
	if !f.ReadOnly() {
		t.Fatal("not read only")
	}
	m := f.Manifest(prefs)
	m.Path = "/Applications/VoxtNew.app"
	if err := f.WriteManifest(m); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(fx.Family.AppFile("dev.hehehai.voxtnew")); !os.IsNotExist(err) {
		t.Error("a manifest in a read-only directory")
	}
	if err := f.Migrate(context.Background(), &prefs); err != nil || prefs.Family.Follow != nil {
		t.Errorf("migrated: %v %+v", err, prefs.Family)
	}
}
