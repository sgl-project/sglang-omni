package settings

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/hehehai/actnowkit/shortcuts"
)

func TestDefaultsAreNormal(t *testing.T) {
	d := Defaults()
	if !reflect.DeepEqual(normalize(clone(d)), d) {
		t.Fatalf("normalize changes the defaults:\n%+v\n%+v", normalize(d), d)
	}
}

func TestMigrate(t *testing.T) {
	s, err := Migrate(nil)
	if err != nil || !reflect.DeepEqual(s, Defaults()) {
		t.Fatalf("empty: %+v %v", s, err)
	}
	// A partial document keeps what it has and gains the defaults.
	s, err = Migrate([]byte(`{"version":1,"general":{"language":"zh-Hans","menuBar":false},"privacy":{"keepHistory":false},"unknown":1}`))
	if err != nil || s.General.Language != "zh-Hans" || s.General.MenuBar || s.Privacy.KeepHistory || s.Appearance.Scale != 1 ||
		s.Appearance.Theme != "system" || len(s.Features.Transcribe.Hotkeys) != 1 {
		t.Fatalf("partial: %+v %v", s, err)
	}
	data, _ := json.Marshal(s)
	again, err := Migrate(data)
	if err != nil || !reflect.DeepEqual(again, s) {
		t.Fatalf("round trip: %+v %v", again, err)
	}
	if _, err := Migrate([]byte(`{`)); err == nil {
		t.Error("bad JSON accepted")
	}
}

// docs is a Documents in memory that can fail.
type docs struct {
	memDocs
	fail error
}

func (d *docs) Put(ctx context.Context, key string, doc []byte) error {
	if d.fail != nil {
		return d.fail
	}
	return d.memDocs.Put(ctx, key, doc)
}

func TestStore(t *testing.T) {
	ctx := context.Background()
	d := &docs{}
	var events []Changed
	s, err := Open(ctx, d, func(c Changed) { events = append(events, c) })
	if err != nil || !reflect.DeepEqual(s.Get(), Defaults()) {
		t.Fatalf("fresh: %+v %v", s.Get(), err)
	}
	if err := s.Update(ctx, func(s *Settings) {
		s.General.Language = "zh-Hans"
		s.Appearance.Scale = 1.12
		s.Appearance.Theme = "sepia"
		s.Appearance.SidebarWidth = 900
		s.General.OtherLanguages = []string{" en ", "en", ""}
		s.Features.Translate.Hotkeys = append(s.Features.Translate.Hotkeys, Hotkey{Keys: " "}, Hotkey{Keys: "Option+T", Trigger: "weird"})
		s.Overlay.Opacity = 0.1
	}); err != nil {
		t.Fatal(err)
	}
	got := s.Get()
	if got.General.Language != "zh-Hans" || got.Appearance.Scale != 1.1 || got.Appearance.Theme != "system" ||
		got.Appearance.SidebarWidth != MaxSidebarWidth || !reflect.DeepEqual(got.General.OtherLanguages, []string{"en"}) ||
		len(got.Features.Translate.Hotkeys) != 2 || got.Features.Translate.Hotkeys[1].Trigger != TriggerTap || got.Overlay.Opacity != 0.5 {
		t.Fatalf("normalized: %+v", got)
	}
	if len(events) != 1 || events[0].Old.General.Language != "" || events[0].New.General.Language != "zh-Hans" {
		t.Fatalf("events %+v", events)
	}
	// A copy: changing it changes nothing.
	got.General.OtherLanguages[0] = "x"
	got.Features.Translate.Hotkeys[0].Keys = "x"
	if s.Get().General.OtherLanguages[0] != "en" || s.Get().Features.Translate.Hotkeys[0].Keys != DefaultTranslate {
		t.Error("Get returned shared state")
	}
	// No change: nothing published.
	s.Update(ctx, func(s *Settings) { s.General.Language = "zh-Hans" })
	if len(events) != 1 {
		t.Errorf("no-op published: %d", len(events))
	}
	// Reopened from the documents.
	again, err := Open(ctx, d, nil)
	if err != nil || !reflect.DeepEqual(again.Get(), s.Get()) {
		t.Fatalf("reopened %+v %v", again.Get(), err)
	}
	// A failed save changes nothing.
	d.fail = context.DeadlineExceeded
	if err := s.Update(ctx, func(s *Settings) { s.General.Language = "en" }); err == nil || s.Get().General.Language != "zh-Hans" {
		t.Errorf("failed save: %v %q", err, s.Get().General.Language)
	}
	// A broken document: defaults, and an error to log.
	d.fail = nil
	d.memDocs.Put(ctx, Key, []byte("{"))
	broken, err := Open(ctx, d, nil)
	if err == nil || !reflect.DeepEqual(broken.Get(), Defaults()) {
		t.Errorf("broken: %+v %v", broken.Get(), err)
	}
}

func TestMemory(t *testing.T) {
	s := NewMemory()
	s.Update(context.Background(), func(s *Settings) { s.Audio.Sounds = false })
	if s.Get().Audio.Sounds {
		t.Error("not updated")
	}
}

// TestSoundPreset: older documents have none (the family's default
// cues); a Voxt preset becomes its system sounds, Soft the defaults, and
// the preset is not written again.
func TestSoundPreset(t *testing.T) {
	s, err := Migrate([]byte(`{"version":3,"audio":{"sounds":true}}`))
	if err != nil || !s.Audio.Sounds || s.Audio.StartSound != "" || s.Audio.StopSound != "" || s.Audio.SoundVolumeOrDefault() != 70 {
		t.Fatalf("old document: %+v %v", s.Audio, err)
	}
	for preset, want := range map[string][2]string{
		"soft": {"", ""}, "nope": {"", ""}, "hero": {"system:Hero", "system:Hero"}, "glass": {"system:Ping", "system:Ping"},
	} {
		s, err := Migrate([]byte(`{"version":5,"audio":{"sounds":true,"soundPreset":"` + preset + `"}}`))
		if err != nil || s.Audio.StartSound != want[0] || s.Audio.StopSound != want[1] || s.Audio.SoundPreset != "" {
			t.Errorf("%s: %+v %v", preset, s.Audio, err)
		}
	}
	// A choice already made wins over a stale preset.
	s, _ = Migrate([]byte(`{"audio":{"soundPreset":"hero","startSound":"tally"}}`))
	if s.Audio.StartSound != "tally" || s.Audio.StopSound != "" {
		t.Errorf("chosen: %+v", s.Audio)
	}
	data, _ := json.Marshal(Defaults())
	for _, key := range []string{"soundPreset", "startSound", "soundVolume"} {
		if strings.Contains(string(data), key) {
			t.Errorf("empty %s written: %s", key, data)
		}
	}
	s.Audio.StartSound, s.Audio.StopSound, s.Audio.SoundVolume = "system:Pop", "none", 40
	data, _ = json.Marshal(s)
	back, err := Migrate(data)
	if err != nil || back.Audio.StartSound != "system:Pop" || back.Audio.StopSound != "none" || back.Audio.SoundVolumeOrDefault() != 40 {
		t.Errorf("round trip: %+v %v", back.Audio, err)
	}
}

func TestDeviceNames(t *testing.T) {
	s := Defaults()
	s.Audio.Devices = []string{"a", " a ", "b"}
	s.Audio.DeviceNames = map[string]string{"a": "Mic A", "gone": "Old mic"}
	n := normalize(clone(s))
	if !reflect.DeepEqual(n.Audio.Devices, []string{"a", "b"}) || !reflect.DeepEqual(n.Audio.DeviceNames, map[string]string{"a": "Mic A"}) {
		t.Fatalf("normalized: %+v", n.Audio)
	}
	c := clone(n)
	c.Audio.DeviceNames["a"] = "changed"
	if n.Audio.DeviceNames["a"] != "Mic A" {
		t.Error("clone shares the names")
	}
	s.Audio.Devices, s.Audio.DeviceNames = nil, map[string]string{"x": "X"}
	if n := normalize(clone(s)); n.Audio.DeviceNames != nil {
		t.Errorf("names without devices: %v", n.Audio.DeviceNames)
	}
}

// TestMigrateOverlayOpacity: version 2 makes the overlay translucent,
// version 3 0.8 and rounder (24), but each keeps a value the user chose
// (anything but an older version's default).
func TestMigrateOverlayOpacity(t *testing.T) {
	for _, tc := range []struct {
		doc            string
		opacity, round float64
	}{
		{`{"version":1,"overlay":{"position":"bottom","opacity":1,"radius":16,"margin":96}}`, DefaultOpacity, DefaultRadius},
		{`{"version":1,"overlay":{"position":"bottom","opacity":0.9,"radius":10,"margin":96}}`, 0.9, 10},
		{`{"version":2,"overlay":{"position":"bottom","opacity":0.72,"radius":16,"margin":96}}`, DefaultOpacity, DefaultRadius},
		{`{"version":2,"overlay":{"position":"bottom","opacity":1,"radius":20,"margin":96}}`, 1, 20},
		{`{"version":2,"overlay":{"position":"bottom","opacity":0.6,"radius":16,"margin":96}}`, 0.6, DefaultRadius},
		{`{"version":3,"overlay":{"position":"bottom","opacity":0.72,"radius":16,"margin":96}}`, 0.72, 16},
		{`{"version":1}`, DefaultOpacity, DefaultRadius},
	} {
		s, err := Migrate([]byte(tc.doc))
		if err != nil || s.Overlay.Opacity != tc.opacity || s.Overlay.Radius != tc.round || s.Version != Version || s.Overlay.ShowTime {
			t.Errorf("%s: %+v %v", tc.doc, s.Overlay, err)
		}
	}
	if d := Defaults().Overlay; d.Opacity != 0.8 || d.Radius != 24 {
		t.Errorf("defaults %+v", d)
	}
	// The detection's settings are all zero (the package's defaults) and
	// stay out of the document.
	data, _ := json.Marshal(Defaults())
	if strings.Contains(string(data), `"vad"`) {
		t.Errorf("vad written: %s", data)
	}
}

// TestCleanOnByDefault: the sound processing is on in a fresh install and
// in every older document (it has no field there: zero is on), stays out
// of the document while it is on, and a choice to turn it off stays.
func TestCleanOnByDefault(t *testing.T) {
	if Defaults().Audio.Clean != (Clean{}) {
		t.Errorf("defaults %+v", Defaults().Audio.Clean)
	}
	for _, doc := range []string{`{"version":1}`, `{"version":3,"audio":{"sounds":true,"vad":{"off":true}}}`} {
		s, err := Migrate([]byte(doc))
		if err != nil || s.Audio.Clean.Off || !s.Audio.VAD.Off == strings.Contains(doc, "vad") {
			t.Errorf("%s: %+v %v", doc, s.Audio, err)
		}
	}
	data, _ := json.Marshal(Defaults())
	if strings.Contains(string(data), `"clean"`) {
		t.Errorf("clean written: %s", data)
	}
	s := Defaults()
	s.Audio.Clean.Off = true
	data, _ = json.Marshal(s)
	if m, err := Migrate(data); err != nil || !m.Audio.Clean.Off {
		t.Errorf("off lost: %s", data)
	}
}

// TestDefaultShortcutsAreTheFamilys: the defaults are VoxtNew's row of
// the ActNow family's table (actnowkit shortcuts.DefaultsOf, spec
// §10.2), in its canonical spelling, so that no two family apps default
// to the same keys (Auralink has fn).
func TestDefaultShortcutsAreTheFamilys(t *testing.T) {
	f := Defaults().Features
	ours := map[string]Hotkey{
		"transcribe": f.Transcribe.Hotkeys[0],
		"translate":  f.Translate.Hotkeys[0],
		"rewrite":    f.Rewrite.Hotkeys[0],
		"pasteLast":  f.PasteLast,
	}
	family := shortcuts.DefaultsOf("voxtnew")
	if len(family) != len(ours) || len(f.Transcribe.Hotkeys)+len(f.Translate.Hotkeys)+len(f.Rewrite.Hotkeys) != 3 {
		t.Fatalf("family %+v, ours %+v", family, ours)
	}
	for _, sc := range family {
		h, ok := ours[sc.ID]
		if !ok {
			t.Errorf("family shortcut %q has no setting", sc.ID)
			continue
		}
		// A modifier-only chord is a tap; a key with modifiers fires on
		// its press, which VoxtNew writes as a tap too.
		want := shortcuts.Tap
		if sc.ID == "pasteLast" {
			want = shortcuts.Press
		}
		if h.Keys != sc.Keys || h.Trigger != TriggerTap || sc.Trigger != want || !sc.Enabled {
			t.Errorf("%s: ours %+v, the family's %+v", sc.ID, h, sc)
		}
	}
}

// TestMigrateShortcuts: version 4 moves the old defaults (Auralink's)
// to the family's, one by one, and keeps every shortcut the user chose.
func TestMigrateShortcuts(t *testing.T) {
	old := `{"version":3,"features":{
		"transcribe":{"hotkeys":[{"keys":"Fn","trigger":"tap"}]},
		"translate":{"hotkeys":[{"keys":"Fn+Shift"}]},
		"rewrite":{"hotkeys":[{"keys":"Fn+Control","trigger":"tap"}]},
		"pasteLast":{"keys":"Control+Command+V","trigger":"tap"}}}`
	s, err := Migrate([]byte(old))
	if err != nil || !reflect.DeepEqual(s.Features, Defaults().Features) || s.Version != Version {
		t.Fatalf("old defaults: %+v %v", s.Features, err)
	}
	chosen := `{"version":3,"features":{
		"transcribe":{"hotkeys":[{"keys":"Fn","trigger":"long"}]},
		"translate":{"hotkeys":[{"keys":"Fn+Shift","trigger":"tap"},{"keys":"F5","trigger":"tap"}]},
		"rewrite":{"hotkeys":[{"keys":"RightCommand","trigger":"tap"}]},
		"pasteLast":{"keys":"Control+Command+V","trigger":"double"}}}`
	s, err = Migrate([]byte(chosen))
	f := s.Features
	if err != nil || f.Transcribe.Hotkeys[0] != (Hotkey{"Fn", TriggerLong}) || len(f.Translate.Hotkeys) != 2 ||
		f.Rewrite.Hotkeys[0].Keys != "RightCommand" || f.PasteLast != (Hotkey{"Control+Command+V", TriggerDouble}) {
		t.Fatalf("chosen: %+v %v", f, err)
	}
	// Version 4 documents are the user's, even with fn.
	s, err = Migrate([]byte(strings.Replace(old, `"version":3`, `"version":4`, 1)))
	if err != nil || s.Features.Transcribe.Hotkeys[0].Keys != "Fn" || s.Features.PasteLast.Keys != "Control+Command+V" {
		t.Fatalf("version 4: %+v %v", s.Features, err)
	}
	// Partly old: each shortcut on its own.
	s, err = Migrate([]byte(`{"version":3,"features":{"transcribe":{"hotkeys":[{"keys":"Fn"}]},"rewrite":{"hotkeys":[{"keys":"F6"}]}}}`))
	if err != nil || s.Features.Transcribe.Hotkeys[0].Keys != DefaultTranscribe || s.Features.Rewrite.Hotkeys[0].Keys != "F6" ||
		s.Features.Translate.Hotkeys[0].Keys != DefaultTranslate || s.Features.PasteLast.Keys != DefaultPasteLast {
		t.Fatalf("partly old: %+v %v", s.Features, err)
	}
}

// TestBetaUpdates: betas follow the running version until the user
// chooses, older documents have no choice, and a choice either way stays.
func TestBetaUpdates(t *testing.T) {
	s, err := Migrate([]byte(`{"version":3}`))
	if err != nil || s.Updates.Beta != nil {
		t.Fatalf("older document: %+v %v", s.Updates, err)
	}
	for v, want := range map[string]bool{"0.1.0-beta.1": true, "1.0.0-rc.2": true, "0.1.0": false, "": false} {
		if got := s.Updates.BetaOn(v); got != want {
			t.Errorf("BetaOn(%q) = %t without a choice", v, got)
		}
	}
	data, _ := json.Marshal(Defaults())
	if strings.Contains(string(data), `"updates"`) {
		t.Errorf("updates written without a choice: %s", data)
	}
	for _, on := range []bool{true, false} {
		s := Defaults()
		s.Updates.Beta = &on
		data, _ := json.Marshal(s)
		m, err := Migrate(data)
		if err != nil || m.Updates.Beta == nil || *m.Updates.Beta != on {
			t.Fatalf("choice %t lost: %s", on, data)
		}
		for _, v := range []string{"0.1.0-beta.1", "0.1.0"} {
			if m.Updates.BetaOn(v) != on {
				t.Errorf("choice %t, running %s: %t", on, v, m.Updates.BetaOn(v))
			}
		}
	}
}

// TestMuteOutput: recordings mute the output unless the user turned it
// off; older documents (none, or the P1 placeholder's false that was
// never written) read as on, and a choice either way stays.
func TestMuteOutput(t *testing.T) {
	s, err := Migrate([]byte(`{"version":4}`))
	if err != nil || s.Audio.MuteOutput != nil || !s.Audio.MuteOn() || !Defaults().Audio.MuteOn() {
		t.Fatalf("older document: %+v %v", s.Audio, err)
	}
	if data, _ := json.Marshal(Defaults()); strings.Contains(string(data), "muteOutput") {
		t.Errorf("written without a choice: %s", data)
	}
	for _, on := range []bool{true, false} {
		s := Defaults()
		s.Audio.MuteOutput = &on
		data, _ := json.Marshal(s)
		m, err := Migrate(data)
		if err != nil || m.Audio.MuteOutput == nil || m.Audio.MuteOn() != on {
			t.Fatalf("choice %t lost: %s", on, data)
		}
		c := clone(m)
		*c.Audio.MuteOutput = !on
		if m.Audio.MuteOn() != on {
			t.Fatal("clone shares the choice")
		}
	}
}
