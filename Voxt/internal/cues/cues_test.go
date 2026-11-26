package cues

import (
	"strings"
	"testing"
	"testing/fstest"

	"github.com/hehehai/actnowkit/sounds"
)

// TestLoadAlone: without the family's sounds the catalogue is Voxt's
// cues, every file there and matching its SHA-256, each start with the
// stop that goes with it.
func TestLoadAlone(t *testing.T) {
	for _, family := range []fstest.MapFS{nil, {}} {
		var cat *sounds.Catalog
		var err error
		if family == nil {
			cat, err = Load(nil, "sounds")
		} else {
			cat, err = Load(family, "sounds")
		}
		if err != nil {
			t.Fatal(err)
		}
		if len(cat.Sounds) == 0 || len(cat.Sounds)%2 != 0 {
			t.Fatalf("%d sounds", len(cat.Sounds))
		}
		for _, err := range cat.Check() {
			t.Error(err)
		}
		for _, s := range cat.Sounds {
			pair, ok := cat.Sound(s.Pair)
			if !ok || pair.Pair != s.ID || s.SHA256 == "" || s.Name.Zh == "" || s.Desc != pair.Desc {
				t.Errorf("%s: pair %q, sha %q, name %+v, desc %+v", s.ID, s.Pair, s.SHA256, s.Name, s.Desc)
			}
			start := strings.HasSuffix(s.ID, "-start")
			if want := map[bool]string{true: "record.start", false: "record.stop"}[start]; len(s.Uses) != 1 || s.Uses[0] != want {
				t.Errorf("%s: uses %q", s.ID, s.Uses)
			}
		}
		if cat.Notice == "" {
			t.Error("no notice")
		}
	}
}

// TestLoadWithFamily: the family's sounds come after Voxt's, their files
// read from their own folder, with their groups, tags, uses and notice; a
// family sound with the id of one of Voxt's is left out.
func TestLoadWithFamily(t *testing.T) {
	const family = `{"schema": "actnow.sounds.catalog/1", "version": "test", "notice": "Family notice.",
 "groups": [{"id": "tap", "name": {"en": "Tap"}}, {"id": "soft", "name": {"en": "Soft (family)"}}],
 "tags": [{"id": "short", "name": {"en": "Short"}}], "uses": [{"id": "poke", "name": {"en": "Poke"}}],
 "sounds": [
  {"id": "pebble", "file": "sounds/pebble.wav", "name": {"en": "Pebble"}, "group": "tap", "duration": 0.23},
  {"id": "soft-kalimba-start", "file": "sounds/other.wav", "name": {"en": "Other"}, "group": "tap", "duration": 0.1}
 ]}`
	fsys := fstest.MapFS{
		"sounds/catalog.json":      {Data: []byte(family)},
		"sounds/sounds/pebble.wav": {Data: []byte("RIFFpebble")},
		"sounds/sounds/other.wav":  {Data: []byte("RIFFother")},
	}
	own, err := Load(nil, "")
	if err != nil {
		t.Fatal(err)
	}
	cat, err := Load(fsys, "sounds")
	if err != nil {
		t.Fatal(err)
	}
	if len(cat.Sounds) != len(own.Sounds)+1 || cat.Sounds[0].ID != own.Sounds[0].ID || cat.Sounds[len(cat.Sounds)-1].ID != "pebble" {
		t.Fatalf("%d sounds, %d of Voxt's", len(cat.Sounds), len(own.Sounds))
	}
	if b, err := cat.Data("pebble"); err != nil || string(b) != "RIFFpebble" {
		t.Errorf("pebble: %q %v", b, err)
	}
	kalimba, _ := cat.Sound("soft-kalimba-start")
	if b, err := cat.Data("soft-kalimba-start"); err != nil || len(b) < 1000 || kalimba.Name.En != "Kalimba (start)" {
		t.Errorf("soft-kalimba-start: %d bytes, %v, %+v", len(b), err, kalimba.Name)
	}
	for _, err := range cat.Check() {
		t.Error(err)
	}
	if cat.GroupName("tap", "en") != "Tap" || cat.GroupName("soft", "en") != "Soft" || cat.TagName("short", "en") != "Short" || cat.UseName("poke", "en") != "Poke" {
		t.Errorf("groups %+v, tags %+v, uses %+v", cat.Groups, cat.Tags, cat.Uses)
	}
	if !strings.HasPrefix(cat.Notice, "Family notice.\n") || !strings.Contains(cat.Notice, "Voxt") {
		t.Errorf("notice %q", cat.Notice)
	}
}

// TestMounts: only files under a mounted folder open.
func TestMounts(t *testing.T) {
	m := mounts{"a": fstest.MapFS{"x.wav": {Data: []byte("x")}}}
	if f, err := m.Open("a/x.wav"); err != nil {
		t.Fatal(err)
	} else {
		f.Close()
	}
	for _, name := range []string{"a", "b/x.wav", "a/../a/x.wav", "x.wav", "a/y.wav"} {
		if f, err := m.Open(name); err == nil {
			f.Close()
			t.Errorf("%q opened", name)
		}
	}
}
