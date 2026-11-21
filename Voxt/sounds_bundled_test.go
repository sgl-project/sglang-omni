package main

import (
	"os"
	"testing"

	"github.com/hehehai/actnowkit/sounds"
)

// TestBundledSounds checks the family's sounds an official build ships
// (resources/sounds, fetched by scripts/sync-sounds.sh, not in the
// repository): the catalogue loads, every file is there and matches its
// SHA-256, and it has the cues' defaults, every sound of Settings → Audio's
// pairs and every sound meant to start or stop a recording. Without the
// copy it is skipped.
func TestBundledSounds(t *testing.T) {
	if _, err := os.Stat("resources/sounds/catalog.json"); err != nil {
		t.Skip("no resources/sounds (scripts/sync-sounds.sh): a source build")
	}
	cat, err := sounds.Load(os.DirFS("resources"), "sounds")
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range cat.Check() {
		t.Error(err)
	}
	for _, ev := range []sounds.Event{recordStartEvent, recordStopEvent} {
		if _, ok := cat.Sound(ev.Default); !ok {
			t.Errorf("%s: no %s", ev.ID, ev.Default)
		}
		if n := len(cat.Suggested(ev)); n < 4 {
			t.Errorf("%s: %d suggested sounds", ev.ID, n)
		}
	}
	for _, p := range soundPairs {
		for _, id := range []string{p.start, p.stop} {
			if _, ok := cat.Sound(id); !ok {
				t.Errorf("pair %s: no %s", p.key, id)
			}
		}
	}
	if cat.Notice == "" {
		t.Error("no notice")
	}
}
