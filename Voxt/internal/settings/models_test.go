package settings

import (
	"encoding/json"
	"testing"
	"time"
)

// TestKeepFor: each choice of Models.KeepLoaded is its idle time; empty
// and unknown values are Automatic.
func TestKeepFor(t *testing.T) {
	for _, c := range []struct {
		v    string
		d    time.Duration
		auto bool
	}{
		{KeepAuto, 0, true},
		{Keep5Min, 5 * time.Minute, false},
		{Keep15Min, 15 * time.Minute, false},
		{Keep30Min, 30 * time.Minute, false},
		{Keep1Hour, time.Hour, false},
		{KeepUntilQuit, -1, false},
		{"2h", 0, true},
		{"forever", 0, true},
	} {
		d, auto := Models{KeepLoaded: c.v}.KeepFor()
		if d != c.d || auto != c.auto {
			t.Errorf("%q: %v %v, want %v %v", c.v, d, auto, c.d, c.auto)
		}
	}
	if len(KeepChoices) != 6 || KeepChoices[0] != KeepAuto {
		t.Errorf("choices %q", KeepChoices)
	}
}

// TestKeepLoadedDocument: older documents have no models part (Automatic,
// no migration); a choice survives a save and a read.
func TestKeepLoadedDocument(t *testing.T) {
	s, err := Migrate([]byte(`{"version":4,"general":{"menuBar":true}}`))
	if err != nil || s.Models.KeepLoaded != KeepAuto || s.Version != Version {
		t.Fatalf("an older document: %+v %v", s.Models, err)
	}
	s.Models.KeepLoaded = KeepUntilQuit
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	back, err := Migrate(data)
	if err != nil || back.Models.KeepLoaded != KeepUntilQuit {
		t.Errorf("read back %+v %v", back.Models, err)
	}
}

// TestLLMWait (#3821): each choice of Models.LLMWait is the least wait for
// a language model; empty and unknown values are Automatic (0). Older
// documents have none (no migration); a choice survives a save and a read.
func TestLLMWait(t *testing.T) {
	for _, c := range []struct {
		v string
		d time.Duration
	}{
		{WaitAuto, 0}, {Wait1Min, time.Minute}, {Wait2Min, 2 * time.Minute},
		{Wait5Min, 5 * time.Minute}, {Wait10Min, 10 * time.Minute}, {"1h", 0},
	} {
		if d := (Models{LLMWait: c.v}).LLMWaitFor(); d != c.d {
			t.Errorf("%q: %v, want %v", c.v, d, c.d)
		}
	}
	s, err := Migrate([]byte(`{"version":4,"models":{"keepLoaded":"5m"}}`))
	if err != nil || s.Models.LLMWait != WaitAuto || s.Models.KeepLoaded != Keep5Min {
		t.Fatalf("an older document: %+v %v", s.Models, err)
	}
	s.Models.LLMWait = Wait5Min
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if back, err := Migrate(data); err != nil || back.Models.LLMWait != Wait5Min {
		t.Errorf("read back %+v %v", back.Models, err)
	}
}

// TestFasterOn (#3873): the Faster transcription switch decides whether a
// speech model uses its draft, which must be installed; a model never
// switched (older documents) uses an installed draft, as 2.0.2 did. The
// switch survives a save and a read, and a clone does not share it.
func TestFasterOn(t *testing.T) {
	m := Models{Faster: map[string]bool{"on": true, "off": false}}
	for _, c := range []struct {
		id        string
		installed bool
		want      bool
	}{
		{"never", true, true}, {"never", false, false},
		{"on", true, true}, {"on", false, false},
		{"off", true, false}, {"off", false, false},
	} {
		if got := m.FasterOn(c.id, c.installed); got != c.want {
			t.Errorf("%s, installed %v: %v", c.id, c.installed, got)
		}
	}
	if !(Models{}).FasterOn("x", true) {
		t.Error("no switches at all: an installed draft is off")
	}
	s := Defaults()
	s.Models.Faster = map[string]bool{"asr.fast": false}
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	back, err := Migrate(data)
	if on, set := back.Models.Faster["asr.fast"]; err != nil || !set || on {
		t.Errorf("read back %+v %v", back.Models, err)
	}
	c := clone(back)
	c.Models.Faster["asr.fast"] = true
	if back.Models.Faster["asr.fast"] {
		t.Error("a clone shares the switches")
	}
}
