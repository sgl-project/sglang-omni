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
