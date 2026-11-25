package settings

// The session shortcuts (Features.Sessions, #4005): their defaults, their
// migration into older documents and the clashes that keep one off.

import (
	"encoding/json"
	"testing"
)

// TestSessionShortcuts: a document saved before sessions existed gets the
// defaults: only "Next session" is bound, to two keys, tapped; the
// shortcuts of session 1, 2 and 3 are off. A document that has sessions
// keeps them, and a hand-written trigger is read as the tap it always is.
func TestSessionShortcuts(t *testing.T) {
	s, err := Migrate([]byte(`{"version":4,"features":{"transcribe":{"hotkeys":[{"keys":"RightOption","trigger":"tap"}]}}}`))
	want := [4]Hotkey{3: {Keys: "Option+Tab", Trigger: TriggerTap}}
	if err != nil || s.Features.Sessions != want || s.Features.Sessions != Defaults().Features.Sessions || s.Features.SessionsOn {
		t.Fatalf("older document: %+v %v", s.Features.Sessions, err)
	}
	// Round trip: a document that has sessions keeps them, an off one too.
	s.Features.Sessions[1] = Hotkey{Keys: "F6", Trigger: TriggerTap}
	s.Features.Sessions[3] = Hotkey{}
	data, _ := json.Marshal(s)
	m, err := Migrate(data)
	if k := m.Features.Sessions; err != nil || k[0].Keys != "" || k[1].Keys != "F6" || k[2].Keys != "" || k[3].Keys != "" {
		t.Errorf("round trip: %+v %v", m.Features.Sessions, err)
	}
	// A hand-written trigger normalizes to a tap.
	m, err = Migrate([]byte(`{"version":4,"features":{"sessions":[{"keys":" F6 ","trigger":"long"}]}}`))
	if err != nil || m.Features.Sessions[0] != (Hotkey{Keys: "F6", Trigger: TriggerTap}) {
		t.Errorf("hand-written: %+v %v", m.Features.Sessions, err)
	}
}

// TestSessionShortcutsYieldToTaken: a document from before sessions whose
// shortcut can match the next-session default's press keeps it off,
// rather than the keys going silently to that shortcut: a sided
// modifier's spelling too ("LeftOption+Tab" takes "Option+Tab").
func TestSessionShortcutsYieldToTaken(t *testing.T) {
	for _, tc := range []struct {
		name, keys string
		off        bool
	}{
		{"the same keys", "Option+Tab", true},
		{"a sided option", "LeftOption+Tab", true},
		{"the other side", "RightOption+Tab", true},
		{"more modifiers", "Control+Option+Tab", false},
		{"another key", "Option+1", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			doc := `{"version":4,"features":{"transcribe":{"hotkeys":[{"keys":"` + tc.keys + `","trigger":"tap"}]}}}`
			s, err := Migrate([]byte(doc))
			if err != nil {
				t.Fatal(err)
			}
			if got := s.Features.Sessions[3].Keys == ""; got != tc.off {
				t.Errorf("%s: next session %+v, off=%v want %v", tc.keys, s.Features.Sessions[3], got, tc.off)
			}
			if s.Features.Sessions[0].Keys != "" {
				t.Errorf("%s: session 1 got a default: %+v", tc.keys, s.Features.Sessions[0])
			}
		})
	}
}
