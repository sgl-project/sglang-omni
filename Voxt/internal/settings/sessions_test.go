package settings

// The session slots shortcuts (Features.Sessions, #4005): their defaults,
// their migration into older documents and the clashes that keep a slot
// off.

import (
	"encoding/json"
	"testing"
)

// TestSessionShortcuts: a document saved before the session slots existed
// gets their defaults, tapped; one whose shortcut already taps those keys
// keeps that slot off; a document that has sessions keeps them, and a
// hand-written trigger is read as the tap it always is.
func TestSessionShortcuts(t *testing.T) {
	s, err := Migrate([]byte(`{"version":4,"features":{"transcribe":{"hotkeys":[{"keys":"RightOption","trigger":"tap"}]}}}`))
	want := [4]Hotkey{
		{Keys: DefaultSession1, Trigger: TriggerTap},
		{Keys: DefaultSession2, Trigger: TriggerTap},
		{Keys: DefaultSession3, Trigger: TriggerTap},
		{Keys: DefaultSessionNext, Trigger: TriggerTap},
	}
	if err != nil || s.Features.Sessions != want || s.Features.SessionsOn {
		t.Fatalf("older document: %+v %v", s.Features.Sessions, err)
	}
	// Round trip: a document that has sessions keeps them, one off slot too.
	s.Features.Sessions[1] = Hotkey{}
	data, _ := json.Marshal(s)
	m, err := Migrate(data)
	if err != nil || m.Features.Sessions[0].Keys != DefaultSession1 || m.Features.Sessions[1].Keys != "" || m.Features.Sessions[2].Keys != DefaultSession3 {
		t.Errorf("round trip: %+v %v", m.Features.Sessions, err)
	}
	// A hand-written trigger normalizes to a tap.
	m, err = Migrate([]byte(`{"version":4,"features":{"sessions":[{"keys":" F6 ","trigger":"long"}]}}`))
	if err != nil || m.Features.Sessions[0] != (Hotkey{Keys: "F6", Trigger: TriggerTap}) {
		t.Errorf("hand-written: %+v %v", m.Features.Sessions, err)
	}
}

// TestSessionShortcutsYieldToTaken: a document from before the session
// slots whose shortcuts can match a slot's default press keeps that slot
// off, rather than the keys going silently to that shortcut — a sided
// modifier's spelling too ("Control+LeftOption+1" takes "Control+Option+1").
func TestSessionShortcutsYieldToTaken(t *testing.T) {
	for _, tc := range []struct {
		name string
		keys string
		off  int // the slot left off, -1 for none
	}{
		{"the same keys", "Control+Option+1", 0},
		{"a sided option", "Control+LeftOption+1", 0},
		{"the other side", "Control+RightOption+2", 1},
		{"fewer modifiers", "Option+1", -1},
		{"another key", "Control+Option+4", -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			doc := `{"version":4,"features":{"transcribe":{"hotkeys":[{"keys":"` + tc.keys + `","trigger":"tap"}]}}}`
			s, err := Migrate([]byte(doc))
			if err != nil {
				t.Fatal(err)
			}
			for i, h := range s.Features.Sessions {
				if i == tc.off {
					if h.Keys != "" {
						t.Errorf("%s: slot %d kept %+v", tc.keys, i+1, h)
					}
				} else if h.Keys == "" {
					t.Errorf("%s: slot %d lost its default", tc.keys, i+1)
				}
			}
		})
	}
}
