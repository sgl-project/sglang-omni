package hotkey

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestLegacyChordFromBinding(t *testing.T) {
	chord, err := legacyChordFromBinding(legacyHotkeyBinding{
		InputType: "keyboard", KeyCode: 96, Modifiers: nsModCommand, Behavior: "longPress",
	})
	if err != nil || chord != "Command+F5" {
		t.Fatalf("%q %v", chord, err)
	}
	if _, err := Parse(chord); err != nil {
		t.Fatal(err)
	}
	if legacyTrigger("longPress") != settings.TriggerLong {
		t.Fatal()
	}
}

func TestParseLegacyMeetingHotkeys(t *testing.T) {
	for _, test := range []struct {
		name       string
		preference map[string]string
		want       []settings.Hotkey
	}{
		{
			name: "binding list",
			preference: map[string]string{
				"meetingHotkeyBindings": `[{"inputType":"keyboard","keyCode":97,"modifiers":1048576,"behavior":"doubleTap"}]`,
			},
			want: []settings.Hotkey{{Keys: "Command+F6", Trigger: settings.TriggerDouble}},
		},
		{
			name: "single fields",
			preference: map[string]string{
				"meetingHotkeyKeyCode":   "96",
				"meetingHotkeyModifiers": "1048576",
				"meetingHotkeyInputType": "keyboard",
			},
			want: []settings.Hotkey{{Keys: "Command+F5", Trigger: settings.TriggerTap}},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			preference := func(_ string, key string) string { return test.preference[key] }
			if got := ParseLegacyMeetingHotkeys("unused", preference); !slices.Equal(got, test.want) {
				t.Fatalf("got %v, want %v", got, test.want)
			}
		})
	}
}
