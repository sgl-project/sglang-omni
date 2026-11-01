package uitext

import (
	"errors"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
)

func TestHotkeyText(t *testing.T) {
	for keys, want := range map[string]string{
		"Fn":                   "fn",
		"Fn+Shift":             "fn ⇧",
		"RightCommand":         "Right ⌘",
		"RightOption":          "Right ⌥",
		"Control+Command+V":    "⌃⌘V",
		"Control+Option+Space": "⌃⌥ Space",
		"Fn+F5":                "fn F5",
		"bogus":                "bogus",
	} {
		if got := HotkeyText(keys); got != want {
			t.Errorf("HotkeyText(%q) = %q, want %q", keys, got, want)
		}
	}
}

func TestParseErrorText(t *testing.T) {
	for keys, want := range map[string]string{
		"":         "Type the keys, as in Control+Option+Space.",
		"A+B":      "Only one key besides the modifiers.",
		"LeftFn":   "fn has no left or right side.",
		"Hyper+K":  "A key is unknown. Write the modifiers first, then one key: Fn, RightCommand, Control+Option+Space, Fn+F5.",
		"Option+K": "",
	} {
		_, err := hotkey.Parse(keys)
		if got := ParseErrorText(err); got != want {
			t.Errorf("%q: %q, want %q", keys, got, want)
		}
	}
	if ParseErrorText(errors.New("x")) == "" {
		t.Error("unknown error has no text")
	}
}

func TestHotkeyGlyphs(t *testing.T) {
	for keys, want := range map[string]string{
		"Fn": "fn", "Fn+Shift": "fn ⇧", "Fn+Control": "fn ⌃", "Control+Command+V": "⌃⌘V",
		"RightCommand": "Right ⌘", "RightOption+Shift": "Right ⌥ ⇧", "Option+Space": "⌥ Space", "": "—", "Command+Shift+k": "⇧⌘K",
	} {
		if got := HotkeyGlyphs(keys); got != want {
			t.Errorf("%q → %q, want %q", keys, got, want)
		}
	}
}
