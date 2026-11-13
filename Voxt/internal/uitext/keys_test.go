package uitext

import (
	"errors"
	"runtime"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
)

func TestHotkeyText(t *testing.T) {
	cases := map[string]string{
		"Fn":                   "fn",
		"Fn+Shift":             "fn ⇧",
		"RightCommand":         "Right ⌘",
		"RightOption":          "Right ⌥",
		"Control+Command+V":    "⌃⌘V",
		"Control+Option+Space": "⌃⌥Space",
		"Option+Space":         "⌥Space",
		"Escape":               "Esc",
		"Return":               "Return",
		"Control+Delete":       "⌃Delete",
		"Option+Tab":           "⌥Tab",
		"Fn+Space":             "fn Space",
		"Control+RightOption":  "⌃ Right ⌥",
		"Option+Shift":         "⌥⇧",
		"Fn+F5":                "fn F5",
		"Mouse3":               "Middle mouse button",
		"Option+Mouse4":        "⌥ Mouse button 4",
		"bogus":                "bogus",
	}
	if runtime.GOOS == "windows" {
		for keys, want := range map[string]string{
			"Fn+Shift": "fn Shift", "RightCommand": "Right Win", "RightOption": "Right Alt",
			"Control+Command+V": "Ctrl Win V", "Control+Option+Space": "Ctrl Alt Space", "Option+Space": "Alt Space",
			"Control+Delete": "Ctrl Delete", "Option+Tab": "Alt Tab", "Control+RightOption": "Ctrl Right Alt",
			"Option+Shift": "Alt Shift", "Option+Mouse4": "Alt Mouse button 4",
		} {
			cases[keys] = want
		}
	}
	for keys, want := range cases {
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
		"Hyper+K":  "Unknown key. Type the modifiers first, then one key, as in Fn, RightCommand, Control+Option+Space or Fn+F5.",
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
	cases := map[string]string{
		"Fn": "fn", "Fn+Shift": "fn ⇧", "Fn+Control": "fn ⌃", "Control+Command+V": "⌃⌘V",
		"RightCommand": "Right ⌘", "RightOption+Shift": "Right ⌥ ⇧", "Option+Space": "⌥Space", "": "—", "Command+Shift+k": "⇧⌘K",
	}
	if runtime.GOOS == "windows" {
		for keys, want := range map[string]string{
			"Fn+Shift": "fn Shift", "Fn+Control": "fn Ctrl", "Control+Command+V": "Ctrl Win V",
			"RightCommand": "Right Win", "RightOption+Shift": "Right Alt Shift", "Option+Space": "Alt Space", "Command+Shift+k": "Shift Win K",
		} {
			cases[keys] = want
		}
	}
	for keys, want := range cases {
		if got := HotkeyGlyphs(keys); got != want {
			t.Errorf("%q → %q, want %q", keys, got, want)
		}
	}
}
