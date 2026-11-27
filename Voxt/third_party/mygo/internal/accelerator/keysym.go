package accelerator

import (
	"fmt"
	"strings"
)

// Keysym returns the XKB (and X11) name of the key, as in "a", "comma",
// "Return", "F5" or "KP_Add".
func (a Accelerator) Keysym() string {
	if name, ok := keysyms[a.Key]; ok {
		return name
	}
	if strings.HasPrefix(a.Key, "Num") && len(a.Key) == 4 {
		return "KP_" + a.Key[3:]
	}
	if r := []rune(a.Key); len(r) == 1 && r[0] > '~' {
		return fmt.Sprintf("U%04X", r[0])
	}
	return a.Key // letters, digits and F1-F24
}

// XDGTrigger formats the accelerator as a trigger of the XDG shortcuts
// specification, as in "CTRL+SHIFT+k", which desktop portals take.
func (a Accelerator) XDGTrigger() string {
	var parts []string
	for _, m := range []struct {
		mod  Modifiers
		name string
	}{{Ctrl, "CTRL"}, {Alt, "ALT"}, {Shift, "SHIFT"}, {Super, "LOGO"}} {
		if a.Has(m.mod) {
			parts = append(parts, m.name)
		}
	}
	return strings.Join(append(parts, a.Keysym()), "+")
}

var keysyms = map[string]string{
	"!": "exclam", `"`: "quotedbl", "#": "numbersign", "$": "dollar", "%": "percent",
	"&": "ampersand", "'": "apostrophe", "(": "parenleft", ")": "parenright", "*": "asterisk",
	"+": "plus", ",": "comma", "-": "minus", ".": "period", "/": "slash", ":": "colon",
	";": "semicolon", "<": "less", "=": "equal", ">": "greater", "?": "question", "@": "at",
	"[": "bracketleft", `\`: "backslash", "]": "bracketright", "^": "asciicircum",
	"_": "underscore", "`": "grave", "{": "braceleft", "|": "bar", "}": "braceright", "~": "asciitilde",
	"Enter": "Return", "Tab": "Tab", "Space": "space", "Backspace": "BackSpace",
	"Delete": "Delete", "Insert": "Insert", "Escape": "Escape",
	"Up": "Up", "Down": "Down", "Left": "Left", "Right": "Right",
	"Home": "Home", "End": "End", "PageUp": "Page_Up", "PageDown": "Page_Down",
	"VolumeUp": "XF86AudioRaiseVolume", "VolumeDown": "XF86AudioLowerVolume", "VolumeMute": "XF86AudioMute",
	"MediaNextTrack": "XF86AudioNext", "MediaPreviousTrack": "XF86AudioPrev",
	"MediaStop": "XF86AudioStop", "MediaPlayPause": "XF86AudioPlay",
	"PrintScreen": "Print", "CapsLock": "Caps_Lock", "NumLock": "Num_Lock", "ScrollLock": "Scroll_Lock",
	"NumDec": "KP_Decimal", "NumAdd": "KP_Add", "NumSub": "KP_Subtract",
	"NumMult": "KP_Multiply", "NumDiv": "KP_Divide",
}
