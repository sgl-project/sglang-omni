// Package accelerator parses Electron style keyboard accelerators such as
// "CmdOrCtrl+Shift+Z" into a platform neutral representation.
package accelerator

import (
	"fmt"
	"strings"
)

// Modifiers is a bit set of modifier keys.
type Modifiers uint8

// Modifier keys. Super is Command on macOS and the Windows/Super key
// elsewhere.
const (
	Super Modifiers = 1 << iota
	Ctrl
	Alt
	Shift
)

// Accelerator is a parsed accelerator.
type Accelerator struct {
	Modifiers Modifiers
	// Key is either a single lower case character ("a", "1", ",") or one of
	// the named keys: F1-F24, Enter, Tab, Space, Backspace, Delete, Insert,
	// Escape, Up, Down, Left, Right, Home, End, PageUp, PageDown,
	// VolumeUp, VolumeDown, VolumeMute, MediaNextTrack, MediaPreviousTrack,
	// MediaStop, MediaPlayPause, PrintScreen, Num0-Num9, NumDec, NumAdd,
	// NumSub, NumMult, NumDiv, CapsLock, NumLock, ScrollLock.
	Key string
}

// Has reports whether all modifiers in m are set.
func (a Accelerator) Has(m Modifiers) bool { return a.Modifiers&m == m }

// String formats the accelerator in canonical form.
func (a Accelerator) String() string {
	var parts []string
	if a.Has(Super) {
		parts = append(parts, "Super")
	}
	if a.Has(Ctrl) {
		parts = append(parts, "Ctrl")
	}
	if a.Has(Alt) {
		parts = append(parts, "Alt")
	}
	if a.Has(Shift) {
		parts = append(parts, "Shift")
	}
	key := a.Key
	if len(key) == 1 {
		key = strings.ToUpper(key)
	}
	return strings.Join(append(parts, key), "+")
}

var namedKeys = map[string]string{
	"plus":               "+",
	"space":              "Space",
	"tab":                "Tab",
	"capslock":           "CapsLock",
	"numlock":            "NumLock",
	"scrolllock":         "ScrollLock",
	"backspace":          "Backspace",
	"delete":             "Delete",
	"del":                "Delete",
	"insert":             "Insert",
	"ins":                "Insert",
	"return":             "Enter",
	"enter":              "Enter",
	"up":                 "Up",
	"down":               "Down",
	"left":               "Left",
	"right":              "Right",
	"home":               "Home",
	"end":                "End",
	"pageup":             "PageUp",
	"pagedown":           "PageDown",
	"escape":             "Escape",
	"esc":                "Escape",
	"volumeup":           "VolumeUp",
	"volumedown":         "VolumeDown",
	"volumemute":         "VolumeMute",
	"medianexttrack":     "MediaNextTrack",
	"mediaprevioustrack": "MediaPreviousTrack",
	"mediastop":          "MediaStop",
	"mediaplaypause":     "MediaPlayPause",
	"printscreen":        "PrintScreen",
	"numdec":             "NumDec",
	"numadd":             "NumAdd",
	"numsub":             "NumSub",
	"nummult":            "NumMult",
	"numdiv":             "NumDiv",
}

// Parse parses an accelerator for the given GOOS. "CmdOrCtrl" resolves to
// Super (Command) on darwin and Ctrl elsewhere.
func Parse(s, goos string) (Accelerator, error) {
	var a Accelerator
	if strings.TrimSpace(s) == "" {
		return a, fmt.Errorf("accelerator: empty accelerator")
	}
	parts := splitParts(s)
	for i, raw := range parts {
		p := strings.TrimSpace(raw)
		if p == "" {
			return a, fmt.Errorf("accelerator: invalid accelerator %q", s)
		}
		last := i == len(parts)-1
		if m, ok := modifier(p, goos); ok && !last {
			a.Modifiers |= m
			continue
		}
		if !last {
			return a, fmt.Errorf("accelerator: unknown modifier %q in %q", p, s)
		}
		key, err := parseKey(p)
		if err != nil {
			return a, fmt.Errorf("accelerator: %w in %q", err, s)
		}
		a.Key = key
	}
	return a, nil
}

// splitParts splits on "+" while allowing "+" itself as the key, as in
// "CmdOrCtrl++".
func splitParts(s string) []string {
	parts := strings.Split(s, "+")
	if strings.HasSuffix(s, "++") {
		parts = append(parts[:len(parts)-2], "+")
	} else if s == "+" {
		parts = []string{"+"}
	}
	return parts
}

func modifier(p, goos string) (Modifiers, bool) {
	switch strings.ToLower(p) {
	case "command", "cmd", "super", "meta", "win":
		return Super, true
	case "control", "ctrl":
		return Ctrl, true
	case "commandorcontrol", "cmdorctrl":
		if goos == "darwin" {
			return Super, true
		}
		return Ctrl, true
	case "alt", "option", "altgr":
		return Alt, true
	case "shift":
		return Shift, true
	}
	return 0, false
}

func parseKey(p string) (string, error) {
	lower := strings.ToLower(p)
	if len([]rune(p)) == 1 {
		return strings.ToLower(p), nil
	}
	if k, ok := namedKeys[lower]; ok {
		return k, nil
	}
	if len(lower) >= 2 && lower[0] == 'f' {
		var n int
		if _, err := fmt.Sscanf(lower[1:], "%d", &n); err == nil && n >= 1 && n <= 24 && fmt.Sprint(n) == lower[1:] {
			return fmt.Sprintf("F%d", n), nil
		}
	}
	if strings.HasPrefix(lower, "num") && len(lower) == 4 && lower[3] >= '0' && lower[3] <= '9' {
		return "Num" + lower[3:], nil
	}
	return "", fmt.Errorf("unknown key %q", p)
}
