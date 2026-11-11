package main

// Chords for the previous Voxt's shortcuts (meetingHotkeyBindings in its
// preferences), for Features.Meeting.Hotkeys on import.

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

const legacyModifierOnlyKeyCode = 0xffff

// nsModifierBits are NSEvent.ModifierFlags for shortcuts (macOS).
const (
	nsModShift   = 1 << 17
	nsModControl = 1 << 18
	nsModOption  = 1 << 19
	nsModCommand = 1 << 20
	nsModFn      = 1 << 23
)

// sidedModifier bits in Voxt 1.x SidedModifierFlags (HotkeySupport.swift).
const (
	sideLeftShift    = 1 << 0
	sideRightShift   = 1 << 1
	sideLeftControl  = 1 << 2
	sideRightControl = 1 << 3
	sideLeftOption   = 1 << 4
	sideRightOption  = 1 << 5
	sideLeftCommand  = 1 << 6
	sideRightCommand = 1 << 7
)

var legacyVirtualKeyNames = map[uint16]string{
	0: "A", 1: "S", 2: "D", 3: "F", 4: "H", 5: "G", 6: "Z", 7: "X", 8: "C", 9: "V", 11: "B", 12: "Q", 13: "W",
	14: "E", 15: "R", 16: "Y", 17: "T", 36: "Return", 49: "Space", 53: "Escape", 96: "F5", 97: "F6", 126: "Up",
}

type legacyHotkeyBinding struct {
	InputType         string `json:"inputType"`
	KeyCode           uint16 `json:"keyCode"`
	MouseButtonNumber int    `json:"mouseButtonNumber"`
	Modifiers         uint   `json:"modifiers"`
	SidedModifiers    int    `json:"sidedModifiers"`
	Behavior          string `json:"behavior"`
}

// parseLegacyMeetingHotkeys reads meetingHotkeyBindings, or the legacy single
// meeting shortcut fields, from the previous Voxt's preferences.
func parseLegacyMeetingHotkeys(voxtDir string) []settings.Hotkey {
	raw := strings.TrimSpace(legacyPreference(voxtDir, "meetingHotkeyBindings"))
	var bindings []legacyHotkeyBinding
	if raw != "" {
		json.Unmarshal([]byte(raw), &bindings)
	}
	if len(bindings) == 0 {
		bindings = []legacyHotkeyBinding{legacyMeetingHotkeyFromSingles(voxtDir)}
	}
	var out []settings.Hotkey
	for _, b := range bindings {
		keys, err := legacyChordFromBinding(b)
		if err != nil || strings.TrimSpace(keys) == "" {
			continue
		}
		if _, err := hotkey.Parse(keys); err != nil {
			continue
		}
		out = append(out, settings.Hotkey{Keys: keys, Trigger: legacyTrigger(b.Behavior)})
	}
	return out
}

func legacyMeetingHotkeyFromSingles(voxtDir string) legacyHotkeyBinding {
	b := legacyHotkeyBinding{Behavior: "tap"}
	if n, err := strconv.Atoi(strings.TrimSpace(legacyPreference(voxtDir, "meetingHotkeyKeyCode"))); err == nil && n >= 0 {
		b.KeyCode = uint16(n)
	}
	if n, err := strconv.Atoi(strings.TrimSpace(legacyPreference(voxtDir, "meetingHotkeyModifiers"))); err == nil && n >= 0 {
		b.Modifiers = uint(n)
	}
	if n, err := strconv.Atoi(strings.TrimSpace(legacyPreference(voxtDir, "meetingHotkeySidedModifiers"))); err == nil {
		b.SidedModifiers = n
	}
	b.InputType = legacyPreference(voxtDir, "meetingHotkeyInputType")
	return b
}

func legacyTrigger(behavior string) string {
	switch strings.TrimSpace(behavior) {
	case "longPress":
		return settings.TriggerLong
	case "doubleTap":
		return settings.TriggerDouble
	default:
		return settings.TriggerTap
	}
}

func legacyChordFromBinding(b legacyHotkeyBinding) (string, error) {
	var parts []string
	sided := b.SidedModifiers
	addSided := func(bit int, left, right string) {
		switch {
		case sided&bit != 0 && sided&(bit<<1) != 0:
			parts = append(parts, left, right)
		case sided&(bit<<1) != 0:
			parts = append(parts, right)
		case sided&bit != 0:
			parts = append(parts, left)
		}
	}
	addSided(sideLeftShift, "LeftShift", "RightShift")
	addSided(sideLeftControl, "LeftControl", "RightControl")
	addSided(sideLeftOption, "LeftOption", "RightOption")
	addSided(sideLeftCommand, "LeftCommand", "RightCommand")

	mods := int(b.Modifiers)
	if mods&nsModFn != 0 && !slicesContains(parts, "Fn") {
		parts = append(parts, "Fn")
	}
	if mods&nsModControl != 0 && !hasFamily(parts, "Control") {
		parts = append(parts, "Control")
	}
	if mods&nsModOption != 0 && !hasFamily(parts, "Option") {
		parts = append(parts, "Option")
	}
	if mods&nsModShift != 0 && !hasFamily(parts, "Shift") {
		parts = append(parts, "Shift")
	}
	if mods&nsModCommand != 0 && !hasFamily(parts, "Command") {
		parts = append(parts, "Command")
	}

	if strings.EqualFold(b.InputType, "mouseButton") || b.MouseButtonNumber >= 2 {
		n := b.MouseButtonNumber
		if n < 2 {
			n = 2
		}
		parts = append(parts, "Mouse"+strconv.Itoa(n+1))
	} else if b.KeyCode != legacyModifierOnlyKeyCode {
		name, ok := legacyVirtualKeyNames[b.KeyCode]
		if !ok {
			return "", fmt.Errorf("unknown key code %d", b.KeyCode)
		}
		parts = append(parts, name)
	}
	if len(parts) == 0 {
		return "", fmt.Errorf("empty chord")
	}
	return strings.Join(parts, "+"), nil
}

func slicesContains(parts []string, want string) bool {
	for _, p := range parts {
		if p == want {
			return true
		}
	}
	return false
}

func hasFamily(parts []string, family string) bool {
	for _, p := range parts {
		if p == family || strings.HasSuffix(p, family) {
			return true
		}
	}
	return false
}
