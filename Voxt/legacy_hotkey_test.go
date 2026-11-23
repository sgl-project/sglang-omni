package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestLegacyChordFromBinding(t *testing.T) {
	chord, err := legacyChordFromBinding(legacyHotkeyBinding{
		InputType: "keyboard", KeyCode: 96, Modifiers: nsModCommand, Behavior: "longPress",
	})
	if err != nil || chord != "Command+F5" {
		t.Fatalf("%q %v", chord, err)
	}
	if _, err := hotkey.Parse(chord); err != nil {
		t.Fatal(err)
	}
	if legacyTrigger("longPress") != settings.TriggerLong {
		t.Fatal()
	}
}
