//go:build windows

package main

import "testing"

func TestChordRecordKeyboard(t *testing.T) {
	var state chordRecordState
	if got := state.keyboard(0xa3, true); got.finished || !got.swallow {
		t.Fatalf("modifier down = %+v", got)
	}
	got := state.keyboard(0x20, true)
	if !got.finished || got.chord != "RightControl+Space" || !got.swallow {
		t.Fatalf("chord = %+v", got)
	}
}

func TestChordRecordModifiersFinishOnLastRelease(t *testing.T) {
	var state chordRecordState
	state.keyboard(0xa4, true)
	state.keyboard(0xa5, true)
	if got := state.keyboard(0xa4, false); got.finished {
		t.Fatalf("recording ended with one modifier held: %+v", got)
	}
	got := state.keyboard(0xa5, false)
	if !got.finished || got.chord != "LeftOption+RightOption" || !got.swallow {
		t.Fatalf("modifier-only chord = %+v", got)
	}
}

func TestChordRecordMouseAndKeyUp(t *testing.T) {
	var state chordRecordState
	state.keyboard(0xa2, true)
	if got := state.mouse(3, true); !got.finished || got.chord != "LeftControl+Mouse4" || !got.swallow {
		t.Fatalf("mouse chord = %+v", got)
	}
	if got := state.keyboard(0x41, false); got.finished || got.swallow {
		t.Fatalf("key up must pass through: %+v", got)
	}
	if got := state.mouse(0, true); got.finished || got.swallow {
		t.Fatalf("left click must pass through: %+v", got)
	}
}

func TestChordRecordUnknownKeyWaits(t *testing.T) {
	var state chordRecordState
	if got := state.keyboard(0xff, true); got.finished || !got.swallow || got.chord != "" {
		t.Fatalf("unknown key = %+v", got)
	}
	if got := recordedChord(modifierFn|modifierLeftControl|modifierRightCommand, hotkeyPress{virtualKey: 0x70}); got != "Fn+LeftControl+RightCommand+F1" {
		t.Fatalf("modifier order = %q", got)
	}
}
