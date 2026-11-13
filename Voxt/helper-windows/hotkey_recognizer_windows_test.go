//go:build windows

package main

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

func testHotkeyRecognizer(t *testing.T, bindings []proto.HotkeyBinding, active bool, cancel *proto.HotkeyBinding) *hotkeyRecognizer {
	t.Helper()
	config, err := makeRecognizerConfig(proto.HotkeyConfigureParams{Bindings: bindings, DoubleTapMS: 400, LongPressMS: 300, TapMaxMS: 1500, Active: active, Cancel: cancel}, 400)
	if err != nil {
		t.Fatal(err)
	}
	recognizer := newHotkeyRecognizer()
	recognizer.configure(config)
	return recognizer
}

func firedNames(events []hotkeyFired) []string {
	names := make([]string, len(events))
	for index, event := range events {
		names[index] = event.binding + ":" + event.phase
	}
	return names
}

func TestModifierChordSidesAccumulationAndSpoiling(t *testing.T) {
	bindings := []proto.HotkeyBinding{
		{ID: "right", Modifiers: map[string]string{proto.ModOption: proto.SideRight}, Behavior: proto.BehaviorTap},
		{ID: "both", Modifiers: map[string]string{proto.ModOption: proto.SideRight, proto.ModShift: proto.SideRight}, Behavior: proto.BehaviorTap},
	}
	recognizer := testHotkeyRecognizer(t, bindings, false, nil)
	if events := recognizer.modifier(0xa4, modifierLeftOption, 0).events; len(events) != 0 {
		t.Fatal(events)
	}
	if events := recognizer.modifier(0xa4, 0, 50).events; len(events) != 0 {
		t.Fatal(events)
	}
	recognizer.modifier(0xa5, modifierRightOption, 100)
	recognizer.modifier(0xa1, modifierRightOption|modifierRightShift, 130)
	recognizer.modifier(0xa5, modifierRightShift, 160)
	if got := firedNames(recognizer.modifier(0xa1, 0, 190).events); !slices.Equal(got, []string{"both:trigger"}) {
		t.Fatalf("accumulated chord %v", got)
	}
	recognizer.modifier(0xa5, modifierRightOption, 300)
	recognizer.keyDown(hotkeyPress{virtualKey: 0x45}, false, 320)
	if events := recognizer.modifier(0xa5, 0, 350).events; len(events) != 0 {
		t.Fatalf("typing did not spoil the chord: %v", events)
	}
}

func TestModifierDoubleTapAndLongPress(t *testing.T) {
	bindings := []proto.HotkeyBinding{
		{ID: "tap", Modifiers: map[string]string{proto.ModOption: proto.SideRight}, Behavior: proto.BehaviorTap},
		{ID: "double", Modifiers: map[string]string{proto.ModOption: proto.SideRight}, Behavior: proto.BehaviorDoubleTap},
	}
	recognizer := testHotkeyRecognizer(t, bindings, false, nil)
	recognizer.modifier(0xa5, modifierRightOption, 0)
	recognizer.modifier(0xa5, 0, 60)
	if events := recognizer.tick(459); len(events) != 0 {
		t.Fatal(events)
	}
	if got := firedNames(recognizer.tick(461)); !slices.Equal(got, []string{"tap:trigger"}) {
		t.Fatalf("deferred tap %v", got)
	}
	recognizer.modifier(0xa5, modifierRightOption, 500)
	recognizer.modifier(0xa5, 0, 560)
	recognizer.modifier(0xa5, modifierRightOption, 700)
	if got := firedNames(recognizer.modifier(0xa5, 0, 760).events); !slices.Equal(got, []string{"double:trigger"}) {
		t.Fatalf("double tap %v", got)
	}

	long := testHotkeyRecognizer(t, []proto.HotkeyBinding{{ID: "hold", Modifiers: map[string]string{proto.ModCommand: proto.SideRight}, Behavior: proto.BehaviorLongPress}}, false, nil)
	long.modifier(0x5c, modifierRightCommand, 0)
	if events := long.tick(299); len(events) != 0 {
		t.Fatal(events)
	}
	if got := firedNames(long.tick(300)); !slices.Equal(got, []string{"hold:down"}) {
		t.Fatalf("long press down %v", got)
	}
	if got := firedNames(long.modifier(0x5c, 0, 1000).events); !slices.Equal(got, []string{"hold:up"}) {
		t.Fatalf("long press up %v", got)
	}
}

func TestKeySwallowCancelAndFnFlaggedKey(t *testing.T) {
	key := 9
	binding := proto.HotkeyBinding{ID: "paste", Modifiers: map[string]string{proto.ModControl: proto.SideAny, proto.ModCommand: proto.SideAny}, Key: &key, Behavior: proto.BehaviorTap}
	recognizer := testHotkeyRecognizer(t, []proto.HotkeyBinding{binding}, false, nil)
	recognizer.pressed = modifierLeftControl | modifierLeftCommand
	press := hotkeyPress{virtualKey: 0x56}
	down := recognizer.keyDown(press, false, 0)
	if !down.swallow || !slices.Equal(firedNames(down.events), []string{"paste:trigger"}) || !recognizer.keyUp(press, 20).swallow {
		t.Fatalf("key down/up was not swallowed: %+v", down)
	}

	escape := 53
	cancel := &proto.HotkeyBinding{ID: "cancel", Modifiers: map[string]string{}, Key: &escape, Behavior: proto.BehaviorTap}
	active := testHotkeyRecognizer(t, nil, true, cancel)
	escapePress := hotkeyPress{virtualKey: 0x1b}
	cancelDown := active.keyDown(escapePress, false, 0)
	if !cancelDown.swallow || !slices.Equal(firedNames(cancelDown.events), []string{"cancel:trigger"}) {
		t.Fatalf("active cancel %+v", cancelDown)
	}
	active.keyUp(escapePress, 20)
	config := active.config
	config.active = false
	active.configure(config)
	if idle := active.keyDown(escapePress, false, 30); idle.swallow || len(idle.events) != 0 {
		t.Fatalf("inactive cancel %+v", idle)
	}

	f5 := 96
	functionBinding := proto.HotkeyBinding{ID: "function", Modifiers: map[string]string{proto.ModFn: proto.SideAny}, Key: &f5, Behavior: proto.BehaviorTap}
	function := testHotkeyRecognizer(t, []proto.HotkeyBinding{functionBinding}, false, nil)
	function.pressed = modifierFn
	if got := function.keyDown(hotkeyPress{virtualKey: 0x74}, false, 0); len(got.events) != 1 {
		t.Fatalf("F5 with Fn flag %+v", got)
	}
	if _, retained := function.config.bindings[0].Modifiers[proto.ModFn]; !retained {
		t.Fatal("matching an Fn-flagged key mutated the binding")
	}
}

func TestMouseBindingAndChordSpoiling(t *testing.T) {
	button := 3
	bindings := []proto.HotkeyBinding{
		{ID: "option", Modifiers: map[string]string{proto.ModOption: proto.SideLeft}, Behavior: proto.BehaviorTap},
		{ID: "button", Modifiers: map[string]string{proto.ModOption: proto.SideLeft}, Button: &button, Behavior: proto.BehaviorTap},
	}
	recognizer := testHotkeyRecognizer(t, bindings, false, nil)
	recognizer.modifier(0xa4, modifierLeftOption, 0)
	press := hotkeyPress{button: 3, mouse: true}
	down := recognizer.keyDown(press, false, 50)
	if !down.swallow || !slices.Equal(firedNames(down.events), []string{"button:trigger"}) {
		t.Fatalf("mouse binding %+v", down)
	}
	if !recognizer.keyUp(press, 80).swallow {
		t.Fatal("mouse release was not swallowed")
	}
	if events := recognizer.modifier(0xa4, 0, 100).events; len(events) != 0 {
		t.Fatalf("mouse click did not spoil modifier tap: %v", events)
	}
}
