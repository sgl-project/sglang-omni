//go:build windows

package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

type chordRecordState struct {
	activeModifiers modifierMask
	peakModifiers   modifierMask
}

type chordRecordResult struct {
	chord    string
	finished bool
	swallow  bool
}

func (state *chordRecordState) keyboard(virtualKey uint16, down bool) chordRecordResult {
	if modifier, isModifier := modifierForVirtualKey(virtualKey); isModifier {
		previous := state.activeModifiers
		if down {
			state.activeModifiers |= modifier
		} else {
			state.activeModifiers &^= modifier
		}
		state.peakModifiers |= state.activeModifiers
		if !down && previous != 0 && state.activeModifiers == 0 {
			chord := recordedChord(state.peakModifiers, hotkeyPress{})
			return chordRecordResult{chord: chord, finished: chord != "", swallow: true}
		}
		return chordRecordResult{swallow: true}
	}
	if !down {
		return chordRecordResult{}
	}
	chord := recordedChord(state.activeModifiers, hotkeyPress{virtualKey: virtualKey})
	return chordRecordResult{chord: chord, finished: chord != "", swallow: true}
}

func (state *chordRecordState) mouse(button int, down bool) chordRecordResult {
	if !down || button < 2 {
		return chordRecordResult{}
	}
	chord := recordedChord(state.activeModifiers, hotkeyPress{button: button, mouse: true})
	return chordRecordResult{chord: chord, finished: chord != "", swallow: true}
}

func recordedChord(modifiers modifierMask, press hotkeyPress) string {
	parts := make([]string, 0, 10)
	for _, entry := range []struct {
		modifier modifierMask
		name     string
	}{
		{modifierFn, "Fn"},
		{modifierLeftControl, "LeftControl"}, {modifierRightControl, "RightControl"},
		{modifierLeftOption, "LeftOption"}, {modifierRightOption, "RightOption"},
		{modifierLeftShift, "LeftShift"}, {modifierRightShift, "RightShift"},
		{modifierLeftCommand, "LeftCommand"}, {modifierRightCommand, "RightCommand"},
	} {
		if modifiers&entry.modifier != 0 {
			parts = append(parts, entry.name)
		}
	}
	if press.mouse && press.button >= 2 {
		parts = append(parts, fmt.Sprintf("Mouse%d", press.button+1))
	} else if !press.mouse {
		if name := recordedKeyName(press.virtualKey); name != "" {
			parts = append(parts, name)
		}
	}
	return strings.Join(parts, "+")
}

func recordedKeyName(virtualKey uint16) string {
	switch {
	case virtualKey >= 0x41 && virtualKey <= 0x5a:
		return string(rune('A' + virtualKey - 0x41))
	case virtualKey >= 0x30 && virtualKey <= 0x39:
		return string(rune('0' + virtualKey - 0x30))
	case virtualKey >= 0x70 && virtualKey <= 0x83:
		return fmt.Sprintf("F%d", virtualKey-0x70+1)
	}
	names := map[uint16]string{
		0x08: "Delete", 0x09: "Tab", 0x0d: "Return", 0x1b: "Escape", 0x20: "Space",
		0x21: "PageUp", 0x22: "PageDown", 0x23: "End", 0x24: "Home",
		0x25: "Left", 0x26: "Up", 0x27: "Right", 0x28: "Down", 0x2e: "ForwardDelete",
		0xba: ";", 0xbb: "=", 0xbc: ",", 0xbd: "-", 0xbe: ".", 0xbf: "/", 0xc0: "`",
		0xdb: "[", 0xdc: "\\", 0xdd: "]", 0xde: "'",
	}
	return names[virtualKey]
}

type windowsHotkeyRecording struct {
	generation uint64
	chords     chordRecordState
	done       chan hotkeyRecordOutcome
	timer      *time.Timer
}

type hotkeyRecordOutcome struct {
	keys string
	err  *proto.Error
}

func (manager *windowsHotkeys) record(timeoutSeconds float64) (string, *proto.Error) {
	if timeoutSeconds <= 0 {
		timeoutSeconds = 15
	}
	if timeoutSeconds > float64((1<<63-1))/float64(time.Second) {
		return "", proto.Errorf(proto.CodeBadRequest, "Recording timeout is too large")
	}
	recording := &windowsHotkeyRecording{done: make(chan hotkeyRecordOutcome, 1)}
	started := manager.loop.callAndWait(func() {
		manager.recordGeneration++
		recording.generation = manager.recordGeneration
		if manager.recording != nil {
			manager.finishRecording(hotkeyRecordOutcome{err: recordError(proto.CodeCancelled, "Cancelled")})
		}
		manager.recording = recording
		if !manager.ensureRecordingHooks() {
			manager.finishRecording(hotkeyRecordOutcome{err: deniedRecordError()})
			return
		}
		timeout := time.Duration(timeoutSeconds * float64(time.Second))
		recording.timer = time.AfterFunc(timeout, func() {
			manager.loop.call(func() {
				if manager.recording == recording {
					manager.finishRecording(hotkeyRecordOutcome{err: recordError(proto.CodeTimeout, "Recording timed out")})
				}
			})
		})
	})
	if !started {
		return "", proto.Errorf(proto.CodeInternal, "The hotkey event loop stopped")
	}
	outcome := <-recording.done
	return outcome.keys, outcome.err
}

func (manager *windowsHotkeys) cancelRecord() {
	manager.loop.callAndWait(func() {
		if manager.recording != nil {
			manager.finishRecording(hotkeyRecordOutcome{err: recordError(proto.CodeCancelled, "Cancelled")})
		}
	})
}

func (manager *windowsHotkeys) finishRecording(outcome hotkeyRecordOutcome) {
	recording := manager.recording
	if recording == nil {
		return
	}
	manager.recording = nil
	if recording.timer != nil {
		recording.timer.Stop()
	}
	manager.releaseRecordingHooks()
	manager.recognizer.pressed = 0
	manager.physicalKeys = make(map[uint16]bool)
	manager.physicalModifiers = make(map[modifierMask]bool)
	recording.done <- outcome
}

func (manager *windowsHotkeys) ensureRecordingHooks() bool {
	module, _, _ := procGetModuleHandle.Call(0)
	if manager.keyboardHook == 0 {
		hook, _, _ := procSetWindowsHookEx.Call(hotkeyKeyboardLL, manager.keyboardCallback, module, 0)
		manager.keyboardHook = hook
	}
	if manager.mouseHook == 0 {
		hook, _, _ := procSetWindowsHookEx.Call(hotkeyMouseLL, manager.mouseCallback, module, 0)
		manager.mouseHook = hook
	}
	return manager.keyboardHook != 0 && manager.mouseHook != 0
}

func (manager *windowsHotkeys) releaseRecordingHooks() {
	needed := manager.recognizer.config.hasModifierChords() || manager.recognizer.config.cancel != nil
	for index := range manager.recognizer.config.bindings {
		if !eligibleForRegisteredHotkey(&manager.recognizer.config.bindings[index]) {
			needed = true
			break
		}
	}
	if needed {
		return
	}
	if manager.keyboardHook != 0 {
		procUnhookWindowsHookEx.Call(manager.keyboardHook)
		manager.keyboardHook = 0
	}
	if manager.mouseHook != 0 {
		procUnhookWindowsHookEx.Call(manager.mouseHook)
		manager.mouseHook = 0
	}
}

func recordError(code, message string) *proto.Error {
	return proto.Errorf(code, "%s", message)
}

func deniedRecordError() *proto.Error {
	data, _ := json.Marshal(map[string]string{"permission": proto.PermissionAccessibility})
	return &proto.Error{Code: proto.CodePermissionDenied, Message: "Accessibility is required to record shortcuts", Data: data}
}
