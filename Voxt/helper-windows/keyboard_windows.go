//go:build windows

package main

import (
	"fmt"
	"unsafe"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

const (
	inputKeyboard     = 1
	keyEventKeyUp     = 0x0002
	helperInputMarker = 0x564F5854
	virtualKeyControl = 0x11
	virtualKeyC       = 0x43
	virtualKeyV       = 0x56
)

var procSendInput = user32.NewProc("SendInput")

type keyboardInput struct {
	virtualKey uint16
	scanCode   uint16
	flags      uint32
	timeStamp  uint32
	extraInfo  uintptr
}

type keyboardEvent struct {
	kind         uint32
	keyboard     keyboardInput
	mousePadding [8]byte
}

var windowsKeyCodes = map[int]uint16{
	0: 0x41, 1: 0x53, 2: 0x44, 3: 0x46, 4: 0x48, 5: 0x47, 6: 0x5a, 7: 0x58, 8: 0x43, 9: 0x56,
	11: 0x42, 12: 0x51, 13: 0x57, 14: 0x45, 15: 0x52, 16: 0x59, 17: 0x54,
	18: 0x31, 19: 0x32, 20: 0x33, 21: 0x34, 22: 0x36, 23: 0x35, 24: 0xbb, 25: 0x39,
	26: 0x37, 27: 0xbd, 28: 0x38, 29: 0x30, 30: 0xdd, 31: 0x4f, 32: 0x55, 33: 0xdb,
	34: 0x49, 35: 0x50, 36: 0x0d, 37: 0x4c, 38: 0x4a, 39: 0xde, 40: 0x4b, 41: 0xba,
	42: 0xdc, 43: 0xbc, 44: 0xbf, 45: 0x4e, 46: 0x4d, 47: 0xbe, 48: 0x09, 49: 0x20,
	50: 0xc0, 51: 0x08, 53: 0x1b, 64: 0x80, 79: 0x81, 80: 0x82, 90: 0x83,
	96: 0x74, 97: 0x75, 98: 0x76, 99: 0x72, 100: 0x77, 101: 0x78, 103: 0x7a,
	105: 0x7c, 106: 0x7f, 107: 0x7d, 109: 0x79, 111: 0x7b, 113: 0x7e,
	115: 0x24, 116: 0x21, 117: 0x2e, 118: 0x73, 119: 0x23, 120: 0x71, 121: 0x22,
	122: 0x70, 123: 0x25, 124: 0x27, 125: 0x28, 126: 0x26,
}

func windowsVirtualKey(macKeyCode int) (uint16, error) {
	virtualKey, found := windowsKeyCodes[macKeyCode]
	if !found {
		return 0, fmt.Errorf("unknown Mac virtual key code %d", macKeyCode)
	}
	return virtualKey, nil
}

func modifierVirtualKeys(modifiers []string) ([]uint16, error) {
	virtualKeys := make([]uint16, 0, len(modifiers))
	for _, modifier := range modifiers {
		switch modifier {
		case proto.ModControl:
			virtualKeys = append(virtualKeys, 0xa2)
		case proto.ModOption:
			virtualKeys = append(virtualKeys, 0xa4)
		case proto.ModShift:
			virtualKeys = append(virtualKeys, 0xa0)
		case proto.ModCommand:
			virtualKeys = append(virtualKeys, 0x5b)
		default:
			return nil, fmt.Errorf("unsupported key modifier %q", modifier)
		}
	}
	return virtualKeys, nil
}

func keyEvent(virtualKey uint16, flags uint32) keyboardEvent {
	return keyboardEvent{
		kind: inputKeyboard,
		keyboard: keyboardInput{
			virtualKey: virtualKey,
			flags:      flags,
			extraInfo:  helperInputMarker,
		},
	}
}

func sendKeyboardEvents(events []keyboardEvent) error {
	if len(events) == 0 {
		return nil
	}
	sent, _, callErr := procSendInput.Call(
		uintptr(len(events)),
		uintptr(unsafe.Pointer(&events[0])),
		unsafe.Sizeof(events[0]),
	)
	if sent != uintptr(len(events)) {
		return fmt.Errorf("SendInput sent %d of %d events: %w", sent, len(events), callErr)
	}
	return nil
}

func sendKeyCombo(virtualKey uint16, modifiers []string) error {
	modifierKeys, err := modifierVirtualKeys(modifiers)
	if err != nil {
		return err
	}
	events := make([]keyboardEvent, 0, 2*len(modifierKeys)+2)
	for _, modifierKey := range modifierKeys {
		events = append(events, keyEvent(modifierKey, 0))
	}
	events = append(events, keyEvent(virtualKey, 0), keyEvent(virtualKey, keyEventKeyUp))
	for index := len(modifierKeys) - 1; index >= 0; index-- {
		events = append(events, keyEvent(modifierKeys[index], keyEventKeyUp))
	}
	return sendKeyboardEvents(events)
}

func validateKeyPresses(presses []proto.KeyPress) error {
	for _, press := range presses {
		if _, err := windowsVirtualKey(press.Key); err != nil {
			return err
		}
		if _, err := modifierVirtualKeys(press.Modifiers); err != nil {
			return err
		}
	}
	return nil
}
