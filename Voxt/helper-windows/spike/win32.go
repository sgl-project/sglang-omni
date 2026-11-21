//go:build windows

package main

import (
	"fmt"
	"strings"
	"syscall"
	"time"
	"unicode/utf16"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	clipboardUnicodeText = 13
	clipboardCantOpen    = 0x800401d0
	clipboardCantClose   = 0x800401d4
	globalMoveable       = 0x0002
	inputKeyboard        = 1
	keyEventKeyUp        = 0x0002
	keyEventUnicode      = 0x0004
	peekSentMessage      = 0x00400000
	queueSendMessage     = 0x0040
	virtualKeyControl    = 0x11
	virtualKeyEnd        = 0x23
	virtualKeyS          = 0x53
	virtualKeyV          = 0x56
	windowMessageClose   = 0x0010
)

var (
	kernel32 = windows.NewLazySystemDLL("kernel32.dll")
	ole32    = windows.NewLazySystemDLL("ole32.dll")
	oleaut32 = windows.NewLazySystemDLL("oleaut32.dll")
	user32   = windows.NewLazySystemDLL("user32.dll")

	procCloseClipboard       = user32.NewProc("CloseClipboard")
	procCoCreateInstance     = ole32.NewProc("CoCreateInstance")
	procCoInitializeEx       = ole32.NewProc("CoInitializeEx")
	procEmptyClipboard       = user32.NewProc("EmptyClipboard")
	procGetClipboardSequence = user32.NewProc("GetClipboardSequenceNumber")
	procGetWindowText        = user32.NewProc("GetWindowTextW")
	procGlobalAlloc          = kernel32.NewProc("GlobalAlloc")
	procGlobalFree           = kernel32.NewProc("GlobalFree")
	procGlobalLock           = kernel32.NewProc("GlobalLock")
	procGlobalUnlock         = kernel32.NewProc("GlobalUnlock")
	procIsWindow             = user32.NewProc("IsWindow")
	procIsWindowVisible      = user32.NewProc("IsWindowVisible")
	procMsgWaitForMessages   = user32.NewProc("MsgWaitForMultipleObjectsEx")
	procOleGetClipboard      = ole32.NewProc("OleGetClipboard")
	procOleInitialize        = ole32.NewProc("OleInitialize")
	procOleSetClipboard      = ole32.NewProc("OleSetClipboard")
	procOleUninitialize      = ole32.NewProc("OleUninitialize")
	procOpenClipboard        = user32.NewProc("OpenClipboard")
	procPeekMessage          = user32.NewProc("PeekMessageW")
	procPostMessage          = user32.NewProc("PostMessageW")
	procRtlMoveMemory        = kernel32.NewProc("RtlMoveMemory")
	procSendInput            = user32.NewProc("SendInput")
	procSetClipboardData     = user32.NewProc("SetClipboardData")
	procSetForegroundWindow  = user32.NewProc("SetForegroundWindow")
	procSysFreeString        = oleaut32.NewProc("SysFreeString")
	procSysStringLen         = oleaut32.NewProc("SysStringLen")
)

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

type windowSearch struct {
	handle    uintptr
	processID uint32
}

func hresultError(operation string, result uintptr) error {
	return fmt.Errorf("%s: HRESULT 0x%08x", operation, uint32(result))
}

func initializeOLE() error {
	result, _, _ := procOleInitialize.Call(0)
	if int32(result) < 0 {
		return hresultError("OleInitialize", result)
	}
	return nil
}

func waitForWindow(titlePart string, timeout time.Duration) (uintptr, uint32, error) {
	deadline := time.Now().Add(timeout)
	for {
		var search windowSearch
		callback := syscall.NewCallback(func(handle uintptr, _ uintptr) uintptr {
			visible, _, _ := procIsWindowVisible.Call(handle)
			if visible == 0 || !strings.Contains(windowTitle(handle), titlePart) {
				return 1
			}
			var processID uint32
			_, _ = windows.GetWindowThreadProcessId(windows.HWND(handle), &processID)
			search.handle = handle
			search.processID = processID
			return 1
		})
		_ = windows.EnumWindows(callback, nil)
		if search.handle != 0 {
			return search.handle, search.processID, nil
		}
		if time.Now().After(deadline) {
			return 0, 0, fmt.Errorf("window containing %q did not appear within %s", titlePart, timeout)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func windowTitle(handle uintptr) string {
	buffer := make([]uint16, 512)
	count, _, _ := procGetWindowText.Call(handle, uintptr(unsafe.Pointer(&buffer[0])), uintptr(len(buffer)))
	return windows.UTF16ToString(buffer[:count])
}

func closeWindow(handle uintptr) {
	procPostMessage.Call(handle, windowMessageClose, 0, 0)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		exists, _, _ := procIsWindow.Call(handle)
		if exists == 0 {
			return
		}
		time.Sleep(50 * time.Millisecond)
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

func keyEvent(virtualKey uint16, scanCode uint16, flags uint32) keyboardEvent {
	return keyboardEvent{
		kind: inputKeyboard,
		keyboard: keyboardInput{
			virtualKey: virtualKey,
			scanCode:   scanCode,
			flags:      flags,
		},
	}
}

func sendKeyWithControl(virtualKey uint16) error {
	return sendKeyboardEvents([]keyboardEvent{
		keyEvent(virtualKeyControl, 0, 0),
		keyEvent(virtualKey, 0, 0),
		keyEvent(virtualKey, 0, keyEventKeyUp),
		keyEvent(virtualKeyControl, 0, keyEventKeyUp),
	})
}

func typeText(text string) error {
	var events []keyboardEvent
	for _, codeUnit := range utf16.Encode([]rune(text)) {
		events = append(events,
			keyEvent(0, codeUnit, keyEventUnicode),
			keyEvent(0, codeUnit, keyEventUnicode|keyEventKeyUp),
		)
	}
	return sendKeyboardEvents(events)
}

type clipboardSnapshot struct {
	dataObject  *comObject
	ourSequence uint32
}

func snapshotAndSetClipboard(text string) (*clipboardSnapshot, error) {
	var dataObject *comObject
	result, _, _ := procOleGetClipboard.Call(uintptr(unsafe.Pointer(&dataObject)))
	if int32(result) < 0 {
		return nil, hresultError("OleGetClipboard", result)
	}
	snapshot := &clipboardSnapshot{dataObject: dataObject}
	if err := setClipboardText(text); err != nil {
		setClipboardObject(dataObject)
		dataObject.release()
		return nil, err
	}
	sequence, _, _ := procGetClipboardSequence.Call()
	snapshot.ourSequence = uint32(sequence)
	return snapshot, nil
}

func setClipboardText(text string) error {
	opened := false
	for attempt := 0; attempt < 25; attempt++ {
		result, _, _ := procOpenClipboard.Call(0)
		if result != 0 {
			opened = true
			break
		}
		waitForClipboardMessage()
	}
	if !opened {
		return fmt.Errorf("OpenClipboard failed after 250 ms")
	}
	defer procCloseClipboard.Call()
	result, _, callErr := procEmptyClipboard.Call()
	if result == 0 {
		return fmt.Errorf("EmptyClipboard: %w", callErr)
	}

	encoded := append(utf16.Encode([]rune(text)), 0)
	memory, _, callErr := procGlobalAlloc.Call(globalMoveable, uintptr(len(encoded))*2)
	if memory == 0 {
		return fmt.Errorf("GlobalAlloc: %w", callErr)
	}
	destination, _, callErr := procGlobalLock.Call(memory)
	if destination == 0 {
		procGlobalFree.Call(memory)
		return fmt.Errorf("GlobalLock: %w", callErr)
	}
	procRtlMoveMemory.Call(destination, uintptr(unsafe.Pointer(&encoded[0])), uintptr(len(encoded))*2)
	procGlobalUnlock.Call(memory)
	result, _, callErr = procSetClipboardData.Call(clipboardUnicodeText, memory)
	if result == 0 {
		procGlobalFree.Call(memory)
		return fmt.Errorf("SetClipboardData: %w", callErr)
	}
	return nil
}

func waitForClipboardMessage() {
	procMsgWaitForMessages.Call(0, 0, 10, queueSendMessage, 0)
	var message [8]uint64
	procPeekMessage.Call(uintptr(unsafe.Pointer(&message[0])), 0, 0, 0, peekSentMessage)
}

func setClipboardObject(dataObject *comObject) uintptr {
	deadline := time.Now().Add(time.Second)
	for {
		result, _, _ := procOleSetClipboard.Call(uintptr(unsafe.Pointer(dataObject)))
		if int32(result) >= 0 || uint32(result) != clipboardCantOpen && uint32(result) != clipboardCantClose || time.Now().After(deadline) {
			return result
		}
		waitForClipboardMessage()
	}
}

func (snapshot *clipboardSnapshot) restore() (restored bool, taken bool, err error) {
	if snapshot == nil {
		return false, false, nil
	}
	defer snapshot.dataObject.release()
	sequence, _, _ := procGetClipboardSequence.Call()
	if uint32(sequence) != snapshot.ourSequence {
		return false, true, nil
	}
	result := setClipboardObject(snapshot.dataObject)
	if int32(result) < 0 {
		return false, false, hresultError("OleSetClipboard", result)
	}
	return true, false, nil
}

func readBSTR(address *uint16) string {
	if address == nil {
		return ""
	}
	defer procSysFreeString.Call(uintptr(unsafe.Pointer(address)))
	length, _, _ := procSysStringLen.Call(uintptr(unsafe.Pointer(address)))
	return windows.UTF16ToString(unsafe.Slice(address, int(length)))
}
