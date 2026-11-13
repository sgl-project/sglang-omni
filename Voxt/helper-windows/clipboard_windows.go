//go:build windows

package main

import (
	"fmt"
	"time"
	"unicode/utf16"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	clipboardUnicodeText = 13
	globalMoveable       = 0x0002
)

var (
	kernel32                       = windows.NewLazySystemDLL("kernel32.dll")
	procCloseClipboard             = user32.NewProc("CloseClipboard")
	procEmptyClipboard             = user32.NewProc("EmptyClipboard")
	procGetClipboardData           = user32.NewProc("GetClipboardData")
	procGetClipboardSequence       = user32.NewProc("GetClipboardSequenceNumber")
	procGlobalAlloc                = kernel32.NewProc("GlobalAlloc")
	procGlobalFree                 = kernel32.NewProc("GlobalFree")
	procGlobalLock                 = kernel32.NewProc("GlobalLock")
	procGlobalSize                 = kernel32.NewProc("GlobalSize")
	procGlobalUnlock               = kernel32.NewProc("GlobalUnlock")
	procIsClipboardFormatAvailable = user32.NewProc("IsClipboardFormatAvailable")
	procOleGetClipboard            = ole32.NewProc("OleGetClipboard")
	procOleSetClipboard            = ole32.NewProc("OleSetClipboard")
	procOpenClipboard              = user32.NewProc("OpenClipboard")
	procRtlMoveMemory              = kernel32.NewProc("RtlMoveMemory")
	procSetClipboardData           = user32.NewProc("SetClipboardData")
)

type clipboardSnapshot struct {
	dataObject   *comObject
	ourSequence  uint32
	originalText string
	hasText      bool
}

func snapshotClipboard() (*clipboardSnapshot, error) {
	snapshot := &clipboardSnapshot{}
	dataObject, err := clipboardObject()
	if err != nil {
		return nil, err
	}
	snapshot.dataObject = dataObject
	available, _, _ := procIsClipboardFormatAvailable.Call(clipboardUnicodeText)
	if available != 0 {
		snapshot.originalText, err = clipboardText()
		if err != nil {
			snapshot.discard()
			return nil, err
		}
		snapshot.hasText = true
	}
	return snapshot, nil
}

func snapshotAndSetClipboard(text string, preserve bool) (*clipboardSnapshot, error) {
	snapshot := &clipboardSnapshot{}
	if preserve {
		var err error
		snapshot, err = snapshotClipboard()
		if err != nil {
			return nil, err
		}
	}
	if err := setClipboardText(text); err != nil {
		if preserve {
			_ = snapshot.writeBack()
			snapshot.discard()
		}
		return nil, err
	}
	sequence, _, _ := procGetClipboardSequence.Call()
	snapshot.ourSequence = uint32(sequence)
	return snapshot, nil
}

func setClipboardText(text string) error {
	return writeClipboardText(text, true)
}

func writeClipboardText(text string, empty bool) error {
	if err := openClipboard(); err != nil {
		return err
	}
	defer procCloseClipboard.Call()
	if empty {
		result, _, callErr := procEmptyClipboard.Call()
		if result == 0 {
			return fmt.Errorf("EmptyClipboard: %w", callErr)
		}
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
	result, _, callErr := procSetClipboardData.Call(clipboardUnicodeText, memory)
	if result == 0 {
		procGlobalFree.Call(memory)
		return fmt.Errorf("SetClipboardData: %w", callErr)
	}
	return nil
}

func openClipboard() error {
	for attempt := 0; attempt < 25; attempt++ {
		result, _, _ := procOpenClipboard.Call(0)
		if result != 0 {
			return nil
		}
		waitForClipboardMessage()
	}
	return fmt.Errorf("OpenClipboard failed after 250 ms")
}

func clipboardText() (string, error) {
	if err := openClipboard(); err != nil {
		return "", err
	}
	defer procCloseClipboard.Call()
	memory, _, callErr := procGetClipboardData.Call(clipboardUnicodeText)
	if memory == 0 {
		return "", fmt.Errorf("GetClipboardData(CF_UNICODETEXT): %w", callErr)
	}
	address, _, callErr := procGlobalLock.Call(memory)
	if address == 0 {
		return "", fmt.Errorf("GlobalLock: %w", callErr)
	}
	byteCount, _, _ := procGlobalSize.Call(memory)
	if byteCount < 2 {
		procGlobalUnlock.Call(memory)
		return "", nil
	}
	encoded := make([]uint16, byteCount/2)
	procRtlMoveMemory.Call(uintptr(unsafe.Pointer(&encoded[0])), address, byteCount)
	procGlobalUnlock.Call(memory)
	return windows.UTF16ToString(encoded), nil
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

func (snapshot *clipboardSnapshot) discard() {
	if snapshot != nil && snapshot.dataObject != nil {
		snapshot.dataObject.release()
		snapshot.dataObject = nil
	}
}

func (snapshot *clipboardSnapshot) writeBack() error {
	if snapshot.dataObject != nil {
		result := setClipboardObject(snapshot.dataObject)
		if int32(result) < 0 {
			return hresultError("OleSetClipboard", result)
		}
	} else {
		if err := openClipboard(); err != nil {
			return err
		}
		result, _, callErr := procEmptyClipboard.Call()
		procCloseClipboard.Call()
		if result == 0 {
			return callErr
		}
	}
	if snapshot.hasText {
		return writeClipboardText(snapshot.originalText, false)
	}
	return nil
}

func (snapshot *clipboardSnapshot) restore() (restored bool, taken bool, err error) {
	if snapshot == nil {
		return false, false, nil
	}
	defer snapshot.discard()
	sequence, _, _ := procGetClipboardSequence.Call()
	if uint32(sequence) != snapshot.ourSequence {
		return false, true, nil
	}
	if err := snapshot.writeBack(); err != nil {
		return false, false, err
	}
	return true, false, nil
}
