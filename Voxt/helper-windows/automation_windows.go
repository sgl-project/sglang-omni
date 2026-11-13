//go:build windows

package main

import (
	"fmt"
	"runtime"
	"sync"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	comRelease                   = 2
	automationCompareElements    = 3
	automationFocusedElement     = 8
	elementCurrentPatternAs      = 14
	elementCurrentProcessID      = 20
	elementCurrentControlType    = 21
	elementCurrentIsPassword     = 35
	elementCurrentFramework      = 40
	patternValue                 = 10002
	patternText                  = 10014
	valuePatternCurrentValue     = 4
	valuePatternCurrentReadOnly  = 5
	textPatternGetSelection      = 5
	textPatternDocumentRange     = 7
	textRangeClone               = 3
	textRangeCompareEndpoints    = 5
	textRangeGetText             = 12
	textRangeMoveEndpointByRange = 15
	arrayLength                  = 3
	arrayElement                 = 4
	textPatternEndpointStart     = 0
	textPatternEndpointEnd       = 1
	clipboardCantOpen            = 0x800401d0
	clipboardCantClose           = 0x800401d4
	peekSentMessage              = 0x00400000
	queueSendMessage             = 0x0040
	maxFieldTextUnits            = 200000
)

var (
	ole32                 = windows.NewLazySystemDLL("ole32.dll")
	oleaut32              = windows.NewLazySystemDLL("oleaut32.dll")
	procCoCreateInstance  = ole32.NewProc("CoCreateInstance")
	procOleInitialize     = ole32.NewProc("OleInitialize")
	procOleUninitialize   = ole32.NewProc("OleUninitialize")
	procSysFreeString     = oleaut32.NewProc("SysFreeString")
	procSysStringLen      = oleaut32.NewProc("SysStringLen")
	procMsgWaitForMessage = user32.NewProc("MsgWaitForMultipleObjectsEx")
	procPeekMessage       = user32.NewProc("PeekMessageW")

	classAutomation       = windows.GUID{Data1: 0xff48dba4, Data2: 0x60ef, Data3: 0x4201, Data4: [8]byte{0xaa, 0x87, 0x54, 0x10, 0x3e, 0xef, 0x59, 0x4e}}
	interfaceAutomation   = windows.GUID{Data1: 0x30cbe57d, Data2: 0xd9d0, Data3: 0x452a, Data4: [8]byte{0xab, 0x13, 0x7a, 0xc5, 0xac, 0x48, 0x25, 0xee}}
	interfaceValuePattern = windows.GUID{Data1: 0xa94cd8b1, Data2: 0x0844, Data3: 0x4cd6, Data4: [8]byte{0x9d, 0x2d, 0x64, 0x05, 0x37, 0xab, 0x39, 0xe9}}
	interfaceTextPattern  = windows.GUID{Data1: 0x32eba289, Data2: 0x3583, Data3: 0x42c9, Data4: [8]byte{0x9c, 0x59, 0x3b, 0x6d, 0x9a, 0x1e, 0x9b, 0x6a}}
)

type comObject struct {
	vtable *[64]uintptr
}

func (object *comObject) call(method int, arguments ...uintptr) uintptr {
	if object == nil {
		return 0
	}
	parameters := append([]uintptr{uintptr(unsafe.Pointer(object))}, arguments...)
	result, _, _ := syscall.SyscallN(object.vtable[method], parameters...)
	return result
}

func (object *comObject) release() {
	object.call(comRelease)
}

func (object *comObject) output(method int, arguments ...uintptr) (*comObject, error) {
	var output *comObject
	arguments = append(arguments, uintptr(unsafe.Pointer(&output)))
	result := object.call(method, arguments...)
	if int32(result) < 0 {
		return nil, hresultError("UIA output", result)
	}
	if output == nil {
		return nil, fmt.Errorf("UIA returned no object")
	}
	return output, nil
}

func hresultError(operation string, result uintptr) error {
	return fmt.Errorf("%s: HRESULT 0x%08x", operation, uint32(result))
}

type automationThread struct {
	automation *comObject
	actions    chan func()
	stop       chan struct{}
	done       chan struct{}
	startup    chan error
	closeOnce  sync.Once
}

func startAutomationThread() (*automationThread, error) {
	thread := &automationThread{
		actions: make(chan func()),
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
		startup: make(chan error, 1),
	}
	go thread.run()
	if err := <-thread.startup; err != nil {
		return nil, err
	}
	return thread, nil
}

func (thread *automationThread) run() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(thread.done)
	if err := windows.CoInitializeEx(0, windows.COINIT_APARTMENTTHREADED); err != nil {
		thread.startup <- err
		return
	}
	defer windows.CoUninitialize()
	result, _, _ := procOleInitialize.Call(0)
	if int32(result) < 0 {
		thread.startup <- hresultError("OleInitialize", result)
		return
	}
	defer procOleUninitialize.Call()
	var automation *comObject
	result, _, _ = procCoCreateInstance.Call(
		uintptr(unsafe.Pointer(&classAutomation)),
		0,
		windows.CLSCTX_INPROC_SERVER,
		uintptr(unsafe.Pointer(&interfaceAutomation)),
		uintptr(unsafe.Pointer(&automation)),
	)
	if int32(result) < 0 {
		thread.startup <- hresultError("CoCreateInstance(CUIAutomation)", result)
		return
	}
	thread.automation = automation
	defer automation.release()
	thread.startup <- nil
	for {
		select {
		case action := <-thread.actions:
			action()
		case <-thread.stop:
			return
		}
	}
}

func (thread *automationThread) call(action func(*comObject)) bool {
	executed := make(chan struct{})
	select {
	case <-thread.stop:
		return false
	case thread.actions <- func() {
		action(thread.automation)
		close(executed)
	}:
	}
	select {
	case <-executed:
		return true
	case <-thread.done:
		return false
	}
}

func (thread *automationThread) close() {
	thread.closeOnce.Do(func() { close(thread.stop) })
	<-thread.done
}

func focusedElement(automation *comObject) (*comObject, error) {
	return automation.output(automationFocusedElement)
}

func sameElement(automation *comObject, first, second *comObject) (bool, error) {
	var same int32
	result := automation.call(automationCompareElements,
		uintptr(unsafe.Pointer(first)),
		uintptr(unsafe.Pointer(second)),
		uintptr(unsafe.Pointer(&same)),
	)
	if int32(result) < 0 {
		return false, hresultError("CompareElements", result)
	}
	return same != 0, nil
}

func currentPattern(element *comObject, patternID int, interfaceID *windows.GUID) (*comObject, error) {
	var pattern *comObject
	result := element.call(elementCurrentPatternAs,
		uintptr(patternID),
		uintptr(unsafe.Pointer(interfaceID)),
		uintptr(unsafe.Pointer(&pattern)),
	)
	if int32(result) < 0 {
		return nil, hresultError("GetCurrentPatternAs", result)
	}
	if pattern == nil {
		return nil, fmt.Errorf("pattern %d is unavailable", patternID)
	}
	return pattern, nil
}

func readBSTR(address *uint16) string {
	if address == nil {
		return ""
	}
	defer procSysFreeString.Call(uintptr(unsafe.Pointer(address)))
	length, _, _ := procSysStringLen.Call(uintptr(unsafe.Pointer(address)))
	return windows.UTF16ToString(unsafe.Slice(address, int(length)))
}

func waitForClipboardMessage() {
	procMsgWaitForMessage.Call(0, 0, 10, queueSendMessage, 0)
	var message [8]uint64
	procPeekMessage.Call(uintptr(unsafe.Pointer(&message[0])), 0, 0, 0, peekSentMessage)
}
