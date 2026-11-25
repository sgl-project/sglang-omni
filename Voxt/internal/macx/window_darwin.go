//go:build darwin

package macx

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

var axWindow struct {
	once sync.Once
	err  error

	createApplication   func(int32) uintptr
	setMessagingTimeout func(uintptr, float32) int32
	copyAttributeValue  func(uintptr, uintptr, *uintptr) int32
	valueGetValue       func(uintptr, uint32, unsafe.Pointer) bool
	release             func(uintptr)

	focusedWindow uintptr
	mainWindow    uintptr
	position      uintptr
	size          uintptr
}

func axConstant(lib uintptr, name string) (uintptr, error) {
	p, err := purego.Dlsym(lib, name)
	if err != nil {
		return 0, fmt.Errorf("macx: %s: %w", name, err)
	}
	if p == 0 {
		return 0, fmt.Errorf("macx: %s is unavailable", name)
	}
	value := **(**uintptr)(unsafe.Pointer(&p))
	if value == 0 {
		return 0, fmt.Errorf("macx: %s is null", name)
	}
	return value, nil
}

func loadAXWindow() error {
	axWindow.once.Do(func() {
		const rtld = purego.RTLD_GLOBAL | purego.RTLD_NOW
		appServices, err := purego.Dlopen("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices", rtld)
		if err != nil {
			axWindow.err = err
			return
		}
		foundation, err := purego.Dlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation", rtld)
		if err != nil {
			axWindow.err = err
			return
		}
		purego.RegisterLibFunc(&axWindow.createApplication, appServices, "AXUIElementCreateApplication")
		purego.RegisterLibFunc(&axWindow.setMessagingTimeout, appServices, "AXUIElementSetMessagingTimeout")
		purego.RegisterLibFunc(&axWindow.copyAttributeValue, appServices, "AXUIElementCopyAttributeValue")
		purego.RegisterLibFunc(&axWindow.valueGetValue, appServices, "AXValueGetValue")
		purego.RegisterLibFunc(&axWindow.release, foundation, "CFRelease")
		for name, target := range map[string]*uintptr{
			"kAXFocusedWindowAttribute": &axWindow.focusedWindow,
			"kAXMainWindowAttribute":    &axWindow.mainWindow,
			"kAXPositionAttribute":      &axWindow.position,
			"kAXSizeAttribute":          &axWindow.size,
		} {
			value, err := axConstant(appServices, name)
			if err != nil {
				axWindow.err = err
				return
			}
			*target = value
		}
	})
	return axWindow.err
}

func axAttribute(element, name uintptr) (uintptr, bool) {
	var value uintptr
	if axWindow.copyAttributeValue(element, name, &value) != 0 || value == 0 {
		return 0, false
	}
	return value, true
}

func accessibilityFocusedWindowFrame(pid int) (WindowFrame, bool) {
	if loadAXWindow() != nil || pid <= 0 {
		return WindowFrame{}, false
	}
	application := axWindow.createApplication(int32(pid))
	if application == 0 {
		return WindowFrame{}, false
	}
	defer axWindow.release(application)
	axWindow.setMessagingTimeout(application, 0.3)

	window, ok := axAttribute(application, axWindow.focusedWindow)
	if !ok {
		window, ok = axAttribute(application, axWindow.mainWindow)
	}
	if !ok {
		return WindowFrame{}, false
	}
	defer axWindow.release(window)

	positionRef, ok := axAttribute(window, axWindow.position)
	if !ok {
		return WindowFrame{}, false
	}
	var position struct{ X, Y float64 }
	const axValueCGPoint = 1
	positionOK := axWindow.valueGetValue(positionRef, axValueCGPoint, unsafe.Pointer(&position))
	axWindow.release(positionRef)
	if !positionOK {
		return WindowFrame{}, false
	}

	sizeRef, ok := axAttribute(window, axWindow.size)
	if !ok {
		return WindowFrame{}, false
	}
	var size struct{ Width, Height float64 }
	const axValueCGSize = 2
	sizeOK := axWindow.valueGetValue(sizeRef, axValueCGSize, unsafe.Pointer(&size))
	axWindow.release(sizeRef)
	if !sizeOK || size.Width <= 0 || size.Height <= 0 {
		return WindowFrame{}, false
	}
	return WindowFrame{X: int(position.X), Y: int(position.Y), Width: int(size.Width), Height: int(size.Height)}, true
}

var cgWindow struct {
	once sync.Once
	err  error

	copyList           func(uint32, uint32) uintptr
	arrayCount         func(uintptr) int
	arrayValueAt       func(uintptr, int) uintptr
	dictionaryGet      func(uintptr, uintptr) uintptr
	numberGetValue     func(uintptr, int, *int32) uint8
	rectWithDictionary func(uintptr, unsafe.Pointer) bool
	release            func(uintptr)

	ownerPID uintptr
	layer    uintptr
	bounds   uintptr
}

func loadCGWindow() error {
	cgWindow.once.Do(func() {
		const rtld = purego.RTLD_GLOBAL | purego.RTLD_NOW
		coreGraphics, err := purego.Dlopen("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics", rtld)
		if err != nil {
			cgWindow.err = err
			return
		}
		foundation, err := purego.Dlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation", rtld)
		if err != nil {
			cgWindow.err = err
			return
		}
		purego.RegisterLibFunc(&cgWindow.copyList, coreGraphics, "CGWindowListCopyWindowInfo")
		purego.RegisterLibFunc(&cgWindow.rectWithDictionary, coreGraphics, "CGRectMakeWithDictionaryRepresentation")
		purego.RegisterLibFunc(&cgWindow.arrayCount, foundation, "CFArrayGetCount")
		purego.RegisterLibFunc(&cgWindow.arrayValueAt, foundation, "CFArrayGetValueAtIndex")
		purego.RegisterLibFunc(&cgWindow.dictionaryGet, foundation, "CFDictionaryGetValue")
		purego.RegisterLibFunc(&cgWindow.numberGetValue, foundation, "CFNumberGetValue")
		purego.RegisterLibFunc(&cgWindow.release, foundation, "CFRelease")
		for name, target := range map[string]*uintptr{
			"kCGWindowOwnerPID": &cgWindow.ownerPID,
			"kCGWindowLayer":    &cgWindow.layer,
			"kCGWindowBounds":   &cgWindow.bounds,
		} {
			value, err := axConstant(coreGraphics, name)
			if err != nil {
				cgWindow.err = err
				return
			}
			*target = value
		}
	})
	return cgWindow.err
}

// The window server needs no Accessibility permission and lists windows front to back.
func windowServerFocusedWindowFrame(pid int) (WindowFrame, bool) {
	if loadCGWindow() != nil || pid <= 0 {
		return WindowFrame{}, false
	}
	const onScreenExcludeDesktop = 0x11
	list := cgWindow.copyList(onScreenExcludeDesktop, 0)
	if list == 0 {
		return WindowFrame{}, false
	}
	defer cgWindow.release(list)
	for i := 0; i < cgWindow.arrayCount(list); i++ {
		window := cgWindow.arrayValueAt(list, i)
		var ownerPID, layer int32
		if cgWindow.numberGetValue(cgWindow.dictionaryGet(window, cgWindow.ownerPID), cfNumberIntType, &ownerPID) == 0 || int(ownerPID) != pid {
			continue
		}
		if cgWindow.numberGetValue(cgWindow.dictionaryGet(window, cgWindow.layer), cfNumberIntType, &layer) == 0 || layer != 0 {
			continue
		}
		bounds := cgWindow.dictionaryGet(window, cgWindow.bounds)
		if bounds == 0 {
			continue
		}
		var frame struct{ X, Y, Width, Height float64 }
		if !cgWindow.rectWithDictionary(bounds, unsafe.Pointer(&frame)) || frame.Width <= 0 || frame.Height <= 0 {
			continue
		}
		return WindowFrame{X: int(frame.X), Y: int(frame.Y), Width: int(frame.Width), Height: int(frame.Height)}, true
	}
	return WindowFrame{}, false
}
