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

func focusedWindowFrame() (WindowFrame, bool) {
	if loadAXWindow() != nil {
		return WindowFrame{}, false
	}
	front := frontmost()
	if front.PID <= 0 {
		return WindowFrame{}, false
	}
	application := axWindow.createApplication(int32(front.PID))
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
