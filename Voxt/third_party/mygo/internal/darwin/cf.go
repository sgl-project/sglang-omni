//go:build darwin

package darwin

import (
	"unsafe"

	"github.com/ebitengine/purego"
)

var (
	cfRunLoopGetMain      func() uintptr
	cfRunLoopSourceCreate func(alloc uintptr, order int, ctx unsafe.Pointer) uintptr
	cfRunLoopAddSource    func(rl, src, mode uintptr)
	cfRelease             func(obj uintptr)
	kCFRunLoopCommonModes uintptr

	// LaunchServices.
	lsSetDefaultHandlerForURLScheme  func(scheme, bundleID id) int32
	lsCopyDefaultHandlerForURLScheme func(scheme id) id

	// Power.
	iopsCopyPowerSourcesInfo          func() uintptr
	iopsGetProvidingPowerSourceType   func(snapshot uintptr) id
	cgEventSourceSecondsSinceLastType func(state int32, eventType uint32) float64

	cgDisplayIsBuiltin func(display uint32) bool
	cgDisplayRotation  func(display uint32) float64
	nsBeep             func()

	// Carbon hot keys.
	getApplicationEventTarget func() uintptr
	registerEventHotKey       func(keyCode, modifiers uint32, hotKeyID eventHotKeyID, target uintptr, options uint32, outRef *uintptr) int32
	unregisterEventHotKey     func(ref uintptr) int32
	installEventHandler       func(target, handler uintptr, numTypes uint32, list *eventTypeSpec, userData uintptr, outRef *uintptr) int32
	getEventParameter         func(event uintptr, name, typ uint32, outType uintptr, size uintptr, outSize uintptr, data unsafe.Pointer) int32
)

// Called by address on every Signal, like the hot functions of objc.go.
var cfRunLoopSourceSignalFn, cfRunLoopWakeUpFn uintptr

type eventHotKeyID struct {
	Signature uint32
	ID        uint32
}

type eventTypeSpec struct {
	Class uint32
	Kind  uint32
}

func loadCF() {
	purego.RegisterLibFunc(&cfRunLoopGetMain, libCF, "CFRunLoopGetMain")
	purego.RegisterLibFunc(&cfRunLoopSourceCreate, libCF, "CFRunLoopSourceCreate")
	purego.RegisterLibFunc(&cfRunLoopAddSource, libCF, "CFRunLoopAddSource")
	cfRunLoopSourceSignalFn = mustDlsym(libCF, "CFRunLoopSourceSignal")
	cfRunLoopWakeUpFn = mustDlsym(libCF, "CFRunLoopWakeUp")
	purego.RegisterLibFunc(&cfRelease, libCF, "CFRelease")
	p, err := purego.Dlsym(libCF, "kCFRunLoopCommonModes")
	if err != nil {
		panic(err)
	}
	kCFRunLoopCommonModes = **(**uintptr)(unsafe.Pointer(&p))

	purego.RegisterLibFunc(&lsSetDefaultHandlerForURLScheme, libCoreServices, "LSSetDefaultHandlerForURLScheme")
	purego.RegisterLibFunc(&lsCopyDefaultHandlerForURLScheme, libCoreServices, "LSCopyDefaultHandlerForURLScheme")

	if iokit, err := purego.Dlopen("/System/Library/Frameworks/IOKit.framework/IOKit", purego.RTLD_GLOBAL|purego.RTLD_NOW); err == nil {
		purego.RegisterLibFunc(&iopsCopyPowerSourcesInfo, iokit, "IOPSCopyPowerSourcesInfo")
		purego.RegisterLibFunc(&iopsGetProvidingPowerSourceType, iokit, "IOPSGetProvidingPowerSourceType")
	}
	purego.RegisterLibFunc(&cgEventSourceSecondsSinceLastType, libCG, "CGEventSourceSecondsSinceLastEventType")

	purego.RegisterLibFunc(&cgDisplayIsBuiltin, libCG, "CGDisplayIsBuiltin")
	purego.RegisterLibFunc(&cgDisplayRotation, libCG, "CGDisplayRotation")
	purego.RegisterLibFunc(&nsBeep, libAppKit, "NSBeep")

	purego.RegisterLibFunc(&getApplicationEventTarget, libCarbon, "GetApplicationEventTarget")
	purego.RegisterLibFunc(&registerEventHotKey, libCarbon, "RegisterEventHotKey")
	purego.RegisterLibFunc(&unregisterEventHotKey, libCarbon, "UnregisterEventHotKey")
	purego.RegisterLibFunc(&installEventHandler, libCarbon, "InstallEventHandler")
	purego.RegisterLibFunc(&getEventParameter, libCarbon, "GetEventParameter")
}

// constString reads an exported NSString constant such as
// NSPasteboardTypeString.
func constString(lib uintptr, name string) id {
	p, err := purego.Dlsym(lib, name)
	if err != nil || p == 0 {
		return 0
	}
	return **(**id)(unsafe.Pointer(&p))
}
