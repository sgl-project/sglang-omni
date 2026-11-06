//go:build darwin

// Package darwin implements the macOS backend (AppKit + WKWebView) in pure
// Go: Objective-C is driven through the runtime with purego, no cgo.
package darwin

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

type id = objc.ID

// NSPoint, NSSize and NSRect mirror the CoreGraphics structs (64-bit).
type NSPoint struct{ X, Y float64 }
type NSSize struct{ Width, Height float64 }
type NSRect struct {
	Origin NSPoint
	Size   NSSize
}

var (
	msgSendAddr      uintptr
	msgSendSuperAddr uintptr

	// Typed variants of objc_msgSend for float and struct arguments.
	msgRect          func(obj id, sel objc.SEL) NSRect
	msgSetRect       func(obj id, sel objc.SEL, r NSRect)
	msgSetRectBool   func(obj id, sel objc.SEL, r NSRect, b bool)
	msgRectForRect   func(obj id, sel objc.SEL, r NSRect, style uint) NSRect
	msgRectToRect    func(obj id, sel objc.SEL, r NSRect) NSRect
	msgFloat         func(obj id, sel objc.SEL) float64
	msgSetFloat      func(obj id, sel objc.SEL, v float64)
	msgSize          func(obj id, sel objc.SEL) NSSize
	msgSetSize       func(obj id, sel objc.SEL, s NSSize)
	msgSuperSetSize  func(sup uintptr, sel objc.SEL, s NSSize)
	msgSuperPoint    func(sup uintptr, sel objc.SEL, p NSPoint) id
	msgPoint         func(obj id, sel objc.SEL) NSPoint
	msgSetPoint      func(obj id, sel objc.SEL, p NSPoint)
	msgInitRect      func(obj id, sel objc.SEL, r NSRect) id
	msgInitRectID    func(obj id, sel objc.SEL, r NSRect, a id) id
	msgInitSize      func(obj id, sel objc.SEL, s NSSize) id
	msgRectRadii     func(cls id, sel objc.SEL, r NSRect, rx, ry float64) id
	msgRectIDUint    func(obj id, sel objc.SEL, r NSRect, a id, n uint)
	msgInitIDPoint   func(obj id, sel objc.SEL, a id, p NSPoint) id
	msgInitWindow    func(obj id, sel objc.SEL, r NSRect, style uint, backing uint, deferFlag bool) id
	msgColor         func(cls id, sel objc.SEL, r, g, b, a float64) id
	msgOtherEvent    func(cls id, sel objc.SEL, typ uint, loc NSPoint, flags uint, ts float64, wn int, ctx id, subtype int16, d1, d2 int) id
	msgPopUpMenu     func(menu id, sel objc.SEL, item id, loc NSPoint, view id) bool
	msgNextEvent     func(app id, sel objc.SEL, mask uint64, until id, mode id, dequeue bool) id
	msgPointFromView func(obj id, sel objc.SEL, p NSPoint, view id) NSPoint
	msgDateSince     func(cls id, sel objc.SEL, seconds float64) id
	msgFloatID       func(obj id, sel objc.SEL, v float64) id
	msgFloat2ID      func(obj id, sel objc.SEL, a, b float64) id
	msgFontFamily    func(obj id, sel objc.SEL, family id, traits uint, weight int, size float64) id
	msgMouseEvent    func(cls id, sel objc.SEL, typ uint, loc NSPoint, flags uint, ts float64, wn int, ctx id, eventNumber int, clickCount int, pressure float32) id
	msgKeyEvent      func(cls id, sel objc.SEL, typ uint, loc NSPoint, flags uint, ts float64, wn int, ctx id, chars, charsIgnoring id, repeat bool, keyCode uint16) id
	msgDrawImage     func(obj id, sel objc.SEL, dst, src NSRect, op uint, fraction float64)
	msgInitBitmap    func(obj id, sel objc.SEL, planes uintptr, width, height, bitsPerSample, samplesPerPixel int, alpha, planar bool, colorSpace id, bytesPerRow, bitsPerPixel int) id

	blockCopy    func(block uintptr) uintptr
	blockRelease func(block uintptr)

	// Functions on hot paths, called by address: RegisterFunc's wrappers
	// use reflection and allocate on every call.
	poolPushFn, poolPopFn, mainNPFn uintptr
)

func mustDlsym(lib uintptr, name string) uintptr {
	p, err := purego.Dlsym(lib, name)
	if err != nil {
		panic(fmt.Sprintf("mygo: %v", err))
	}
	return p
}

func mustDlopen(path string) uintptr {
	h, err := purego.Dlopen(path, purego.RTLD_GLOBAL|purego.RTLD_NOW)
	if err != nil {
		panic(fmt.Sprintf("mygo: cannot load %s: %v", path, err))
	}
	return h
}

var (
	libObjC, libCF, libCG, libAppKit, libWebKit, libFoundation, libCarbon uintptr
	libCoreServices                                                       uintptr
	loadOnce                                                              sync.Once
)

// load opens the frameworks and prepares the typed message functions.
func load() {
	loadOnce.Do(func() {
		libObjC = mustDlopen("/usr/lib/libobjc.A.dylib")
		libFoundation = mustDlopen("/System/Library/Frameworks/Foundation.framework/Foundation")
		libCF = mustDlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
		libCG = mustDlopen("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
		libAppKit = mustDlopen("/System/Library/Frameworks/AppKit.framework/AppKit")
		libWebKit = mustDlopen("/System/Library/Frameworks/WebKit.framework/WebKit")
		libCarbon = mustDlopen("/System/Library/Frameworks/Carbon.framework/Carbon")
		libCoreServices = mustDlopen("/System/Library/Frameworks/CoreServices.framework/CoreServices")
		// Optional frameworks, loaded so their classes are registered.
		_, _ = purego.Dlopen("/System/Library/Frameworks/UniformTypeIdentifiers.framework/UniformTypeIdentifiers", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		_, _ = purego.Dlopen("/System/Library/Frameworks/UserNotifications.framework/UserNotifications", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		_, _ = purego.Dlopen("/System/Library/Frameworks/ServiceManagement.framework/ServiceManagement", purego.RTLD_GLOBAL|purego.RTLD_NOW)

		var err error
		msgSendAddr, err = purego.Dlsym(libObjC, "objc_msgSend")
		if err != nil {
			panic(err)
		}
		// Methods returning structs larger than 16 bytes use a different
		// entry point on amd64.
		stret := msgSendAddr
		if runtime.GOARCH == "amd64" {
			stret, err = purego.Dlsym(libObjC, "objc_msgSend_stret")
			if err != nil {
				panic(err)
			}
		}
		msgSendSuperAddr, err = purego.Dlsym(libObjC, "objc_msgSendSuper")
		if err != nil {
			panic(err)
		}
		purego.RegisterFunc(&msgRect, stret)
		purego.RegisterFunc(&msgRectForRect, stret)
		purego.RegisterFunc(&msgRectToRect, stret)
		purego.RegisterFunc(&msgSetRect, msgSendAddr)
		purego.RegisterFunc(&msgDrawImage, msgSendAddr)
		purego.RegisterFunc(&msgInitBitmap, msgSendAddr)
		purego.RegisterFunc(&msgSetRectBool, msgSendAddr)
		purego.RegisterFunc(&msgFloat, msgSendAddr)
		purego.RegisterFunc(&msgSetFloat, msgSendAddr)
		purego.RegisterFunc(&msgSize, msgSendAddr)
		purego.RegisterFunc(&msgSetSize, msgSendAddr)
		purego.RegisterFunc(&msgSuperSetSize, msgSendSuperAddr)
		purego.RegisterFunc(&msgSuperPoint, msgSendSuperAddr)
		purego.RegisterFunc(&msgPoint, msgSendAddr)
		purego.RegisterFunc(&msgSetPoint, msgSendAddr)
		purego.RegisterFunc(&msgInitRect, msgSendAddr)
		purego.RegisterFunc(&msgInitRectID, msgSendAddr)
		purego.RegisterFunc(&msgInitSize, msgSendAddr)
		purego.RegisterFunc(&msgRectRadii, msgSendAddr)
		purego.RegisterFunc(&msgRectIDUint, msgSendAddr)
		purego.RegisterFunc(&msgInitIDPoint, msgSendAddr)
		purego.RegisterFunc(&msgInitWindow, msgSendAddr)
		purego.RegisterFunc(&msgColor, msgSendAddr)
		purego.RegisterFunc(&msgOtherEvent, msgSendAddr)
		purego.RegisterFunc(&msgPopUpMenu, msgSendAddr)
		purego.RegisterFunc(&msgNextEvent, msgSendAddr)
		purego.RegisterFunc(&msgPointFromView, msgSendAddr)
		purego.RegisterFunc(&msgDateSince, msgSendAddr)
		purego.RegisterFunc(&msgFloatID, msgSendAddr)
		purego.RegisterFunc(&msgFloat2ID, msgSendAddr)
		purego.RegisterFunc(&msgFontFamily, msgSendAddr)
		purego.RegisterFunc(&msgKeyEvent, msgSendAddr)
		purego.RegisterFunc(&msgMouseEvent, msgSendAddr)

		poolPushFn = mustDlsym(libObjC, "objc_autoreleasePoolPush")
		poolPopFn = mustDlsym(libObjC, "objc_autoreleasePoolPop")
		purego.RegisterLibFunc(&blockCopy, libObjC, "_Block_copy")
		purego.RegisterLibFunc(&blockRelease, libObjC, "_Block_release")
		mainNPFn = mustDlsym(purego.RTLD_DEFAULT, "pthread_main_np")
		loadCF()
	})
}

var (
	selMu      sync.RWMutex
	selCache   = map[string]objc.SEL{}
	clsCache   = map[string]id{}
	superCache = map[string]id{}
)

// sel returns the selector for name, cached.
func sel(name string) objc.SEL {
	selMu.RLock()
	s, ok := selCache[name]
	selMu.RUnlock()
	if ok {
		return s
	}
	s = objc.RegisterName(name)
	selMu.Lock()
	selCache[name] = s
	selMu.Unlock()
	return s
}

// class returns the class object for name, cached. It panics for classes
// that do not exist since that is a programming error.
func class(name string) id {
	selMu.RLock()
	c, ok := clsCache[name]
	selMu.RUnlock()
	if ok {
		return c
	}
	c = id(objc.GetClass(name))
	if c == 0 {
		panic("mygo: Objective-C class " + name + " not found")
	}
	selMu.Lock()
	clsCache[name] = c
	selMu.Unlock()
	return c
}

// hasClass reports whether a class exists (for APIs of newer macOS versions).
func hasClass(name string) bool { return objc.GetClass(name) != 0 }

// send sends a message whose arguments are integers or pointers. Go
// pointers converted in the call, as in uintptr(unsafe.Pointer(&x)), are
// moved to the heap and kept alive until it returns (go:uintptrescapes):
// otherwise stack growth could move x while Objective-C uses it.
//
//go:uintptrescapes
func send(obj id, selector string, args ...uintptr) id {
	return sendSel(obj, sel(selector), args...)
}

//go:uintptrescapes
func sendSel(obj id, s objc.SEL, args ...uintptr) id {
	var a [12]uintptr
	a[0], a[1] = uintptr(obj), uintptr(s)
	n := copy(a[2:], args)
	r, _, _ := purego.SyscallN(msgSendAddr, a[:n+2]...)
	return id(r)
}

// sendBool sends a message returning BOOL.
//
//go:uintptrescapes
func sendBool(obj id, selector string, args ...uintptr) bool {
	return byte(send(obj, selector, args...)) != 0
}

// sendInt sends a message returning NSInteger.
//
//go:uintptrescapes
func sendInt(obj id, selector string, args ...uintptr) int {
	return int(send(obj, selector, args...))
}

// objcSuper mirrors struct objc_super.
type objcSuper struct {
	receiver   id
	superClass id
}

// superOf returns the struct objc_super that calls the superclass of
// className with self, pinned until pin is unpinned: passed as an integer,
// it must stay off the stack, which may move. The superclass is resolved
// from className rather than from the object's class, which the runtime may
// replace with a generated subclass (key-value observing does): resolving
// "super" from it would call the override again and recurse forever.
func superOf(self id, className string, pin *runtime.Pinner) uintptr {
	sup := &objcSuper{receiver: self, superClass: superclass(className)}
	pin.Pin(sup)
	return uintptr(unsafe.Pointer(sup))
}

// superclass returns the superclass of a class, cached: overrides call it
// as often as AppKit calls them, as setFrameSize: while a window resizes.
func superclass(name string) id {
	selMu.RLock()
	c, ok := superCache[name]
	selMu.RUnlock()
	if ok {
		return c
	}
	c = id(objc.Class(class(name)).SuperClass())
	selMu.Lock()
	superCache[name] = c
	selMu.Unlock()
	return c
}

// sendSuper calls the superclass implementation of a method overridden in
// className (see superOf).
//
//go:uintptrescapes
func sendSuper(self id, className string, s objc.SEL, args ...uintptr) id {
	var pin runtime.Pinner
	defer pin.Unpin()
	var a [10]uintptr
	a[0], a[1] = superOf(self, className, &pin), uintptr(s)
	n := copy(a[2:], args)
	r, _, _ := purego.SyscallN(msgSendSuperAddr, a[:n+2]...)
	return id(r)
}

// sendSuperSize is sendSuper for a method taking an NSSize, which travels in
// floating-point registers and so cannot go through the integer arguments.
func sendSuperSize(self id, className string, s objc.SEL, size NSSize) {
	var pin runtime.Pinner
	defer pin.Unpin()
	msgSuperSetSize(superOf(self, className, &pin), s, size)
}

// sendSuperPoint is sendSuper for a method taking an NSPoint and
// returning an object, as hitTest:.
func sendSuperPoint(self id, className string, s objc.SEL, p NSPoint) id {
	var pin runtime.Pinner
	defer pin.Unpin()
	return msgSuperPoint(superOf(self, className, &pin), s, p)
}

func respondsTo(obj id, selector string) bool {
	return obj != 0 && sendBool(obj, "respondsToSelector:", uintptr(sel(selector)))
}

func boolArg(b bool) uintptr {
	if b {
		return 1
	}
	return 0
}

func retain(obj id) id {
	if obj == 0 {
		return 0
	}
	return send(obj, "retain")
}

func release(obj id) {
	if obj != 0 {
		send(obj, "release")
	}
}

func autorelease(obj id) id {
	if obj == 0 {
		return 0
	}
	return send(obj, "autorelease")
}

// appKitString returns an NSString constant that AppKit exports, such as
// NSPrintJobSavingURL, whose value may differ from its name.
func appKitString(name string) id {
	p, err := purego.Dlsym(libAppKit, name)
	if err != nil {
		return nsString(name)
	}
	return **(**id)(unsafe.Pointer(&p))
}

// alloc creates an owned (+1) instance of a class with init.
func alloc(className string) id {
	return send(send(class(className), "alloc"), "init")
}

// withPool runs fn inside an autorelease pool. A pool belongs to the thread
// that pushed it, and popping it on another one crashes, so the goroutine
// stays on its thread meanwhile: off the main thread, where public methods
// such as App.Name read the bundle, it would move between threads.
func withPool(fn func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool, _, _ := purego.SyscallN(poolPushFn)
	defer purego.SyscallN(poolPopFn, pool)
	fn()
}

const nsUTF8StringEncoding = 4

// nsString returns an autoreleased NSString.
func nsString(s string) id {
	b := unsafe.StringData(s)
	str := send(send(class("NSString"), "alloc"), "initWithBytes:length:encoding:",
		uintptr(unsafe.Pointer(b)), uintptr(len(s)), nsUTF8StringEncoding)
	runtime.KeepAlive(s)
	return autorelease(str)
}

// goString copies an NSString into a Go string.
func goString(str id) string {
	if str == 0 {
		return ""
	}
	p := send(str, "UTF8String")
	if p == 0 {
		return ""
	}
	n := sendInt(str, "lengthOfBytesUsingEncoding:", nsUTF8StringEncoding)
	return string(unsafe.Slice(*(**byte)(unsafe.Pointer(&p)), n))
}

// nsData returns an autoreleased NSData with a copy of b.
func nsData(b []byte) id {
	var p unsafe.Pointer
	if len(b) > 0 {
		p = unsafe.Pointer(&b[0])
	}
	d := send(class("NSData"), "dataWithBytes:length:", uintptr(p), uintptr(len(b)))
	runtime.KeepAlive(b)
	return d
}

// goBytes copies the contents of an NSData.
func goBytes(data id) []byte {
	if data == 0 {
		return nil
	}
	n := sendInt(data, "length")
	if n == 0 {
		return []byte{}
	}
	p := send(data, "bytes")
	return append([]byte(nil), unsafe.Slice(*(**byte)(unsafe.Pointer(&p)), n)...)
}

func nsURL(s string) id { return send(class("NSURL"), "URLWithString:", uintptr(nsString(s))) }

// filePathURLBytes resolves a file reference URL (file:///.file/id=…,
// which Finder puts on the pasteboard when it drags or copies files) to a
// file URL of its path. Other URLs come back unchanged.
func filePathURLBytes(b []byte) []byte {
	s := strings.TrimRight(string(b), "\x00")
	if !strings.HasPrefix(s, "file:///.file/") {
		return b
	}
	u := nsURL(s)
	if u == 0 {
		return b
	}
	if p := send(u, "filePathURL"); p != 0 {
		if r := goString(send(p, "absoluteString")); r != "" {
			return []byte(r)
		}
	}
	return b
}

func fileURL(path string) id {
	return send(class("NSURL"), "fileURLWithPath:", uintptr(nsString(path)))
}

// nsArray returns an autoreleased NSArray of objs.
func nsArray(objs ...id) id {
	arr := send(class("NSMutableArray"), "arrayWithCapacity:", uintptr(len(objs)))
	for _, o := range objs {
		send(arr, "addObject:", uintptr(o))
	}
	return arr
}

func arrayItems(arr id) []id {
	if arr == 0 {
		return nil
	}
	n := sendInt(arr, "count")
	out := make([]id, n)
	for i := range n {
		out[i] = send(arr, "objectAtIndex:", uintptr(i))
	}
	return out
}

func nsNumberInt(v int) id { return send(class("NSNumber"), "numberWithLongLong:", uintptr(v)) }

func nsBool(v bool) id { return send(class("NSNumber"), "numberWithBool:", boolArg(v)) }

// callBlock invokes an Objective-C block received from the system, such as
// a WebKit completion handler.
func callBlock(block uintptr, args ...uintptr) {
	lit := *(**[3]uintptr)(unsafe.Pointer(&block))
	var a [8]uintptr
	a[0] = block
	n := copy(a[1:], args)
	purego.SyscallN(lit[2], a[:n+1]...)
}

// newBlock creates a block backed by a Go function. The function must take
// objc.Block as its first parameter.
func newBlock(fn any) objc.Block { return objc.NewBlock(fn) }

// nsError describes an NSError, preferring the JavaScript exception message
// WebKit attaches.
func nsError(err id) error {
	if err == 0 {
		return nil
	}
	info := send(err, "userInfo")
	if msg := send(info, "objectForKey:", uintptr(nsString("WKJavaScriptExceptionMessage"))); msg != 0 {
		return fmt.Errorf("%s", goString(msg))
	}
	return fmt.Errorf("%s", goString(send(err, "localizedDescription")))
}

// classDef builds an Objective-C class once.
func classDef(name, super string, protocols []string, methods []objc.MethodDef) id {
	var ps []*objc.Protocol
	for _, p := range protocols {
		if proto := objc.GetProtocol(p); proto != nil {
			ps = append(ps, proto)
		}
	}
	c, err := objc.RegisterClass(name, objc.GetClass(super), ps, nil, methods)
	if err != nil {
		panic(fmt.Sprintf("mygo: cannot register class %s: %v", name, err))
	}
	selMu.Lock()
	clsCache[name] = id(c)
	selMu.Unlock()
	return id(c)
}

func method(selector string, fn any) objc.MethodDef {
	return objc.MethodDef{Cmd: sel(selector), Fn: fn}
}
