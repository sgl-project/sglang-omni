//go:build darwin

package darwin

import (
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

var (
	frameworksOnce sync.Once
	frameworksErr  error

	poolPush func() uintptr
	poolPop  func(uintptr)

	secCodeCopySelf                func(flags uint32, code *uintptr) int32
	secCodeCopySigningInformation  func(code uintptr, flags uint32, info *uintptr) int32
	cfDictionaryGetValue           func(dict, key uintptr) uintptr
	cfRelease                      func(uintptr)
	kSecCodeInfoTeamIdentifierAddr uintptr

	selCache sync.Map
)

func loadFrameworks() error {
	frameworksOnce.Do(func() {
		libobjc, err := purego.Dlopen("/usr/lib/libobjc.A.dylib", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			frameworksErr = err
			return
		}
		purego.RegisterLibFunc(&poolPush, libobjc, "objc_autoreleasePoolPush")
		purego.RegisterLibFunc(&poolPop, libobjc, "objc_autoreleasePoolPop")
		// Foundation and AppKit register NSWorkspace, NSRunningApplication.
		for _, fw := range []string{
			"/System/Library/Frameworks/Foundation.framework/Foundation",
			"/System/Library/Frameworks/AppKit.framework/AppKit",
		} {
			if _, err := purego.Dlopen(fw, purego.RTLD_GLOBAL|purego.RTLD_NOW); err != nil {
				frameworksErr = err
				return
			}
		}
		cf, err := purego.Dlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			frameworksErr = err
			return
		}
		purego.RegisterLibFunc(&cfDictionaryGetValue, cf, "CFDictionaryGetValue")
		purego.RegisterLibFunc(&cfRelease, cf, "CFRelease")
		sec, err := purego.Dlopen("/System/Library/Frameworks/Security.framework/Security", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			frameworksErr = err
			return
		}
		purego.RegisterLibFunc(&secCodeCopySelf, sec, "SecCodeCopySelf")
		purego.RegisterLibFunc(&secCodeCopySigningInformation, sec, "SecCodeCopySigningInformation")
		kSecCodeInfoTeamIdentifierAddr, _ = purego.Dlsym(sec, "kSecCodeInfoTeamIdentifier")
	})
	return frameworksErr
}

func sel(name string) objc.SEL {
	if s, ok := selCache.Load(name); ok {
		return s.(objc.SEL)
	}
	s := objc.RegisterName(name)
	selCache.Store(name, s)
	return s
}

func class(name string) objc.ID { return objc.ID(objc.GetClass(name)) }

const nsUTF8StringEncoding = 4

// nsString returns an autoreleased NSString of s.
func nsString(s string) objc.ID {
	b := []byte(s)
	var p unsafe.Pointer
	if len(b) > 0 {
		p = unsafe.Pointer(&b[0])
	}
	str := class("NSString").Send(sel("alloc")).Send(sel("initWithBytes:length:encoding:"), p, uintptr(len(b)), uintptr(nsUTF8StringEncoding))
	runtime.KeepAlive(b)
	return str.Send(sel("autorelease"))
}

// goString copies an NSString (or a toll-free bridged CFString).
func goString(str objc.ID) string {
	if str == 0 {
		return ""
	}
	p := objc.Send[uintptr](str, sel("UTF8String"))
	if p == 0 {
		return ""
	}
	n := objc.Send[uintptr](str, sel("lengthOfBytesUsingEncoding:"), uintptr(nsUTF8StringEncoding))
	return string(unsafe.Slice(*(**byte)(unsafe.Pointer(&p)), n))
}

// withPool runs fn inside an autorelease pool. An autorelease pool belongs
// to the thread that pushed it, so the goroutine stays on one OS thread from
// the push to the pop: otherwise the Go scheduler may move it in between,
// and the pop (or an autorelease) lands on another thread's pool stack, which
// crashes. Callers may be any goroutine.
func withPool(fn func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := poolPush()
	defer poolPop(pool)
	fn()
}

// RunningApp is a running process of an app.
type RunningApp struct {
	PID  int
	Path string // the .app bundle
}

// AppURLs asks LaunchServices where apps with this bundle id are installed
// (NSWorkspace URLsForApplicationsWithBundleIdentifier:, macOS 12+).
func AppURLs(bundleID string) ([]string, error) {
	if err := loadFrameworks(); err != nil {
		return nil, err
	}
	var out []string
	withPool(func() {
		ws := class("NSWorkspace").Send(sel("sharedWorkspace"))
		arr := ws.Send(sel("URLsForApplicationsWithBundleIdentifier:"), nsString(bundleID))
		n := objc.Send[uintptr](arr, sel("count"))
		for i := range n {
			u := arr.Send(sel("objectAtIndex:"), i)
			if p := goString(u.Send(sel("path"))); p != "" {
				out = append(out, p)
			}
		}
	})
	return out, nil
}

// Running lists the running processes of a bundle id
// (NSRunningApplication runningApplicationsWithBundleIdentifier:).
func Running(bundleID string) ([]RunningApp, error) {
	if err := loadFrameworks(); err != nil {
		return nil, err
	}
	var out []RunningApp
	withPool(func() {
		arr := class("NSRunningApplication").Send(sel("runningApplicationsWithBundleIdentifier:"), nsString(bundleID))
		n := objc.Send[uintptr](arr, sel("count"))
		for i := range n {
			app := arr.Send(sel("objectAtIndex:"), i)
			if objc.Send[bool](app, sel("isTerminated")) {
				continue
			}
			pid := objc.Send[int32](app, sel("processIdentifier"))
			path := goString(app.Send(sel("bundleURL")).Send(sel("path")))
			out = append(out, RunningApp{PID: int(pid), Path: path})
		}
	})
	return out, nil
}

// TeamID is the Team ID of the running process's code signature; "" when
// unsigned or ad hoc.
func TeamID() string {
	if loadFrameworks() != nil || kSecCodeInfoTeamIdentifierAddr == 0 {
		return ""
	}
	var code uintptr
	if secCodeCopySelf(0, &code) != 0 || code == 0 {
		return ""
	}
	defer cfRelease(code)
	var info uintptr
	const kSecCSSigningInformation = 1 << 1
	if secCodeCopySigningInformation(code, kSecCSSigningInformation, &info) != 0 || info == 0 {
		return ""
	}
	defer cfRelease(info)
	addr := kSecCodeInfoTeamIdentifierAddr
	key := **(**uintptr)(unsafe.Pointer(&addr)) // the CFStringRef the symbol holds
	v := cfDictionaryGetValue(info, key)
	if v == 0 {
		return ""
	}
	var s string
	withPool(func() { s = goString(objc.ID(v)) })
	return s
}
