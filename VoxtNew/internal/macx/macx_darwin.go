//go:build darwin

package macx

import (
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

var mac struct {
	once     sync.Once
	err      error
	poolPush func() uintptr
	poolPop  func(uintptr)
}

func load() error {
	mac.once.Do(func() {
		const rtld = purego.RTLD_GLOBAL | purego.RTLD_NOW
		objcLib, err := purego.Dlopen("/usr/lib/libobjc.A.dylib", rtld)
		if err != nil {
			mac.err = err
			return
		}
		if _, err := purego.Dlopen("/System/Library/Frameworks/AppKit.framework/AppKit", rtld); err != nil {
			mac.err = err
			return
		}
		purego.RegisterLibFunc(&mac.poolPush, objcLib, "objc_autoreleasePoolPush")
		purego.RegisterLibFunc(&mac.poolPop, objcLib, "objc_autoreleasePoolPop")
	})
	return mac.err
}

// pooled runs fn inside an autorelease pool, on one OS thread: the pool
// is the thread's, and a goroutine may move between threads (Frontmost
// from a goroutine popped another thread's pool and crashed, found by the
// overlay check).
func pooled(fn func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	p := mac.poolPush()
	defer mac.poolPop(p)
	fn()
}

var (
	selShared       = objc.RegisterName("sharedWorkspace")
	selFrontmost    = objc.RegisterName("frontmostApplication")
	selPID          = objc.RegisterName("processIdentifier")
	selBundleID     = objc.RegisterName("bundleIdentifier")
	selName         = objc.RegisterName("localizedName")
	selUTF8         = objc.RegisterName("UTF8String")
	selRunningApp   = objc.RegisterName("runningApplicationWithProcessIdentifier:")
	selSharedApp    = objc.RegisterName("sharedApplication")
	selIsActive     = objc.RegisterName("isActive")
	selResponds     = objc.RegisterName("respondsToSelector:")
	selYield        = objc.RegisterName("yieldActivationToApplication:")
	selActivateWith = objc.RegisterName("activateWithOptions:")
)

func goString(s objc.ID) string {
	if s == 0 {
		return ""
	}
	p := objc.Send[*byte](s, selUTF8)
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}

func class(name string) objc.ID { return objc.ID(objc.GetClass(name)) }

func frontmost() App {
	if load() != nil {
		return App{}
	}
	var a App
	pooled(func() {
		app := class("NSWorkspace").Send(selShared).Send(selFrontmost)
		if app == 0 {
			return
		}
		a.PID = int(objc.Send[int32](app, selPID))
		a.BundleID = goString(app.Send(selBundleID))
		a.Name = goString(app.Send(selName))
	})
	return a
}

func activate(pid int, wait time.Duration, runOnMain func(func())) (bool, time.Duration) {
	start := time.Now()
	if load() != nil || pid <= 0 {
		return false, 0
	}
	if frontmost().PID == pid {
		return true, time.Since(start)
	}
	runOnMain(func() {
		pooled(func() {
			app := class("NSRunningApplication").Send(selRunningApp, int32(pid))
			if app == 0 {
				return
			}
			// macOS 14+: the active app (VoxtNew, after a click on one of
			// its windows) hands activation over, then the target takes it.
			nsApp := class("NSApplication").Send(selSharedApp)
			if nsApp != 0 && objc.Send[bool](nsApp, selIsActive) && objc.Send[bool](nsApp, selResponds, selYield) {
				nsApp.Send(selYield, app)
			}
			objc.Send[bool](app, selActivateWith, uint64(0))
		})
	})
	deadline := start.Add(wait)
	for {
		if frontmost().PID == pid {
			return true, time.Since(start)
		}
		if time.Now().After(deadline) {
			return false, time.Since(start)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

var selWindowNumber = objc.RegisterName("windowNumber")

func windowNumber(handle uintptr) int {
	if handle == 0 || load() != nil {
		return 0
	}
	return int(objc.Send[int](objc.ID(handle), selWindowNumber))
}
