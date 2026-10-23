//go:build darwin

package permission

// The macOS side, through purego (as MyGo's internal/darwin does): no
// cgo. Frameworks are opened at the first use.

import (
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

var mac struct {
	once sync.Once
	err  error

	poolPush func() uintptr
	poolPop  func(uintptr)

	axIsProcessTrusted            func() bool
	axIsProcessTrustedWithOptions func(options uintptr) bool
	kAXTrustedCheckOptionPrompt   uintptr
	avMediaTypeAudio              uintptr
}

func load() error {
	mac.once.Do(func() {
		const rtld = purego.RTLD_GLOBAL | purego.RTLD_NOW
		libs := map[string]uintptr{}
		for name, path := range map[string]string{
			"objc":   "/usr/lib/libobjc.A.dylib",
			"appsvc": "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices",
			"av":     "/System/Library/Frameworks/AVFoundation.framework/AVFoundation",
		} {
			h, err := purego.Dlopen(path, rtld)
			if err != nil {
				mac.err = err
				return
			}
			libs[name] = h
		}
		purego.RegisterLibFunc(&mac.poolPush, libs["objc"], "objc_autoreleasePoolPush")
		purego.RegisterLibFunc(&mac.poolPop, libs["objc"], "objc_autoreleasePoolPop")
		purego.RegisterLibFunc(&mac.axIsProcessTrusted, libs["appsvc"], "AXIsProcessTrusted")
		purego.RegisterLibFunc(&mac.axIsProcessTrustedWithOptions, libs["appsvc"], "AXIsProcessTrustedWithOptions")
		mac.kAXTrustedCheckOptionPrompt = constant(libs["appsvc"], "kAXTrustedCheckOptionPrompt")
		mac.avMediaTypeAudio = constant(libs["av"], "AVMediaTypeAudio")
	})
	return mac.err
}

// constant reads an exported object constant (a CFStringRef / NSString *).
func constant(lib uintptr, name string) uintptr {
	p, err := purego.Dlsym(lib, name)
	if err != nil || p == 0 {
		return 0
	}
	return **(**uintptr)(unsafe.Pointer(&p))
}

// pooled runs fn inside an autorelease pool: objects autoreleased on a
// goroutine's thread are freed at its end.
func pooled(fn func()) {
	p := mac.poolPush()
	defer mac.poolPop(p)
	fn()
}

var (
	selAuthStatus = objc.RegisterName("authorizationStatusForMediaType:")
	selRequest    = objc.RegisterName("requestAccessForMediaType:completionHandler:")
	selNumberBool = objc.RegisterName("numberWithBool:")
	selDictWith   = objc.RegisterName("dictionaryWithObject:forKey:")
)

type native struct{}

// Native is the macOS implementation.
func Native() System { return native{} }

func (native) AccessibilityTrusted() bool {
	return load() == nil && mac.axIsProcessTrusted()
}

func (native) PromptAccessibility() {
	if load() != nil || mac.kAXTrustedCheckOptionPrompt == 0 {
		return
	}
	pooled(func() {
		yes := objc.ID(objc.GetClass("NSNumber")).Send(selNumberBool, true)
		opts := objc.ID(objc.GetClass("NSDictionary")).Send(selDictWith, yes, mac.kAXTrustedCheckOptionPrompt)
		mac.axIsProcessTrustedWithOptions(uintptr(opts))
	})
}

func (native) Microphone() State {
	if load() != nil || mac.avMediaTypeAudio == 0 {
		return Unknown
	}
	var st int
	pooled(func() {
		st = int(objc.Send[int](objc.ID(objc.GetClass("AVCaptureDevice")), selAuthStatus, mac.avMediaTypeAudio))
	})
	switch st {
	case 0:
		return NotDetermined
	case 1:
		return Restricted
	case 2:
		return Denied
	case 3:
		return Granted
	}
	return Unknown
}

// The completion handler of requestAccessForMediaType: calls every
// waiting done, so one block serves concurrent requests.
var (
	micMu   sync.Mutex
	micDone []func(bool)
)

func (native) RequestMicrophone(done func(bool)) {
	if load() != nil || mac.avMediaTypeAudio == 0 {
		go done(false)
		return
	}
	micMu.Lock()
	micDone = append(micDone, done)
	micMu.Unlock()
	block := objc.NewBlock(func(_ objc.Block, granted bool) {
		micMu.Lock()
		fns := micDone
		micDone = nil
		micMu.Unlock()
		for _, fn := range fns {
			go fn(granted)
		}
	})
	pooled(func() {
		objc.ID(objc.GetClass("AVCaptureDevice")).Send(selRequest, mac.avMediaTypeAudio, block)
	})
	block.Release()
}
