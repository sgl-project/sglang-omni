//go:build windows && (amd64 || arm64)

package windows

import (
	"strings"
	"sync"
	"syscall"
	"unsafe"
)

// envOptions is an ICoreWebView2EnvironmentOptions, a COM object of ours
// that the WebView2 environment reads its options from as it is created:
// the language of the browser, which is the language of its own texts (the
// context menus, the dialogs of pages) and the first of the pages'
// navigator.languages and Accept-Language. Its other options are those of
// an environment created without options: no further arguments, no
// version the runtime must be, no single sign-on. It implements no later
// version of the interface, which the environment asks for and does
// without.
//
// The objects share one vtable, whose callbacks are created once; they
// stay reachable through envOptionsLive until COM releases them.
type envOptions struct {
	vtbl     *[11]uintptr
	refs     int32
	language []uint16 // NUL-terminated, or nil
}

var (
	iidICoreWebView2EnvironmentOptions = guid("2fde08a8-1e9a-4766-8c05-95a9ceb9d1c5")

	envOptionsVtbl [11]uintptr
	envOptionsOnce sync.Once
	envOptionsMu   sync.Mutex
	envOptionsLive = map[uintptr]*envOptions{}
)

const ePointer = 0x80004003 // E_POINTER

// newEnvOptions returns an options object of language, holding one
// reference, which the caller releases.
func newEnvOptions(language string) uintptr {
	envOptionsOnce.Do(func() {
		str := func(get func(o *envOptions) []uint16) uintptr {
			return syscall.NewCallback(func(this, out uintptr) uintptr {
				if out == 0 {
					return ePointer
				}
				*(*uintptr)(native(out)) = coTaskString(get(envOptionsOf(this)))
				return sOK
			})
		}
		ignore := syscall.NewCallback(func(this, value uintptr) uintptr { return sOK })
		envOptionsVtbl = [11]uintptr{
			syscall.NewCallback(func(this, riid, out uintptr) uintptr {
				if out == 0 {
					return ePointer
				}
				iid := *(*GUID)(native(riid))
				if iid != iidIUnknown && iid != iidICoreWebView2EnvironmentOptions {
					*(*uintptr)(native(out)) = 0
					return eNoInterface
				}
				*(*uintptr)(native(out)) = this
				envOptionsRef(this, 1)
				return sOK
			}),
			syscall.NewCallback(func(this uintptr) uintptr { return uintptr(envOptionsRef(this, 1)) }),
			syscall.NewCallback(func(this uintptr) uintptr { return uintptr(envOptionsRef(this, -1)) }),
			str(func(*envOptions) []uint16 { return nil }), // get_AdditionalBrowserArguments
			ignore, // put_AdditionalBrowserArguments
			str(func(o *envOptions) []uint16 { // get_Language
				if o == nil {
					return nil
				}
				return o.language
			}),
			ignore, // put_Language
			// The version of the runtime the app needs at least, which the
			// official loader picks a runtime by: as Microsoft's options
			// class does, never none. MyGo needs no newer one than the
			// first release of WebView2.
			str(func(*envOptions) []uint16 { return targetBrowserVersion }), // get_TargetCompatibleBrowserVersion
			ignore, // put_TargetCompatibleBrowserVersion
			syscall.NewCallback(func(this, out uintptr) uintptr { // get_AllowSingleSignOnUsingOSPrimaryAccount
				if out == 0 {
					return ePointer
				}
				*(*int32)(native(out)) = 0
				return sOK
			}),
			ignore, // put_AllowSingleSignOnUsingOSPrimaryAccount
		}
	})
	o := &envOptions{vtbl: &envOptionsVtbl, refs: 1}
	if language != "" {
		o.language = utf16z(chromiumLanguage(language))
	}
	p := uintptr(unsafe.Pointer(o))
	envOptionsMu.Lock()
	envOptionsLive[p] = o
	envOptionsMu.Unlock()
	return p
}

func envOptionsOf(this uintptr) *envOptions {
	envOptionsMu.Lock()
	defer envOptionsMu.Unlock()
	return envOptionsLive[this]
}

func envOptionsRef(this uintptr, delta int32) int32 {
	envOptionsMu.Lock()
	defer envOptionsMu.Unlock()
	o := envOptionsLive[this]
	if o == nil {
		return 0
	}
	o.refs += delta
	if o.refs <= 0 {
		delete(envOptionsLive, this)
		return 0
	}
	return o.refs
}

// coTaskString returns a copy of a NUL-terminated string that the caller
// frees with CoTaskMemFree, as COM getters of strings return them, or 0
// for none.
func coTaskString(s []uint16) uintptr {
	if len(s) == 0 {
		return 0
	}
	p, _, _ := procCoTaskMemAlloc.Call(uintptr(len(s) * 2))
	if p == 0 {
		return 0
	}
	copy(unsafe.Slice((*uint16)(native(p)), len(s)), s)
	return p
}

// targetBrowserVersion is the version of the first release of WebView2's
// runtime.
var targetBrowserVersion = utf16z("86.0.616.0")

// chromiumLanguage returns the language the browser of WebView2 knows a
// tag by: Chromium names the scripts of Chinese by the regions that write
// them, zh-CN and zh-TW, and reads any other zh as zh-CN.
func chromiumLanguage(tag string) string {
	parts := strings.Split(tag, "-")
	if !strings.EqualFold(parts[0], "zh") {
		return tag
	}
	for _, p := range parts[1:] {
		switch strings.ToUpper(p) {
		case "HK", "MO":
			return "zh-" + strings.ToUpper(p)
		case "TW":
			return "zh-TW"
		}
	}
	for _, p := range parts[1:] {
		if strings.EqualFold(p, "Hant") {
			return "zh-TW"
		}
	}
	return "zh-CN"
}
