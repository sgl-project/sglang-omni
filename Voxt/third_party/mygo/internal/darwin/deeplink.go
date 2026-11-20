//go:build darwin

package darwin

import (
	"fmt"
	"strings"

	"github.com/egoist/mygo/internal/platform"
)

// The URL schemes of a bundle are declared in its Info.plist
// (CFBundleURLTypes), which Launch Services registers. What an app can do
// at run time is become the default handler of one of them.

func (a appController) RegisterURLScheme(scheme, id, name string) error {
	var err error
	withPool(func() {
		bundle := send(class("NSBundle"), "mainBundle")
		if !declaresScheme(bundle, scheme) {
			err = fmt.Errorf("mygo: the app does not declare the URL scheme %q: list it in urlSchemes in mygo.config.ts", scheme)
			return
		}
		if status := lsSetDefaultHandlerForURLScheme(nsString(scheme), send(bundle, "bundleIdentifier")); status != 0 {
			err = fmt.Errorf("mygo: cannot handle the URL scheme %q (Launch Services error %d)", scheme, status)
		}
	})
	return err
}

// UnregisterURLScheme is not supported: the scheme belongs to the bundle.
func (a appController) UnregisterURLScheme(string, string, string) error {
	return platform.ErrUnsupported
}

func (a appController) IsURLSchemeRegistered(scheme, id, name string) bool {
	ok := false
	withPool(func() {
		handler := lsCopyDefaultHandlerForURLScheme(nsString(scheme))
		if handler == 0 {
			return
		}
		defer cfRelease(uintptr(handler))
		ok = strings.EqualFold(goString(handler), goString(send(send(class("NSBundle"), "mainBundle"), "bundleIdentifier")))
	})
	return ok
}

// declaresScheme reports whether the Info.plist of bundle lists scheme.
func declaresScheme(bundle id, scheme string) bool {
	for _, t := range arrayItems(send(bundle, "objectForInfoDictionaryKey:", uintptr(nsString("CFBundleURLTypes")))) {
		for _, s := range arrayItems(send(t, "objectForKey:", uintptr(nsString("CFBundleURLSchemes")))) {
			if strings.EqualFold(goString(s), scheme) {
				return true
			}
		}
	}
	return false
}
