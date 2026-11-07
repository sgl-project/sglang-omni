//go:build darwin

package macx

import (
	"net"
	"strconv"
	"sync"

	"github.com/ebitengine/purego"
)

// CoreFoundation's constants.
const (
	cfStringEncodingUTF8 = 0x08000100
	cfNumberIntType      = 9 // kCFNumberIntType
)

var cf struct {
	once sync.Once
	err  error

	copyProxySettings func() uintptr
	release           func(uintptr)
	getTypeID         func(uintptr) uint
	stringTypeID      func() uint
	numberTypeID      func() uint
	arrayTypeID       func() uint
	dictionaryGet     func(dictionary, key uintptr) uintptr
	stringCreate      func(allocator uintptr, text string, encoding uint32) uintptr
	stringGetCString  func(text uintptr, buffer *byte, size int, encoding uint32) uint8
	numberGetValue    func(number uintptr, numberType int, value *int32) uint8
	arrayCount        func(array uintptr) int
	arrayValueAt      func(array uintptr, index int) uintptr
}

func loadCF() error {
	cf.once.Do(func() {
		const rtld = purego.RTLD_GLOBAL | purego.RTLD_NOW
		foundation, err := purego.Dlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation", rtld)
		if err != nil {
			cf.err = err
			return
		}
		network, err := purego.Dlopen("/System/Library/Frameworks/CFNetwork.framework/CFNetwork", rtld)
		if err != nil {
			cf.err = err
			return
		}
		purego.RegisterLibFunc(&cf.copyProxySettings, network, "CFNetworkCopySystemProxySettings")
		purego.RegisterLibFunc(&cf.release, foundation, "CFRelease")
		purego.RegisterLibFunc(&cf.getTypeID, foundation, "CFGetTypeID")
		purego.RegisterLibFunc(&cf.stringTypeID, foundation, "CFStringGetTypeID")
		purego.RegisterLibFunc(&cf.numberTypeID, foundation, "CFNumberGetTypeID")
		purego.RegisterLibFunc(&cf.arrayTypeID, foundation, "CFArrayGetTypeID")
		purego.RegisterLibFunc(&cf.dictionaryGet, foundation, "CFDictionaryGetValue")
		purego.RegisterLibFunc(&cf.stringCreate, foundation, "CFStringCreateWithCString")
		purego.RegisterLibFunc(&cf.stringGetCString, foundation, "CFStringGetCString")
		purego.RegisterLibFunc(&cf.numberGetValue, foundation, "CFNumberGetValue")
		purego.RegisterLibFunc(&cf.arrayCount, foundation, "CFArrayGetCount")
		purego.RegisterLibFunc(&cf.arrayValueAt, foundation, "CFArrayGetValueAtIndex")
	})
	return cf.err
}

func systemProxies() Proxies {
	if loadCF() != nil {
		return Proxies{}
	}
	settings := cf.copyProxySettings()
	if settings == 0 {
		return Proxies{}
	}
	defer cf.release(settings)
	value := func(key string) uintptr {
		cfKey := cf.stringCreate(0, key, cfStringEncodingUTF8)
		defer cf.release(cfKey)
		return cf.dictionaryGet(settings, cfKey)
	}
	number := func(key string) int {
		var n int32
		if v := value(key); v != 0 && cf.getTypeID(v) == cf.numberTypeID() && cf.numberGetValue(v, cfNumberIntType, &n) != 0 {
			return int(n)
		}
		return 0
	}
	proxy := func(prefix string) string {
		host := cfString(value(prefix + "Proxy"))
		port := number(prefix + "Port")
		if number(prefix+"Enable") == 0 || host == "" || port <= 0 {
			return ""
		}
		return net.JoinHostPort(host, strconv.Itoa(port))
	}
	p := Proxies{HTTP: proxy("HTTP"), HTTPS: proxy("HTTPS"), ExcludeSimple: number("ExcludeSimpleHostnames") != 0}
	if list := value("ExceptionsList"); list != 0 && cf.getTypeID(list) == cf.arrayTypeID() {
		for i := range cf.arrayCount(list) {
			if s := cfString(cf.arrayValueAt(list, i)); s != "" {
				p.Exceptions = append(p.Exceptions, s)
			}
		}
	}
	return p
}

// cfString is a CFString's text, "" for anything else.
func cfString(text uintptr) string {
	if text == 0 || cf.getTypeID(text) != cf.stringTypeID() {
		return ""
	}
	buffer := make([]byte, 1024) // host names are at most 253 bytes
	if cf.stringGetCString(text, &buffer[0], len(buffer), cfStringEncodingUTF8) == 0 {
		return ""
	}
	for i, b := range buffer {
		if b == 0 {
			return string(buffer[:i])
		}
	}
	return ""
}
