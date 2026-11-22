//go:build darwin

package lang

import (
	"sync"

	"github.com/ebitengine/purego"
)

const chineseCFStringEncodingUTF8 = 0x08000100

var chineseCF struct {
	once sync.Once
	err  error

	create            func(allocator uintptr, text string, encoding uint32) uintptr
	createMutableCopy func(allocator uintptr, maxLength int, text uintptr) uintptr
	transform         func(text, rangePointer, transformID uintptr, reverse bool) uint8
	length            func(text uintptr) int
	maximumSize       func(length int, encoding uint32) int
	getCString        func(text uintptr, buffer *byte, size int, encoding uint32) uint8
	release           func(uintptr)
}

func loadChineseCF() error {
	chineseCF.once.Do(func() {
		foundation, err := purego.Dlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			chineseCF.err = err
			return
		}
		purego.RegisterLibFunc(&chineseCF.create, foundation, "CFStringCreateWithCString")
		purego.RegisterLibFunc(&chineseCF.createMutableCopy, foundation, "CFStringCreateMutableCopy")
		purego.RegisterLibFunc(&chineseCF.transform, foundation, "CFStringTransform")
		purego.RegisterLibFunc(&chineseCF.length, foundation, "CFStringGetLength")
		purego.RegisterLibFunc(&chineseCF.maximumSize, foundation, "CFStringGetMaximumSizeForEncoding")
		purego.RegisterLibFunc(&chineseCF.getCString, foundation, "CFStringGetCString")
		purego.RegisterLibFunc(&chineseCF.release, foundation, "CFRelease")
	})
	return chineseCF.err
}

func convertChineseScript(text, transformID string) (string, bool) {
	if text == "" {
		return text, true
	}
	if loadChineseCF() != nil {
		return "", false
	}
	source := chineseCF.create(0, text, chineseCFStringEncodingUTF8)
	if source == 0 {
		return "", false
	}
	defer chineseCF.release(source)
	mutable := chineseCF.createMutableCopy(0, 0, source)
	if mutable == 0 {
		return "", false
	}
	defer chineseCF.release(mutable)
	identifier := chineseCF.create(0, transformID, chineseCFStringEncodingUTF8)
	if identifier == 0 {
		return "", false
	}
	defer chineseCF.release(identifier)
	if chineseCF.transform(mutable, 0, identifier, false) == 0 {
		return text, true
	}
	buffer := make([]byte, chineseCF.maximumSize(chineseCF.length(mutable), chineseCFStringEncodingUTF8)+1)
	if len(buffer) == 0 || chineseCF.getCString(mutable, &buffer[0], len(buffer), chineseCFStringEncodingUTF8) == 0 {
		return "", false
	}
	for index, value := range buffer {
		if value == 0 {
			return string(buffer[:index]), true
		}
	}
	return "", false
}
