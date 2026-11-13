//go:build windows

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

const (
	elementGetRuntimeID = 4
	safeArrayI4         = 3
)

var (
	procSafeArrayAccessData   = oleaut32.NewProc("SafeArrayAccessData")
	procSafeArrayDestroy      = oleaut32.NewProc("SafeArrayDestroy")
	procSafeArrayGetDim       = oleaut32.NewProc("SafeArrayGetDim")
	procSafeArrayGetLBound    = oleaut32.NewProc("SafeArrayGetLBound")
	procSafeArrayGetUBound    = oleaut32.NewProc("SafeArrayGetUBound")
	procSafeArrayGetVarType   = oleaut32.NewProc("SafeArrayGetVartype")
	procSafeArrayUnaccessData = oleaut32.NewProc("SafeArrayUnaccessData")
)

func readFocusedText(automation *comObject, params proto.ReadParams) (proto.ReadResult, error) {
	element, err := focusedElement(automation)
	if err != nil {
		return proto.ReadResult{Error: proto.ReadNoFocus}, nil
	}
	defer element.release()
	identity, err := identifyElement(element)
	if err != nil {
		return proto.ReadResult{}, err
	}
	front, err := frontmostApp()
	if err != nil {
		return proto.ReadResult{}, err
	}
	if front.PID != params.PID || identity.processID != params.PID {
		return proto.ReadResult{Error: proto.ReadNotFront}, nil
	}
	if identity.isPassword {
		return proto.ReadResult{Error: proto.ReadSecure}, nil
	}
	text, readable := readElementText(element)
	if !readable {
		return proto.ReadResult{Error: proto.ReadUnreadable}, nil
	}
	elementID, err := runtimeElementID(element)
	if err != nil {
		return proto.ReadResult{}, err
	}
	result := cutReadRange(text, params.Range)
	result.Element = elementID
	return result, nil
}

func readElementText(element *comObject) (string, bool) {
	if pattern, err := currentPattern(element, patternText, &interfaceTextPattern); err == nil {
		document, documentErr := pattern.output(textPatternDocumentRange)
		pattern.release()
		if documentErr == nil {
			text, readErr := rangeText(document, maxFieldTextUnits+1)
			document.release()
			if readErr == nil {
				if !withinReadCeiling(text) {
					return "", false
				}
				if text != "" {
					return text, true
				}
				if value, ok := readElementValue(element); ok && value != "" {
					if !withinReadCeiling(value) {
						return "", false
					}
					return value, true
				}
				return "", true
			}
		}
	}
	value, ok := readElementValue(element)
	if !ok || !withinReadCeiling(value) {
		return "", false
	}
	return value, true
}

func readElementValue(element *comObject) (string, bool) {
	pattern, err := currentPattern(element, patternValue, &interfaceValuePattern)
	if err != nil {
		return "", false
	}
	defer pattern.release()
	var valueAddress *uint16
	result := pattern.call(valuePatternCurrentValue, uintptr(unsafe.Pointer(&valueAddress)))
	if int32(result) < 0 {
		return "", false
	}
	return readBSTR(valueAddress), true
}

func runtimeElementID(element *comObject) (int, error) {
	var array uintptr
	result := element.call(elementGetRuntimeID, uintptr(unsafe.Pointer(&array)))
	if int32(result) < 0 || array == 0 {
		return 0, hresultError("IUIAutomationElement.GetRuntimeId", result)
	}
	defer procSafeArrayDestroy.Call(array)
	dimensions, _, _ := procSafeArrayGetDim.Call(array)
	var valueType uint16
	result, _, _ = procSafeArrayGetVarType.Call(array, uintptr(unsafe.Pointer(&valueType)))
	if dimensions != 1 || int32(result) < 0 || valueType != safeArrayI4 {
		return 0, fmt.Errorf("UIA runtime ID has an unsupported shape")
	}
	var lowerBound, upperBound int32
	result, _, _ = procSafeArrayGetLBound.Call(array, 1, uintptr(unsafe.Pointer(&lowerBound)))
	if int32(result) < 0 {
		return 0, hresultError("SafeArrayGetLBound", result)
	}
	result, _, _ = procSafeArrayGetUBound.Call(array, 1, uintptr(unsafe.Pointer(&upperBound)))
	if int32(result) < 0 {
		return 0, hresultError("SafeArrayGetUBound", result)
	}
	count := int(upperBound - lowerBound + 1)
	if count <= 0 || count > 64 {
		return 0, fmt.Errorf("UIA runtime ID has %d integers", count)
	}
	var address unsafe.Pointer
	result, _, _ = procSafeArrayAccessData.Call(array, uintptr(unsafe.Pointer(&address)))
	if int32(result) < 0 || address == nil {
		return 0, hresultError("SafeArrayAccessData", result)
	}
	values := unsafe.Slice((*int32)(address), count)
	digest := sha256.New()
	for _, value := range values {
		var encoded [4]byte
		binary.LittleEndian.PutUint32(encoded[:], uint32(value))
		digest.Write(encoded[:])
	}
	procSafeArrayUnaccessData.Call(array)
	identifier := int(binary.LittleEndian.Uint32(digest.Sum(nil)[:4]) & 0x7fffffff)
	if identifier == 0 {
		identifier = 1
	}
	return identifier, nil
}
