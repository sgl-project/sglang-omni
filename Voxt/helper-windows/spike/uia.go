//go:build windows

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"syscall"
	"unicode/utf16"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	patternValue             = 10002
	patternText              = 10014
	controlTypeEdit          = 50004
	controlTypeDocument      = 50030
	treeScopeDescendants     = 4
	textPatternEndpointStart = 0
	textPatternEndpointEnd   = 1

	comRelease                    = 2
	automationCompareElements     = 3
	automationElementFromHandle   = 6
	automationFocusedElement      = 8
	automationCreateTrueCondition = 21
	elementSetFocus               = 3
	elementFindAll                = 6
	elementCurrentPatternAs       = 14
	elementCurrentProcessID       = 20
	elementCurrentControlType     = 21
	elementCurrentName            = 23
	elementCurrentIsPassword      = 35
	elementCurrentFramework       = 40
	arrayLength                   = 3
	arrayElement                  = 4
	valuePatternCurrentValue      = 4
	valuePatternCurrentReadOnly   = 5
	textPatternGetSelection       = 5
	textPatternDocumentRange      = 7
	textRangeClone                = 3
	textRangeGetText              = 12
	textRangeMoveEndpointByRange  = 15
)

var (
	classAutomation       = windows.GUID{Data1: 0xff48dba4, Data2: 0x60ef, Data3: 0x4201, Data4: [8]byte{0xaa, 0x87, 0x54, 0x10, 0x3e, 0xef, 0x59, 0x4e}}
	interfaceAutomation   = windows.GUID{Data1: 0x30cbe57d, Data2: 0xd9d0, Data3: 0x452a, Data4: [8]byte{0xab, 0x13, 0x7a, 0xc5, 0xac, 0x48, 0x25, 0xee}}
	interfaceValuePattern = windows.GUID{Data1: 0xa94cd8b1, Data2: 0x0844, Data3: 0x4cd6, Data4: [8]byte{0x9d, 0x2d, 0x64, 0x05, 0x37, 0xab, 0x39, 0xe9}}
	interfaceTextPattern  = windows.GUID{Data1: 0x32eba289, Data2: 0x3583, Data3: 0x42c9, Data4: [8]byte{0x9c, 0x59, 0x3b, 0x6d, 0x9a, 0x1e, 0x9b, 0x6a}}
)

type comObject struct {
	vtable *[64]uintptr
}

func (object *comObject) call(method int, arguments ...uintptr) uintptr {
	if object == nil {
		return 0
	}
	parameters := append([]uintptr{uintptr(unsafe.Pointer(object))}, arguments...)
	result, _, _ := syscall.SyscallN(object.vtable[method], parameters...)
	return result
}

func (object *comObject) release() {
	object.call(comRelease)
}

func (object *comObject) output(method int, arguments ...uintptr) (*comObject, error) {
	var output *comObject
	arguments = append(arguments, uintptr(unsafe.Pointer(&output)))
	result := object.call(method, arguments...)
	if int32(result) < 0 {
		return nil, hresultError("UIA output", result)
	}
	if output == nil {
		return nil, fmt.Errorf("UIA returned no object")
	}
	return output, nil
}

func newAutomation() (*comObject, error) {
	result, _, _ := procCoInitializeEx.Call(0, windows.COINIT_APARTMENTTHREADED)
	if int32(result) < 0 {
		return nil, hresultError("CoInitializeEx", result)
	}
	var automation *comObject
	result, _, _ = procCoCreateInstance.Call(
		uintptr(unsafe.Pointer(&classAutomation)),
		0,
		windows.CLSCTX_INPROC_SERVER,
		uintptr(unsafe.Pointer(&interfaceAutomation)),
		uintptr(unsafe.Pointer(&automation)),
	)
	if int32(result) < 0 {
		windows.CoUninitialize()
		return nil, hresultError("CoCreateInstance(CUIAutomation)", result)
	}
	return automation, nil
}

func elementFromWindow(automation *comObject, handle uintptr) (*comObject, error) {
	return automation.output(automationElementFromHandle, handle)
}

func focusedElement(automation *comObject) (*comObject, error) {
	return automation.output(automationFocusedElement)
}

func sameElement(automation *comObject, first *comObject, second *comObject) (bool, error) {
	var same int32
	result := automation.call(automationCompareElements,
		uintptr(unsafe.Pointer(first)),
		uintptr(unsafe.Pointer(second)),
		uintptr(unsafe.Pointer(&same)),
	)
	if int32(result) < 0 {
		return false, hresultError("CompareElements", result)
	}
	return same != 0, nil
}

func currentPattern(element *comObject, patternID int, interfaceID *windows.GUID) (*comObject, error) {
	var pattern *comObject
	result := element.call(elementCurrentPatternAs,
		uintptr(patternID),
		uintptr(unsafe.Pointer(interfaceID)),
		uintptr(unsafe.Pointer(&pattern)),
	)
	if int32(result) < 0 {
		return nil, hresultError("GetCurrentPatternAs", result)
	}
	if pattern == nil {
		return nil, fmt.Errorf("pattern %d is unavailable", patternID)
	}
	return pattern, nil
}

type elementIdentity struct {
	processID   int32
	controlType int32
	framework   string
	name        string
	isPassword  bool
	readOnly    string
}

func identifyElement(element *comObject) elementIdentity {
	identity := elementIdentity{readOnly: "unknown"}
	element.call(elementCurrentProcessID, uintptr(unsafe.Pointer(&identity.processID)))
	element.call(elementCurrentControlType, uintptr(unsafe.Pointer(&identity.controlType)))
	var frameworkAddress *uint16
	if int32(element.call(elementCurrentFramework, uintptr(unsafe.Pointer(&frameworkAddress)))) >= 0 {
		identity.framework = readBSTR(frameworkAddress)
	}
	var nameAddress *uint16
	if int32(element.call(elementCurrentName, uintptr(unsafe.Pointer(&nameAddress)))) >= 0 {
		identity.name = readBSTR(nameAddress)
	}
	var isPassword int32
	if int32(element.call(elementCurrentIsPassword, uintptr(unsafe.Pointer(&isPassword)))) >= 0 {
		identity.isPassword = isPassword != 0
	}
	if pattern, err := currentPattern(element, patternValue, &interfaceValuePattern); err == nil {
		defer pattern.release()
		var readOnly int32
		if int32(pattern.call(valuePatternCurrentReadOnly, uintptr(unsafe.Pointer(&readOnly)))) >= 0 {
			identity.readOnly = fmt.Sprint(readOnly != 0)
		}
	}
	return identity
}

func controlTypeName(controlType int32) string {
	names := map[int32]string{
		50000: "Button", 50001: "Calendar", 50002: "CheckBox", 50003: "ComboBox",
		50004: "Edit", 50005: "Hyperlink", 50006: "Image", 50007: "ListItem",
		50008: "List", 50009: "Menu", 50010: "MenuBar", 50011: "MenuItem",
		50015: "Slider", 50018: "Tab", 50020: "Text", 50021: "ToolBar",
		50022: "ToolTip", 50025: "Custom", 50026: "Group", 50030: "Document",
		50031: "SplitButton", 50032: "Window", 50033: "Pane", 50036: "Table",
		50037: "TitleBar", 50038: "Separator",
	}
	if name := names[controlType]; name != "" {
		return fmt.Sprintf("%s(%d)", name, controlType)
	}
	return fmt.Sprintf("ControlType(%d)", controlType)
}

func focusEditor(automation *comObject, windowHandle uintptr) error {
	procSetForegroundWindow.Call(windowHandle)
	root, err := elementFromWindow(automation, windowHandle)
	if err != nil {
		return err
	}
	defer root.release()
	condition, err := automation.output(automationCreateTrueCondition)
	if err != nil {
		return err
	}
	defer condition.release()
	elements, err := root.output(elementFindAll, treeScopeDescendants, uintptr(unsafe.Pointer(condition)))
	if err != nil {
		return err
	}
	defer elements.release()
	var count int32
	if result := elements.call(arrayLength, uintptr(unsafe.Pointer(&count))); int32(result) < 0 {
		return hresultError("IUIAutomationElementArray.Length", result)
	}
	candidates := []*comObject{root}
	defer func() {
		for _, candidate := range candidates[1:] {
			candidate.release()
		}
	}()
	for index := int32(0); index < count; index++ {
		candidate, err := elements.output(arrayElement, uintptr(index))
		if err != nil {
			return err
		}
		candidates = append(candidates, candidate)
	}
	for _, candidate := range candidates {
		identity := identifyElement(candidate)
		if identity.controlType != controlTypeEdit && identity.controlType != controlTypeDocument || identity.readOnly == "true" {
			continue
		}
		if result := candidate.call(elementSetFocus); int32(result) < 0 {
			return hresultError("IUIAutomationElement.SetFocus", result)
		}
		return nil
	}
	return nil
}

type textReading struct {
	available bool
	length    int
	digest    string
}

type selectionReading struct {
	available bool
	location  int
	length    int
}

type fieldReading struct {
	value     textReading
	text      textReading
	selection selectionReading
	problems  []string
}

func newTextReading(text string) textReading {
	digest := sha256.Sum256([]byte(text))
	return textReading{
		available: true,
		length:    len(utf16.Encode([]rune(text))),
		digest:    hex.EncodeToString(digest[:8]),
	}
}

func rangeText(textRange *comObject) (string, error) {
	var textAddress *uint16
	result := textRange.call(textRangeGetText, uintptr(^uint32(0)), uintptr(unsafe.Pointer(&textAddress)))
	if int32(result) < 0 {
		return "", hresultError("IUIAutomationTextRange.GetText", result)
	}
	return readBSTR(textAddress), nil
}

func readField(element *comObject) fieldReading {
	var reading fieldReading
	if pattern, err := currentPattern(element, patternValue, &interfaceValuePattern); err == nil {
		var valueAddress *uint16
		result := pattern.call(valuePatternCurrentValue, uintptr(unsafe.Pointer(&valueAddress)))
		pattern.release()
		if int32(result) < 0 {
			reading.problems = append(reading.problems, fmt.Sprintf("value HRESULT 0x%08x", uint32(result)))
		} else {
			reading.value = newTextReading(readBSTR(valueAddress))
		}
	}
	pattern, err := currentPattern(element, patternText, &interfaceTextPattern)
	if err != nil {
		return reading
	}
	defer pattern.release()
	document, err := pattern.output(textPatternDocumentRange)
	if err != nil {
		reading.problems = append(reading.problems, "document range unavailable")
		return reading
	}
	defer document.release()
	text, err := rangeText(document)
	if err != nil {
		reading.problems = append(reading.problems, err.Error())
	} else {
		reading.text = newTextReading(text)
	}
	selections, err := pattern.output(textPatternGetSelection)
	if err != nil {
		reading.problems = append(reading.problems, "selection unavailable")
		return reading
	}
	defer selections.release()
	var selectionCount int32
	selections.call(arrayLength, uintptr(unsafe.Pointer(&selectionCount)))
	if selectionCount == 0 {
		return reading
	}
	selection, err := selections.output(arrayElement, 0)
	if err != nil {
		reading.problems = append(reading.problems, "selection range unavailable")
		return reading
	}
	defer selection.release()
	prefix, err := document.output(textRangeClone)
	if err != nil {
		reading.problems = append(reading.problems, "selection prefix unavailable")
		return reading
	}
	defer prefix.release()
	result := prefix.call(textRangeMoveEndpointByRange,
		textPatternEndpointEnd,
		uintptr(unsafe.Pointer(selection)),
		textPatternEndpointStart,
	)
	if int32(result) < 0 {
		reading.problems = append(reading.problems, fmt.Sprintf("selection offset HRESULT 0x%08x", uint32(result)))
		return reading
	}
	prefixText, prefixErr := rangeText(prefix)
	selectedText, selectionErr := rangeText(selection)
	if prefixErr != nil || selectionErr != nil {
		reading.problems = append(reading.problems, "selection text unavailable")
		return reading
	}
	reading.selection = selectionReading{
		available: true,
		location:  len(utf16.Encode([]rune(prefixText))),
		length:    len(utf16.Encode([]rune(selectedText))),
	}
	return reading
}

func (reading fieldReading) String() string {
	textState := func(state textReading) string {
		if !state.available {
			return "-"
		}
		return fmt.Sprintf("len=%d sha256=%s", state.length, state.digest)
	}
	selection := "-"
	if reading.selection.available {
		selection = fmt.Sprintf("%d+%d", reading.selection.location, reading.selection.length)
	}
	result := fmt.Sprintf("value[%s] text[%s] selection=%s", textState(reading.value), textState(reading.text), selection)
	if len(reading.problems) > 0 {
		result += " problems=" + strings.Join(reading.problems, ",")
	}
	return result
}
