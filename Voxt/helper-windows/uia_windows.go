//go:build windows

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"hash"
	"unicode/utf16"
	"unsafe"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

const (
	controlTypeButton   = 50000
	controlTypeComboBox = 50003
	controlTypeEdit     = 50004
	controlTypeText     = 50020
	controlTypeCustom   = 50025
	controlTypeGroup    = 50026
	controlTypeDocument = 50030
	controlTypeWindow   = 50032
	controlTypePane     = 50033
)

type elementIdentity struct {
	processID     int
	controlType   int
	framework     string
	isPassword    bool
	readOnly      bool
	readOnlyKnown bool
}

func identifyElement(element *comObject) (elementIdentity, error) {
	var identity elementIdentity
	var processID, controlType, password int32
	if result := element.call(elementCurrentProcessID, uintptr(unsafe.Pointer(&processID))); int32(result) < 0 {
		return identity, hresultError("CurrentProcessId", result)
	}
	if result := element.call(elementCurrentControlType, uintptr(unsafe.Pointer(&controlType))); int32(result) < 0 {
		return identity, hresultError("CurrentControlType", result)
	}
	if result := element.call(elementCurrentIsPassword, uintptr(unsafe.Pointer(&password))); int32(result) < 0 {
		return identity, hresultError("CurrentIsPassword", result)
	}
	identity.processID = int(processID)
	identity.controlType = int(controlType)
	identity.isPassword = password != 0
	var frameworkAddress *uint16
	if result := element.call(elementCurrentFramework, uintptr(unsafe.Pointer(&frameworkAddress))); int32(result) >= 0 {
		identity.framework = readBSTR(frameworkAddress)
	}
	if pattern, err := currentPattern(element, patternValue, &interfaceValuePattern); err == nil {
		defer pattern.release()
		var readOnly int32
		if result := pattern.call(valuePatternCurrentReadOnly, uintptr(unsafe.Pointer(&readOnly))); int32(result) >= 0 {
			identity.readOnlyKnown = true
			identity.readOnly = readOnly != 0
		}
	}
	return identity, nil
}

func accessibilityRole(identity elementIdentity) string {
	switch identity.controlType {
	case controlTypeEdit:
		return "AXTextField"
	case controlTypeDocument:
		return "AXTextArea"
	case controlTypeComboBox:
		return "AXComboBox"
	case controlTypeText:
		return "AXStaticText"
	case controlTypeButton:
		return "AXButton"
	case controlTypeWindow:
		return "AXWindow"
	case controlTypeCustom, controlTypeGroup, controlTypePane:
		return "AXGroup"
	default:
		return "AXGroup"
	}
}

func (identity elementIdentity) editable() bool {
	if identity.readOnly {
		return false
	}
	if identity.readOnlyKnown {
		return true
	}
	switch identity.controlType {
	case controlTypeEdit, controlTypeDocument, controlTypeComboBox:
		return true
	default:
		return false
	}
}

type textReading struct {
	available bool
	length    int
	digest    [sha256.Size]byte
}

type selectionReading struct {
	available bool
	location  int
	length    int
	digest    [sha256.Size]byte
}

type fieldReading struct {
	value     textReading
	text      textReading
	selection selectionReading
	problems  bool
}

func newTextReading(text string) textReading {
	return textReading{
		available: true,
		length:    len(utf16.Encode([]rune(text))),
		digest:    sha256.Sum256([]byte(text)),
	}
}

func rangeText(textRange *comObject, maximumUnits int) (string, error) {
	var textAddress *uint16
	result := textRange.call(textRangeGetText, uintptr(maximumUnits), uintptr(unsafe.Pointer(&textAddress)))
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
			reading.problems = true
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
		reading.problems = true
		return reading
	}
	defer document.release()
	text, err := rangeText(document, maxFieldTextUnits)
	if err != nil || len(utf16.Encode([]rune(text))) >= maxFieldTextUnits {
		reading.problems = true
	} else {
		reading.text = newTextReading(text)
	}
	reading.selection = readSelections(pattern, document, &reading.problems)
	return reading
}

func readSelections(pattern, document *comObject, problems *bool) selectionReading {
	var result selectionReading
	selections, err := pattern.output(textPatternGetSelection)
	if err != nil {
		*problems = true
		return result
	}
	defer selections.release()
	var selectionCount int32
	if callResult := selections.call(arrayLength, uintptr(unsafe.Pointer(&selectionCount))); int32(callResult) < 0 {
		*problems = true
		return result
	}
	if selectionCount == 0 {
		return result
	}
	digest := sha256.New()
	for index := int32(0); index < selectionCount; index++ {
		location, length, err := readSelectionOffset(selections, document, index, digest)
		if err != nil {
			*problems = true
			return selectionReading{}
		}
		if index == 0 {
			result.location = location
			result.length = length
		}
	}
	copy(result.digest[:], digest.Sum(nil))
	result.available = true
	return result
}

func readSelectionOffset(selections, document *comObject, index int32, digest hash.Hash) (int, int, error) {
	selection, err := selections.output(arrayElement, uintptr(index))
	if err != nil {
		return 0, 0, err
	}
	defer selection.release()
	prefix, err := document.output(textRangeClone)
	if err != nil {
		return 0, 0, err
	}
	defer prefix.release()
	result := prefix.call(textRangeMoveEndpointByRange,
		textPatternEndpointEnd,
		uintptr(unsafe.Pointer(selection)),
		textPatternEndpointStart,
	)
	if int32(result) < 0 {
		return 0, 0, hresultError("MoveEndpointByRange", result)
	}
	prefixText, err := rangeText(prefix, -1)
	if err != nil {
		return 0, 0, err
	}
	selectedText, err := rangeText(selection, -1)
	if err != nil {
		return 0, 0, err
	}
	location := len(utf16.Encode([]rune(prefixText)))
	length := len(utf16.Encode([]rune(selectedText)))
	var units [8]byte
	binary.LittleEndian.PutUint64(units[:], uint64(location))
	digest.Write(units[:])
	binary.LittleEndian.PutUint64(units[:], uint64(length))
	digest.Write(units[:])
	digest.Write([]byte(selectedText))
	return location, length, nil
}

func focusResult(automation *comObject) (proto.FocusResult, error) {
	element, err := focusedElement(automation)
	if err != nil {
		front, frontErr := frontmostApp()
		if frontErr != nil {
			return proto.FocusResult{}, err
		}
		return proto.FocusResult{PID: front.PID, App: &front, Error: "no focused element"}, nil
	}
	defer element.release()
	identity, err := identifyElement(element)
	if err != nil {
		return proto.FocusResult{}, err
	}
	app := appForProcess(uint32(identity.processID))
	result := proto.FocusResult{
		PID:      identity.processID,
		App:      &app,
		Role:     accessibilityRole(identity),
		Editable: identity.editable(),
	}
	if identity.isPassword {
		result.Subrole = "AXSecureTextField"
		return result, nil
	}
	reading := readField(element)
	if reading.value.available {
		result.ValueLength = reading.value.length
	} else if reading.text.available {
		result.ValueLength = reading.text.length
	}
	if reading.selection.available {
		result.Selection = &proto.TextRange{Location: reading.selection.location, Length: reading.selection.length}
	}
	return result, nil
}

func (reading fieldReading) state() *proto.FieldState {
	state := &proto.FieldState{}
	if reading.value.available {
		state.Length = reading.value.length
	} else if reading.text.available {
		state.Length = reading.text.length
	}
	if reading.selection.available {
		state.Selection = &proto.TextRange{Location: reading.selection.location, Length: reading.selection.length}
	}
	return state
}

func changedFieldSignal(before, after fieldReading) bool {
	if before.value.available && after.value.available && before.value != after.value {
		return true
	}
	if before.text.available && after.text.available && before.text != after.text {
		return true
	}
	return before.selection.available && after.selection.available && before.selection != after.selection
}

func comparableFieldSignal(before, after fieldReading) bool {
	return before.value.available && after.value.available ||
		before.text.available && after.text.available ||
		before.selection.available && after.selection.available
}

func readableFieldSignal(reading fieldReading) bool {
	return reading.value.available || reading.text.available || reading.selection.available
}
