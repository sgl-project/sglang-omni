//go:build windows

package main

import (
	"fmt"
	"os"
	"time"
	"unsafe"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

type selectionProbe struct {
	available    bool
	hasSelection bool
	text         string
}

type selectionDecision struct {
	text     string
	via      string
	copyNext bool
}

func decideSelection(probe selectionProbe, allowCopy bool) selectionDecision {
	if probe.text != "" {
		return selectionDecision{text: probe.text, via: proto.ViaAXRange}
	}
	if probe.available && probe.hasSelection && allowCopy {
		return selectionDecision{via: proto.ViaNone, copyNext: true}
	}
	return selectionDecision{via: proto.ViaNone}
}

func (server *helperServer) selection(automation *comObject, params proto.SelectionParams) proto.SelectionResult {
	started := time.Now()
	result := proto.SelectionResult{Via: proto.ViaNone}
	finish := func() proto.SelectionResult {
		result.MS = millisecondsSince(started)
		return result
	}
	element, err := focusedElement(automation)
	if err != nil {
		result.Steps = append(result.Steps, "focus=nil")
		return finish()
	}
	defer element.release()
	identity, err := identifyElement(element)
	if err == nil {
		result.Steps = append(result.Steps, "focus="+accessibilityRole(identity))
		if identity.isPassword {
			result.Steps = append(result.Steps, "secure")
			return finish()
		}
	}
	probe := readSelectionProbe(element)
	switch {
	case !probe.available:
		result.Steps = append(result.Steps, "range=nil")
	case !probe.hasSelection:
		result.Steps = append(result.Steps, "caret-only")
	default:
		result.Steps = append(result.Steps, "range=nonempty")
	}
	decision := decideSelection(probe, params.AllowCopy)
	if !decision.copyNext {
		result.Text = decision.text
		result.Via = decision.via
		return finish()
	}
	timeoutMS := params.CopyTimeoutMS
	if timeoutMS == 0 {
		timeoutMS = 300
	}
	text, note, err := server.copySelection(time.Duration(max(0, timeoutMS)) * time.Millisecond)
	result.Steps = append(result.Steps, note)
	if err != nil {
		fmt.Fprintln(os.Stderr, "selection copy:", err)
		return finish()
	}
	if text != "" {
		result.Text = text
		result.Via = proto.ViaCopy
	}
	return finish()
}

func readSelectionProbe(element *comObject) selectionProbe {
	pattern, err := currentPattern(element, patternText, &interfaceTextPattern)
	if err != nil {
		return selectionProbe{}
	}
	defer pattern.release()
	selections, err := pattern.output(textPatternGetSelection)
	if err != nil {
		return selectionProbe{}
	}
	defer selections.release()
	var selectionCount int32
	if result := selections.call(arrayLength, uintptr(unsafe.Pointer(&selectionCount))); int32(result) < 0 || selectionCount == 0 {
		return selectionProbe{}
	}
	selection, err := selections.output(arrayElement, 0)
	if err != nil {
		return selectionProbe{}
	}
	defer selection.release()
	var same int32
	result := selection.call(textRangeCompareEndpoints,
		textPatternEndpointStart,
		uintptr(unsafe.Pointer(selection)),
		textPatternEndpointEnd,
		uintptr(unsafe.Pointer(&same)),
	)
	if int32(result) < 0 {
		return selectionProbe{}
	}
	probe := selectionProbe{available: true, hasSelection: same != 0}
	if !probe.hasSelection {
		return probe
	}
	probe.text, _ = rangeText(selection, -1)
	return probe
}

func (server *helperServer) copySelection(timeout time.Duration) (string, string, error) {
	snapshot, err := snapshotClipboard()
	if err != nil {
		return "", "copy: snapshot failed", err
	}
	defer snapshot.discard()
	sequence, _, _ := procGetClipboardSequence.Call()
	before := uint32(sequence)
	if err := sendKeyCombo(virtualKeyC, []string{proto.ModControl}); err != nil {
		return "", "copy: key failed", err
	}
	deadline := time.Now().Add(timeout)
	changed := false
	for {
		sequence, _, _ = procGetClipboardSequence.Call()
		if uint32(sequence) != before {
			changed = true
			break
		}
		if !time.Now().Before(deadline) || !server.waitPaste(10*time.Millisecond) {
			break
		}
	}
	if !changed {
		return "", "copy: clipboard unchanged", nil
	}
	text, readErr := clipboardText()
	restoreErr := snapshot.writeBack()
	if readErr != nil {
		return "", "copy: read failed", readErr
	}
	if restoreErr != nil {
		return "", "copy: restore failed", restoreErr
	}
	return text, "copy: changed", nil
}

func clipboardObject() (*comObject, error) {
	deadline := time.Now().Add(250 * time.Millisecond)
	for {
		var dataObject *comObject
		result, _, _ := procOleGetClipboard.Call(uintptr(unsafe.Pointer(&dataObject)))
		if int32(result) >= 0 {
			return dataObject, nil
		}
		if uint32(result) != clipboardCantOpen && uint32(result) != clipboardCantClose || !time.Now().Before(deadline) {
			return nil, hresultError("OleGetClipboard", result)
		}
		waitForClipboardMessage()
	}
}
