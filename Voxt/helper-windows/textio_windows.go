//go:build windows

package main

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

const pastePollInterval = 25 * time.Millisecond

func (server *helperServer) serveTextIO(message proto.Message) {
	switch message.Method {
	case proto.MethodTextFocus:
		var result proto.FocusResult
		var readErr error
		if !server.automation.call(func(automation *comObject) {
			result, readErr = focusResult(automation)
		}) {
			server.fail(message.ID, proto.CodeInternal, "The UI Automation thread stopped")
			return
		}
		if readErr != nil {
			server.fail(message.ID, proto.CodeInternal, "Could not read the focused element: %v", readErr)
			return
		}
		server.result(message.ID, result)
	case proto.MethodTextSelection:
		var params proto.SelectionParams
		if err := message.DecodeParams(&params); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
			return
		}
		var result proto.SelectionResult
		if !server.automation.call(func(automation *comObject) {
			result = server.selection(automation, params)
		}) {
			server.fail(message.ID, proto.CodeInternal, "The UI Automation thread stopped")
			return
		}
		server.result(message.ID, result)
	case proto.MethodTextRead:
		var params proto.ReadParams
		if err := message.DecodeParams(&params); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
			return
		}
		var result proto.ReadResult
		var readErr error
		if !server.automation.call(func(automation *comObject) {
			result, readErr = readFocusedText(automation, params)
		}) {
			server.fail(message.ID, proto.CodeInternal, "The UI Automation thread stopped")
			return
		}
		if readErr != nil {
			server.fail(message.ID, proto.CodeInternal, "Could not read the focused text: %v", readErr)
			return
		}
		server.result(message.ID, result)
	case proto.MethodTextPaste:
		var params proto.PasteParams
		if err := message.DecodeParams(&params); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
			return
		}
		if err := validateKeyPresses(params.Press); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad key press: %v", err)
			return
		}
		var result proto.PasteResult
		var pasteErr error
		if !server.automation.call(func(automation *comObject) {
			result, pasteErr = server.paste(automation, params)
		}) {
			server.fail(message.ID, proto.CodeInternal, "The UI Automation thread stopped")
			return
		}
		if pasteErr != nil {
			server.fail(message.ID, proto.CodeInternal, "Could not paste: %v", pasteErr)
			return
		}
		server.result(message.ID, result)
	}
}

func (server *helperServer) paste(automation *comObject, params proto.PasteParams) (result proto.PasteResult, returnErr error) {
	started := time.Now()
	front, frontErr := frontmostApp()
	watched, focusErr := focusedElement(automation)
	var identity elementIdentity
	if focusErr == nil {
		defer watched.release()
		identity, focusErr = identifyElement(watched)
	}
	if params.TargetPID != 0 {
		targetReached := frontErr == nil && front.PID == params.TargetPID || focusErr == nil && identity.processID == params.TargetPID
		if !targetReached {
			result.MS = millisecondsSince(started)
			if frontErr == nil {
				result.Front = &front
			} else {
				return result, frontErr
			}
			return result, nil
		}
	}
	if focusErr == nil && identity.isPassword {
		return result, fmt.Errorf("refusing to paste into a password field")
	}

	var before fieldReading
	verify := params.VerifyMS > 0 && focusErr == nil
	if verify {
		before = readField(watched)
	}
	snapshot, err := snapshotAndSetClipboard(params.Text, !params.KeepClipboard)
	if err != nil {
		return result, err
	}
	completed := false
	defer func() {
		if !completed && snapshot != nil && snapshot.dataObject != nil {
			snapshot.restore()
		}
	}()
	if err := sendKeyCombo(virtualKeyV, []string{proto.ModControl}); err != nil {
		return result, err
	}
	posted := time.Now()
	result.Posted = true

	var last fieldReading
	if verify {
		result.Field = &proto.PasteField{Role: accessibilityRole(identity), Before: before.state()}
		result.Check, result.CheckMS, last = server.pollPaste(automation, watched, identity, before, posted, params.VerifyMS)
		if readableFieldSignal(last) {
			result.Field.After = last.state()
		}
	}
	unchanged := result.Check == proto.CheckUnchanged
	if len(params.Press) > 0 {
		if unchanged {
			result.PressSkipped = true
		} else {
			pressDelayMS := params.PressDelayMS
			if pressDelayMS == 0 {
				pressDelayMS = 120
			} else if pressDelayMS < 0 {
				pressDelayMS = 0
			}
			pressDelay := time.Duration(pressDelayMS) * time.Millisecond
			if server.waitPaste(pressDelay - time.Since(posted)) {
				for _, press := range params.Press {
					virtualKey, _ := windowsVirtualKey(press.Key)
					if err := sendKeyCombo(virtualKey, press.Modifiers); err != nil {
						return result, err
					}
				}
			}
		}
	}

	if !params.KeepClipboard && unchanged {
		sequence, _, _ := procGetClipboardSequence.Call()
		result.Kept = uint32(sequence) == snapshot.ourSequence
		snapshot.discard()
	} else if !params.KeepClipboard {
		restoreDelayMS := params.RestoreDelayMS
		if restoreDelayMS == 0 {
			restoreDelayMS = 500
		} else if restoreDelayMS < 0 {
			restoreDelayMS = 0
		}
		restoreDelay := time.Duration(restoreDelayMS) * time.Millisecond
		if result.Check == proto.CheckInserted {
			restoreDelay = 150*time.Millisecond - time.Since(posted)
		}
		server.waitPaste(restoreDelay)
		restored, taken, restoreErr := snapshot.restore()
		result.Restored = restored
		result.ClipboardTaken = taken
		if restoreErr != nil {
			fmt.Fprintln(os.Stderr, "clipboard restore:", restoreErr)
		}
	}
	completed = true
	result.MS = millisecondsSince(started)
	return result, nil
}

func (server *helperServer) pollPaste(automation, watched *comObject, identity elementIdentity, before fieldReading, posted time.Time, verifyMS int) (string, float64, fieldReading) {
	if !readableFieldSignal(before) {
		return proto.CheckUnknown, 0, fieldReading{}
	}
	deadline := posted.Add(time.Duration(verifyMS) * time.Millisecond)
	last := before
	for {
		select {
		case <-server.shutdown:
			return proto.CheckUnknown, 0, last
		case <-time.After(pastePollInterval):
		}
		last = readField(watched)
		if changedFieldSignal(before, last) {
			return proto.CheckInserted, millisecondsSince(posted), last
		}
		if time.Now().After(deadline) {
			break
		}
	}
	elapsed := millisecondsSince(posted)
	focused, err := focusedElement(automation)
	if err != nil {
		return proto.CheckUnknown, 0, last
	}
	defer focused.release()
	same, err := sameElement(automation, watched, focused)
	if err != nil || !same || last.problems || !comparableFieldSignal(before, last) {
		return proto.CheckUnknown, 0, last
	}
	valueComparable := before.value.available && last.value.available
	if !valueComparable && identity.controlType != controlTypeEdit && identity.controlType != controlTypeDocument {
		return proto.CheckUnknown, 0, last
	}
	return proto.CheckUnchanged, elapsed, last
}

func (server *helperServer) waitPaste(delay time.Duration) bool {
	if delay <= 0 {
		return true
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-server.shutdown:
		return false
	}
}

func millisecondsSince(started time.Time) float64 {
	return math.Round(float64(time.Since(started))/float64(time.Microsecond)/100) / 10
}
