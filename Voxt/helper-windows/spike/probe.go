//go:build windows

package main

import (
	"fmt"
	"strings"
	"time"
)

const (
	probeText      = "UIA-paste-probe"
	pollInterval   = 25 * time.Millisecond
	verifyTimeout  = time.Second
	clipboardDelay = 150 * time.Millisecond
)

func probeTarget(automation *comObject, targetName string, windowHandle uintptr, processID uint32) error {
	fmt.Printf("\nTARGET %s: window=0x%x pid=%d\n", targetName, windowHandle, processID)
	if err := focusEditor(automation, windowHandle); err != nil {
		return err
	}
	time.Sleep(150 * time.Millisecond)
	watched, err := focusedElement(automation)
	if err != nil {
		return err
	}
	defer watched.release()
	identity := identifyElement(watched)
	fmt.Printf("focused: control=%s framework=%q name=%q pid=%d IsPassword=%t ValueIsReadOnly=%s\n",
		controlTypeName(identity.controlType), identity.framework, identity.name,
		identity.processID, identity.isPassword, identity.readOnly)
	if identity.processID != int32(processID) {
		return fmt.Errorf("focus belongs to pid %d, not target pid %d; no input sent", identity.processID, processID)
	}
	if identity.isPassword {
		fmt.Println("VERDICT unknown: IsPassword=true; paste deliberately skipped")
		return nil
	}
	if err := typeText("seed "); err != nil {
		return err
	}
	time.Sleep(150 * time.Millisecond)
	if err := sendKeyWithControl(virtualKeyEnd); err != nil {
		return err
	}
	time.Sleep(50 * time.Millisecond)
	before := readField(watched)
	fmt.Printf("before: %s\n", before)

	snapshot, err := snapshotAndSetClipboard(probeText)
	if err != nil {
		return err
	}
	posted := time.Now()
	restoreDone := false
	restoreClipboard := func() {
		if restoreDone {
			return
		}
		restoreDone = true
		if remaining := clipboardDelay - time.Since(posted); remaining > 0 {
			time.Sleep(remaining)
		}
		restored, taken, err := snapshot.restore()
		fmt.Printf("clipboard: restored=%t taken=%t", restored, taken)
		if err != nil {
			fmt.Printf(" error=%v", err)
		}
		fmt.Println()
	}
	defer restoreClipboard()
	if err := sendKeyWithControl(virtualKeyV); err != nil {
		return err
	}
	posted = time.Now()
	fmt.Printf("posted: Ctrl+V textLength=%d clipboardSequence=%d\n", len(probeText), snapshot.ourSequence)

	comparison, checkDuration := pollPaste(automation, watched, before, posted)
	verdict := comparison
	if comparison == "unchanged" && !before.value.available && identity.controlType != controlTypeEdit && identity.controlType != controlTypeDocument {
		verdict = "unknown"
	}
	fmt.Printf("VERDICT %s: comparison=%s checkMs=%.1f signals=%s\n", verdict, comparison,
		float64(checkDuration)/float64(time.Millisecond), comparisonSummary(before))
	restoreClipboard()
	if verdict == "inserted" {
		controlBefore := readField(watched)
		controlPosted := time.Now()
		fmt.Println("negative control: polling the same field with no further input")
		controlVerdict, controlDuration := pollPaste(automation, watched, controlBefore, controlPosted)
		fmt.Printf("CONTROL %s: checkMs=%.1f\n", controlVerdict,
			float64(controlDuration)/float64(time.Millisecond))
	}
	return nil
}

func pollPaste(automation *comObject, watched *comObject, before fieldReading, posted time.Time) (string, time.Duration) {
	deadline := posted.Add(verifyTimeout)
	last := before
	for {
		time.Sleep(pollInterval)
		last = readField(watched)
		elapsed := time.Since(posted)
		changes := changedSignals(before, last)
		fmt.Printf("sample %7.1f ms: %s changes=%s\n",
			float64(elapsed)/float64(time.Millisecond), last, formatChanges(changes))
		if len(changes) > 0 {
			return "inserted", elapsed
		}
		if time.Now().After(deadline) {
			break
		}
	}
	elapsed := time.Since(posted)
	focused, err := focusedElement(automation)
	if err != nil {
		fmt.Printf("focus check: %v\n", err)
		return "unknown", elapsed
	}
	defer focused.release()
	same, err := sameElement(automation, watched, focused)
	if err != nil || !same {
		fmt.Printf("focus check: same=%t error=%v\n", same, err)
		return "unknown", elapsed
	}
	if !hasComparableSignals(before, last) {
		return "unknown", elapsed
	}
	return "unchanged", elapsed
}

func changedSignals(before fieldReading, after fieldReading) []string {
	var changes []string
	if before.value.available && after.value.available {
		if before.value.length != after.value.length {
			changes = append(changes, fmt.Sprintf("value-length(%+d)", after.value.length-before.value.length))
		}
		if before.value.digest != after.value.digest {
			changes = append(changes, "value-digest")
		}
	}
	if before.text.available && after.text.available {
		if before.text.length != after.text.length {
			changes = append(changes, fmt.Sprintf("text-length(%+d)", after.text.length-before.text.length))
		}
		if before.text.digest != after.text.digest {
			changes = append(changes, "text-digest")
		}
	}
	if before.selection.available && after.selection.available && before.selection != after.selection {
		changes = append(changes, fmt.Sprintf("selection(%d+%d->%d+%d)",
			before.selection.location, before.selection.length,
			after.selection.location, after.selection.length))
	}
	return changes
}

func hasComparableSignals(before fieldReading, after fieldReading) bool {
	return before.value.available && after.value.available ||
		before.text.available && after.text.available ||
		before.selection.available && after.selection.available
}

func comparisonSummary(reading fieldReading) string {
	var signals []string
	if reading.value.available {
		signals = append(signals, "value-length,value-digest")
	}
	if reading.text.available {
		signals = append(signals, "text-length,text-digest")
	}
	if reading.selection.available {
		signals = append(signals, "selection")
	}
	if len(signals) == 0 {
		return "none"
	}
	return strings.Join(signals, ",")
}

func formatChanges(changes []string) string {
	if len(changes) == 0 {
		return "none"
	}
	return strings.Join(changes, ",")
}
