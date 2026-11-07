package main

import (
	"slices"
	"testing"
)

// TestReachableButtons: VoiceOver reaches the buttons drawn inside a
// card or a row: a remote service card's Test, a failed history entry's
// Retry. Both sat inside a button, a leaf to assistive technology, and
// could not be reached; the card and the row are groups now.
func TestReachableButtons(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, tt, _ := retryApp(t, "Recovered dictation.")
	a.navigate(remotePath)
	frames(tt)
	click(t, tt, "DeepSeek")
	frames(tt)
	a.provDlg.key = "sk-deep-0123456789"
	click(t, tt, "Save")
	frames(tt)
	if labels := tt.AccessLabels(); !slices.Contains(labels, "Test DeepSeek") || !slices.Contains(labels, "DeepSeek") {
		t.Errorf("the card's Test cannot be reached: %q", labels)
	}

	seedRetryEntry(t, a)
	a.navigate("/history")
	frames(tt)
	if labels := tt.AccessLabels(); !slices.Contains(labels, "Retry") {
		t.Errorf("the row's Retry cannot be reached: %q", labels)
	}
}
