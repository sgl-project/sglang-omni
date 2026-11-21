package main

import (
	"slices"
	"testing"
)

// TestSpeechModelsPageOrder: the Speech models page lists what runs on
// this Mac, System dictation among it, then the remote services: #3594
// and #3598 each added theirs below the local models, and the merge put
// System dictation first.
func TestSpeechModelsPageOrder(t *testing.T) {
	a, fs, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	a.dictation.SetHelper(fs) // as helperChanged does for a helper with speech.*
	a.navigate(modelsPath)
	frames(tt)
	texts := tt.Texts()
	local, dictation, online := slices.Index(texts, platformText("On this Mac")), slices.Index(texts, "System dictation"), slices.Index(texts, "Online services")
	if local < 0 || dictation < 0 || online < 0 || !(local < dictation && dictation < online) {
		t.Fatalf("On this Mac %d, System dictation %d, Online services %d: %q", local, dictation, online, texts)
	}
}
