package main

import (
	"slices"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// TestPrivacySendWindow: sending a screenshot of the window is off by
// default; turning it on saves it and asks for Screen Recording, and
// while Voxt is not allowed a toast says where to allow it; turning it
// off asks nothing.
func TestPrivacySendWindow(t *testing.T) {
	a, fs, tt := newTestApp(t)
	if a.settings.Get().Privacy.SendWindow {
		t.Fatal("on by default")
	}
	openPrivacy(t, a, tt)
	if !has(tt, "sends it to the language model with the text") {
		t.Fatalf("no explanation: %q", firstTexts(tt))
	}
	click(t, tt, "Send the window to the language model")
	if !a.settings.Get().Privacy.SendWindow || !slices.Equal(fs.asked, []string{proto.PermissionScreenRecording}) {
		t.Fatalf("on: setting %v, asked %v", a.settings.Get().Privacy.SendWindow, fs.asked)
	}
	if where := "Screen & System Audio Recording"; !strings.Contains(strings.Join(a.toasts, "\n"), where) && !has(tt, where) {
		t.Errorf("no toast: %q, %q", a.toasts, firstTexts(tt))
	}
	click(t, tt, "Send the window to the language model")
	if a.settings.Get().Privacy.SendWindow || len(fs.asked) != 1 {
		t.Errorf("off: setting %v, asked %v", a.settings.Get().Privacy.SendWindow, fs.asked)
	}
}
