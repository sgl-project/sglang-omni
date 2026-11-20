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
	tt.Scroll(float32(windowWidth)*0.6, 900, 0, -10000)
	frames(tt)
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

// The history setting says what turning it off removes from the menus.
func TestPrivacyHistoryExplainsReopen(t *testing.T) {
	a, _, tt := newTestApp(t)
	openPrivacy(t, a, tt)
	if !has(tt, "the menu bar and Dock menus do not list cards to reopen") {
		t.Fatalf("no Reopen card privacy explanation: %q", firstTexts(tt))
	}
}

// The list names the content and destination of each local or API request,
// including optional images and Translate's fallback (#4159).
func TestPrivacyNamesBothDataPaths(t *testing.T) {
	a, _, tt := newTestApp(t)
	openPrivacy(t, a, tt)
	for _, want := range []string{
		"Your recordings are recognised on this Mac",
		"its recordings go to that service",
		"text Voxt recognised",
		"text you selected or typed",
		"meeting transcripts",
		"your speech context",
		"your rules and enabled dictionary content",
		"a screenshot of the window or an image you copied",
		"A request to a language-model server on this Mac",
		"stays on this Mac",
		"A request to an external API goes to the provider handling it",
		"If Translate uses your fallback model, that provider gets the same request too",
	} {
		if !has(tt, want) {
			t.Errorf("the list does not say %q: %q", want, firstTexts(tt))
		}
	}
}

// TestPrivacySendClipboardImage: sending the copied image with Rewrite is
// off by default; the switch saves it, asks for no permission, and its
// note says what goes and what does not.
func TestPrivacySendClipboardImage(t *testing.T) {
	a, fs, tt := newTestApp(t)
	if a.settings.Get().Privacy.SendClipboardImage {
		t.Fatal("on by default")
	}
	openPrivacy(t, a, tt)
	if !has(tt, "Copied text, and copied files that are not images, are not sent") {
		t.Fatalf("no explanation: %q", firstTexts(tt))
	}
	click(t, tt, "Send the copied image with Rewrite")
	if !a.settings.Get().Privacy.SendClipboardImage || len(fs.asked) != 0 {
		t.Fatalf("on: setting %v, asked %v", a.settings.Get().Privacy.SendClipboardImage, fs.asked)
	}
	click(t, tt, "Send the copied image with Rewrite")
	if a.settings.Get().Privacy.SendClipboardImage {
		t.Error("off: still on")
	}
	if a.settings.Get().Privacy.SendClipboardImageWithSelection {
		t.Fatal("text and image together is on by default")
	}
	click(t, tt, "Send both with Rewrite")
	if !a.settings.Get().Privacy.SendClipboardImageWithSelection || len(fs.asked) != 0 {
		t.Fatalf("text and image together: setting %v, asked %v", a.settings.Get().Privacy.SendClipboardImageWithSelection, fs.asked)
	}
	click(t, tt, "Send both with Rewrite")
	if a.settings.Get().Privacy.SendClipboardImageWithSelection {
		t.Error("text and image together: still on")
	}
}
