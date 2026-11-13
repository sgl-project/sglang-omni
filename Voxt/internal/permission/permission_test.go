package permission

import (
	"os"
	"runtime"
	"testing"
)

func TestFake(t *testing.T) {
	f := &Fake{Mic: NotDetermined, Grant: true}
	var got bool
	f.RequestMicrophone(func(g bool) { got = g })
	if !got || f.Microphone() != Granted {
		t.Errorf("request: %v %v", got, f.Microphone())
	}
	f.RequestSpeechRecognition(func(g bool) { got = g })
	if !got || f.SpeechRecognition() != Granted {
		t.Errorf("speech request: %v %v", got, f.SpeechRecognition())
	}
	f.PromptAccessibility()
	if f.Prompted != 1 || f.AccessibilityTrusted() {
		t.Errorf("accessibility %+v", f)
	}
}

// TestNative reads the real permissions: on a Mac with VOXTNEW_MAC=1 it
// logs them; Windows has no macOS-style gate; elsewhere they are unknown.
func TestNative(t *testing.T) {
	n := Native()
	if runtime.GOOS == "windows" {
		if n.Microphone() != Granted || !n.AccessibilityTrusted() || n.SpeechRecognition() != Granted {
			t.Error("a Windows permission is not granted")
		}
		return
	}
	if runtime.GOOS != "darwin" {
		if n.Microphone() != Unknown || n.AccessibilityTrusted() || n.SpeechRecognition() != Unknown {
			t.Error("a permission off macOS")
		}
		return
	}
	if os.Getenv("VOXTNEW_MAC") == "" {
		t.Skip("set VOXTNEW_MAC=1 on a Mac")
	}
	t.Logf("microphone %d, accessibility %v, speech recognition %d", n.Microphone(), n.AccessibilityTrusted(), n.SpeechRecognition())
}
