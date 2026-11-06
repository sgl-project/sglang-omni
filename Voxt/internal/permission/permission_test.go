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
	f.PromptAccessibility()
	if f.Prompted != 1 || f.AccessibilityTrusted() {
		t.Errorf("accessibility %+v", f)
	}
}

// TestNative reads the real permissions: on a Mac with VOXTNEW_MAC=1 it
// logs them (they depend on the machine); elsewhere they are unknown.
func TestNative(t *testing.T) {
	n := Native()
	if runtime.GOOS != "darwin" {
		if n.Microphone() != Unknown || n.AccessibilityTrusted() {
			t.Error("a permission off macOS")
		}
		return
	}
	if os.Getenv("VOXTNEW_MAC") == "" {
		t.Skip("set VOXTNEW_MAC=1 on a Mac")
	}
	t.Logf("microphone %d, accessibility %v", n.Microphone(), n.AccessibilityTrusted())
}
