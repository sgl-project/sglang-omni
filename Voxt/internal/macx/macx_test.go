package macx

import (
	"os"
	"runtime"
	"testing"
	"time"
)

// TestActivate: off macOS nothing happens; on a Mac (VOXTNEW_MAC=1) the
// app in front is reactivated at once.
func TestActivate(t *testing.T) {
	run := func(fn func()) { fn() }
	if runtime.GOOS != "darwin" {
		if ok, _ := Activate(1, time.Millisecond, run); ok || Frontmost() != (App{}) {
			t.Error("activated off macOS")
		}
		return
	}
	if os.Getenv("VOXTNEW_MAC") == "" {
		t.Skip("set VOXTNEW_MAC=1 on a Mac")
	}
	front := Frontmost()
	t.Logf("front: %+v", front)
	if front.PID == 0 {
		t.Skip("no app in front (no window server?)")
	}
	if ok, d := Activate(front.PID, 500*time.Millisecond, run); !ok {
		t.Errorf("could not activate the app in front (%s)", d)
	}
	if ok, _ := Activate(0, time.Millisecond, run); ok {
		t.Error("activated pid 0")
	}
}

// TestSystemProxies: off macOS nothing; on a Mac (VOXTNEW_MAC=1) the
// settings scutil --proxy shows (mac-air: 127.0.0.1:7897 for both).
func TestSystemProxies(t *testing.T) {
	p := SystemProxies()
	if runtime.GOOS != "darwin" {
		if p.HTTP != "" || p.HTTPS != "" || p.Exceptions != nil || p.ExcludeSimple {
			t.Errorf("proxies off macOS: %+v", p)
		}
		return
	}
	if os.Getenv("VOXTNEW_MAC") == "" {
		t.Skip("set VOXTNEW_MAC=1 on a Mac")
	}
	t.Logf("system proxies: %+v", p)
}
