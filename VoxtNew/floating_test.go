package main

import (
	"runtime"
	"testing"
)

// TestFloatingWindowOptions: floating windows never activate VoxtNew
// (mygo patch 0003), and start hidden above other windows.
func TestFloatingWindowOptions(t *testing.T) {
	o := floatingWindowOptions("pill", 220, 48, nil)
	if !o.NonActivating || !o.AlwaysOnTop || !o.Hidden || !o.SkipTaskbar || !o.Frameless || o.Width != 220 || o.Height != 48 {
		t.Errorf("options %+v", o)
	}
	// Off macOS nothing can be activated.
	if runtime.GOOS != "darwin" {
		if ok, _ := activateTarget(1); ok {
			t.Error("activated off macOS")
		}
	}
}
