package main

import (
	"context"
	"testing"
)

// The meeting manager is made before the helper starts, so whether the
// helper can capture system audio must be asked at each start, not then.
func TestSystemCaptureOKFollowsHelper(t *testing.T) {
	a, fs, _ := newTestApp(t)
	if !a.systemCaptureOK() {
		t.Fatal("helper with capture.system: not OK")
	}
	fs.Stop()
	if a.systemCaptureOK() {
		t.Fatal("stopped helper: OK")
	}
	if err := fs.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !a.systemCaptureOK() {
		t.Fatal("restarted helper: not OK")
	}
}
