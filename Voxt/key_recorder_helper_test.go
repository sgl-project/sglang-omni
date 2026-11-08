package main

import (
	"encoding/json"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func trHelperTester(t *testing.T, setup func(*fakeSupervisor)) (*app, *ui.Tester) {
	t.Helper()
	a, fs, tt := newTestApp(t)
	if setup != nil {
		setup(fs)
	}
	a.hotkeys = proto.HotkeyStatus{Installed: true, Accessibility: true}
	tt.SetSize(windowWidth, 2400)
	a.navigate("/transcribe")
	frames(tt)
	return a, tt
}

// TestKeyRecorderHelper: hotkey.record returns a chord; it is saved and shown.
func TestKeyRecorderHelper(t *testing.T) {
	a, tt := trHelperTester(t, func(fs *fakeSupervisor) {
		fs.handlers[proto.MethodHotkeyRecord] = func(json.RawMessage) (any, error) {
			return proto.HotkeyRecordResult{Keys: "RightOption+Space"}, nil
		}
		fs.handlers[proto.MethodHotkeyCancelRecord] = func(json.RawMessage) (any, error) {
			return proto.Empty{}, nil
		}
	})
	saved := func() string { return a.lib.settings.Get().Features.Transcribe.Hotkeys[0].Keys }
	if err := tt.ClickControl("Keys of shortcut 1"); err != nil {
		t.Fatal(err)
	}
	click(t, tt, "Custom…")
	for i := 0; i < 30 && saved() != "RightOption+Space"; i++ {
		frames(tt)
	}
	if saved() != "RightOption+Space" || !hasExact(tt, "Right ⌥ Space") {
		t.Fatalf("helper record: saved %q, texts %q", saved(), tt.Texts())
	}
}

// TestKeyRecorderNoAccessibility: permission_denied shows the notice.
func TestKeyRecorderNoAccessibility(t *testing.T) {
	a, tt := trHelperTester(t, func(fs *fakeSupervisor) {
		fs.handlers[proto.MethodHotkeyRecord] = func(json.RawMessage) (any, error) {
			return nil, proto.Errorf(proto.CodePermissionDenied, "Accessibility is required")
		}
	})
	a.hotkeys.Accessibility = true
	if err := tt.ClickControl("Keys of shortcut 1"); err != nil {
		t.Fatal(err)
	}
	click(t, tt, "Custom…")
	for i := 0; i < 20; i++ {
		frames(tt)
	}
	if !has(tt, "Recording a shortcut needs Accessibility permission.") || !has(tt, "Open Permissions") {
		t.Errorf("expected accessibility notice, got %q", tt.Texts())
	}
}

// TestKeyRecorderTimeout: timeout leaves the saved keys unchanged.
func TestKeyRecorderTimeout(t *testing.T) {
	a, tt := trHelperTester(t, func(fs *fakeSupervisor) {
		fs.handlers[proto.MethodHotkeyRecord] = func(json.RawMessage) (any, error) {
			return nil, proto.Errorf(proto.CodeTimeout, "Recording timed out")
		}
	})
	h := settings.Hotkey{Keys: "Fn+F5", Trigger: settings.TriggerTap}
	a.setHotkeys(hotkey.Transcribe, false, func(hs []settings.Hotkey) []settings.Hotkey {
		if len(hs) > 0 {
			hs[0] = h
		}
		return hs
	})
	saved := func() string { return a.lib.settings.Get().Features.Transcribe.Hotkeys[0].Keys }
	if err := tt.ClickControl("Keys of shortcut 1"); err != nil {
		t.Fatal(err)
	}
	click(t, tt, "Custom…")
	for i := 0; i < 25; i++ {
		frames(tt)
	}
	if saved() != "Fn+F5" {
		t.Fatalf("timeout changed keys: %q", saved())
	}
}

// TestKeyRecorderCancelOnBlur: leaving the field cancels hotkey.record while
// a helper capture is still in flight (helperCall is synchronous in view tests,
// so we mark recorderActive as the UI would during a live capture).
func TestKeyRecorderCancelOnBlur(t *testing.T) {
	cancelled := false
	a, tt := trHelperTester(t, func(fs *fakeSupervisor) {
		fs.handlers[proto.MethodHotkeyCancelRecord] = func(json.RawMessage) (any, error) {
			cancelled = true
			return proto.Empty{}, nil
		}
	})
	if err := tt.ClickControl("Keys of shortcut 1"); err != nil {
		t.Fatal(err)
	}
	click(t, tt, "Custom…")
	if err := tt.ClickControl("Custom keys of shortcut 1"); err != nil {
		t.Fatal(err)
	}
	a.recorderActive = true
	click(t, tt, "Add a shortcut")
	for i := 0; i < 15; i++ {
		frames(tt)
	}
	if !cancelled {
		t.Error("hotkey.cancelRecord was not sent")
	}
}
