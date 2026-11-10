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

// TestKeyRecorderTapPaused: while the field records, the shortcuts are
// off and the helper, as the Swift one does with no bindings, removes its
// tap. That is not a missing Accessibility: the page keeps the tap as
// installed, so no "Without Accessibility" notice comes up beside the
// field (11-09: it did, and its Open Permissions went with it as soon as
// the field lost the focus); once the field lets go, the tap is back.
func TestKeyRecorderTapPaused(t *testing.T) {
	var fake *fakeSupervisor
	a, tt := trHelperTester(t, func(fs *fakeSupervisor) {
		fake = fs
		fs.handlers[proto.MethodHotkeyConfigure] = func(raw json.RawMessage) (any, error) {
			var p proto.HotkeyConfigureParams
			json.Unmarshal(raw, &p)
			return proto.HotkeyStatus{Installed: len(p.Bindings) > 0 || p.Cancel != nil, Accessibility: true}, nil
		}
		fs.handlers[proto.MethodHotkeyRecord] = func(json.RawMessage) (any, error) {
			return nil, proto.Errorf(proto.CodeTimeout, "Recording timed out")
		}
		fs.handlers[proto.MethodHotkeyCancelRecord] = func(json.RawMessage) (any, error) { return proto.Empty{}, nil }
	})
	notice := "Without Accessibility only plain shortcuts work (⌃⌥⇧⌘ with a key); fn, a single modifier, double taps and long presses need it."
	lastBindings := func() int {
		fake.mu.Lock()
		defer fake.mu.Unlock()
		for i := len(fake.calls) - 1; i >= 0; i-- {
			if fake.calls[i].Method == proto.MethodHotkeyConfigure {
				var p proto.HotkeyConfigureParams
				json.Unmarshal(fake.calls[i].Params, &p)
				return len(p.Bindings)
			}
		}
		return -1
	}
	if err := tt.ClickControl("Keys of shortcut 1"); err != nil {
		t.Fatal(err)
	}
	click(t, tt, "Custom…")
	for i := 0; i < 20; i++ {
		frames(tt)
	}
	if !a.keysListening || lastBindings() != 0 {
		t.Fatalf("the field records: listening %v, last hotkey.configure with %d bindings", a.keysListening, lastBindings())
	}
	if !a.hotkeys.Installed || has(tt, notice) {
		t.Errorf("while the field records: installed %v, notice %v", a.hotkeys.Installed, has(tt, notice))
	}
	click(t, tt, "Add a shortcut") // the field lets go
	for i := 0; i < 20; i++ {
		frames(tt)
	}
	if a.keysListening || lastBindings() == 0 || !a.hotkeys.Installed || has(tt, notice) {
		t.Errorf("after the field: listening %v, %d bindings, installed %v, notice %v", a.keysListening, lastBindings(), a.hotkeys.Installed, has(tt, notice))
	}
}
