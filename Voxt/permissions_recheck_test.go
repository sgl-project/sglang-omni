package main

import (
	"bytes"
	"encoding/json"
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
)

// TestRecheckFirstReadShowsOwn: before anything was read, a recheck shows
// this process's read until the helper answers, not empty rows; the
// helper's answer then replaces it.
func TestRecheckFirstReadShowsOwn(t *testing.T) {
	a, _, tt := newTestApp(t)
	frames(tt)
	a.perms.mic, a.perms.speech, a.perms.access, a.perms.checked = permission.Unknown, permission.Unknown, false, false
	var queued []func()
	a.postQueue = func(fn func()) { queued = append(queued, fn) }
	a.recheckPermissions()
	if !a.perms.checked || a.perms.mic != permission.Granted || len(queued) == 0 {
		t.Errorf("before the helper answers: checked %v, microphone %v, %d posts", a.perms.checked, a.perms.mic, len(queued))
	}
	for _, fn := range queued {
		fn()
	}
	a.postQueue = nil
	if a.perms.mic != permission.NotDetermined {
		t.Errorf("after the helper answered: microphone %v", a.perms.mic)
	}
}

// TestAudioRefreshStaysEnabled: reading the microphones (each time the
// section opens) does not grey out Refresh for a frame.
func TestAudioRefreshStaysEnabled(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/audio")
	frames(tt)
	idle := slices.Clone(tt.Image().Pix)
	var queued []func()
	a.postQueue = func(fn func()) { queued = append(queued, fn) }
	a.loadDevices()
	frames(tt)
	if !a.audio.loading || !bytes.Equal(idle, tt.Image().Pix) {
		t.Errorf("the section changed while reading the microphones (loading %v)", a.audio.loading)
	}
	for _, fn := range queued {
		fn()
	}
	a.postQueue = nil
}

// TestSelfCheckReadsPermissionsNow: Run check reports the permissions as
// the helper reads them now, not as they were last read.
func TestSelfCheckReadsPermissionsNow(t *testing.T) {
	a, fs, tt := newTestApp(t)
	frames(tt)
	if a.perms.mic != permission.NotDetermined {
		t.Fatalf("microphone %v before", a.perms.mic)
	}
	fs.mu.Lock()
	fs.handlers[proto.MethodPermissionsStatus] = func(json.RawMessage) (any, error) {
		return proto.PermissionsResult{Microphone: proto.PermGranted, SpeechRecognition: proto.PermGranted}, nil
	}
	fs.mu.Unlock()
	var queued []func() // the helper answers later, as it does in the app
	a.postQueue = func(fn func()) { queued = append(queued, fn) }
	a.runSelfCheck()
	for len(queued) > 0 {
		fn := queued[0]
		queued = queued[1:]
		fn()
	}
	a.postQueue = nil
	for _, check := range a.checks {
		if check.id == "microphone" && check.status != "pass" {
			t.Errorf("microphone check %q: %s", check.status, check.detail)
		}
	}
	if len(a.checks) == 0 || a.selfChecking {
		t.Errorf("checks %d, still checking %v", len(a.checks), a.selfChecking)
	}
}
