package main

import (
	"context"
	"slices"
	"sync"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestMeetingBlocksDictation(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.meetSnap = meeting.Snapshot{ID: "m1", State: meeting.StateRecording}
	a.hotkeyFired(proto.HotkeyEventParams{
		Binding: hotkey.ID(hotkey.Transcribe, 0), Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap,
	})
	if a.sess.State.Live() {
		t.Fatal("transcribe shortcut started a session during a meeting")
	}
}

// silentAudio records silence until it is stopped: a microphone or the
// system audio for a meeting.
type silentAudio struct {
	mu    sync.Mutex
	stops map[string]chan struct{}
}

func (s *silentAudio) Record(ctx context.Context, id string, _ proto.CaptureParams, fn func(session.AudioEvent)) (proto.CaptureResult, error) {
	stop := make(chan struct{})
	s.mu.Lock()
	if s.stops == nil {
		s.stops = map[string]chan struct{}{}
	}
	s.stops[id] = stop
	s.mu.Unlock()
	fn(session.AudioEvent{Started: &proto.CaptureStartedParams{Job: id, SampleRate: asr.SampleRate}})
	fn(session.AudioEvent{PCM: &proto.PCMParams{Job: id, Data: make([]byte, asr.BytesPerSecond/10)}})
	select {
	case <-ctx.Done():
		return proto.CaptureResult{}, ctx.Err()
	case <-stop:
		return proto.CaptureResult{Reason: proto.CaptureStopped}, nil
	}
}

func (s *silentAudio) Stop(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ch, ok := s.stops[id]; ok {
		close(ch)
		delete(s.stops, id)
	}
	return nil
}

// newMeetingShortcutApp is a test app with a meeting manager on silent
// audio and one meeting shortcut, ⌃⌥M: keys Carbon can bind too.
func newMeetingShortcutApp(t *testing.T) *app {
	t.Helper()
	a, _, _ := newTestApp(t)
	a.meetMgr = meeting.NewManager(meeting.Config{
		Mic:    &silentAudio{},
		System: &silentAudio{},
		Store:  a.lib.store.Meetings(),
		Dir:    t.TempDir(),
	})
	a.prefs.Features.Meeting.Mode = "recording"
	a.prefs.Features.Meeting.Hotkeys = []settings.Hotkey{{Keys: "Control+Option+M", Trigger: settings.TriggerTap}}
	return a
}

// pressMeetingShortcut fires the meeting shortcut, then takes the meeting
// manager's snapshot, as its subscription would post it.
func pressMeetingShortcut(a *app) {
	a.hotkeyFired(proto.HotkeyEventParams{Binding: hotkey.ID(hotkey.Meeting, 0), Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap})
	a.meetSnap = a.meetMgr.Current()
}

// planBinds says whether p binds id through the helper's event tap and
// through Carbon (the fallback without Accessibility).
func planBinds(p hotkey.Plan, id string) (tap, carbon bool) {
	tap = slices.ContainsFunc(p.Bindings, func(b proto.HotkeyBinding) bool { return b.ID == id })
	carbon = slices.ContainsFunc(p.Carbon, func(c hotkey.Carbon) bool { return c.ID == id })
	return tap, carbon
}

// #4641: a meeting shortcut is bound, through the helper's event tap and,
// for keys Carbon knows, without Accessibility too. A press starts a
// meeting, not a dictation, and the next press ends it, as the menu bar's
// Start meeting and End meeting do.
func TestMeetingShortcutStartsAndEndsMeeting(t *testing.T) {
	a := newMeetingShortcutApp(t)
	if tap, carbon := planBinds(a.hotkeyPlan(), "meeting/0"); !tap || !carbon {
		t.Fatalf("meeting/0 bound through the event tap %v, through Carbon %v; want both", tap, carbon)
	}
	pressMeetingShortcut(a)
	if a.meetSnap.State != meeting.StateRecording {
		t.Fatalf("meeting after a press: %q, want recording", a.meetSnap.State)
	}
	if a.sess.ID != "" {
		t.Fatalf("a press started a dictation too: %+v", a.sess)
	}
	pressMeetingShortcut(a)
	if s := a.meetSnap.State; s == meeting.StateRecording || s == meeting.StatePaused {
		t.Fatalf("meeting after a second press: %q, want it ended", s)
	}
}

// A meeting shortcut is off while Meetings is hidden, and a press while a
// dictation runs starts no meeting.
func TestMeetingShortcutOff(t *testing.T) {
	a := newMeetingShortcutApp(t)
	a.prefs.Features.Hidden = []string{settings.HideMeetings}
	if tap, carbon := planBinds(a.hotkeyPlan(), "meeting/0"); tap || carbon {
		t.Fatalf("hidden Meetings: meeting/0 bound through the event tap %v, through Carbon %v", tap, carbon)
	}
	a.prefs.Features.Hidden = nil
	a.sess = session.Snapshot{ID: "s1", State: session.Recording}
	pressMeetingShortcut(a)
	if s := a.meetSnap.State; s.Active() {
		t.Fatalf("a press during a dictation started a meeting: %q", s)
	}
}
