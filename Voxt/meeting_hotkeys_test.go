package main

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
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

// TestMeetingIgnoresQueuedFinish: a Return the helper still reports as
// Finish from a dictation's old configuration (its new one has not landed
// yet) does not stop the meeting that started right after, and a meeting's
// plan never binds Return.
func TestMeetingIgnoresQueuedFinish(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.prefs.Features.Transcribe.ReturnFinishes = true
	a.meetSnap = meeting.Snapshot{ID: "m1", State: meeting.StateRecording}
	a.hotkeyFired(proto.HotkeyEventParams{
		Binding: hotkey.FinishID, Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap,
	})
	if a.meetSnap.State != meeting.StateRecording {
		t.Fatal("a queued Return-finish event stopped the meeting")
	}
	if a.sess.State.Live() {
		t.Fatal("a queued Return-finish event started a session during a meeting")
	}
	cfg := a.hotkeyPlan().Configure(true)
	if slices.ContainsFunc(cfg.Bindings, func(b proto.HotkeyBinding) bool { return b.ID == hotkey.FinishID }) {
		t.Fatal("a meeting's hotkey plan binds Return")
	}
}
