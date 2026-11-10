package main

import (
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
