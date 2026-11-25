package main

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/triggerapi"
)

// fireCommand sends action as the socket does, from another goroutine,
// and runs the app's queue until the app has answered. It waits on the
// answer itself: an idle app posts nothing more that would wake a pump.
func fireCommand(a *sessionApp, action string) error {
	a.t.Helper()
	done := make(chan error, 1)
	go func() { done <- a.fireForCommand(context.Background(), action) }()
	deadline := time.After(10 * time.Second)
	for {
		select {
		case err := <-done:
			return err
		case run := <-a.queue:
			run()
		case <-deadline:
			a.t.Fatalf("timed out waiting for the answer to %s", action)
		}
	}
}

// TestTriggerTogglesLikeTheShortcut: `Voxt toggle transcribe` twice records,
// then pastes the text into the app in front, as two taps of the shortcut do.
func TestTriggerTogglesLikeTheShortcut(t *testing.T) {
	set, keys, _ := fileSetup(t, "hello aura link", &fakeChat{text: "Hello, VoxtNew."})
	a := newSessionApp(t, "", set, keys)
	a.front = helpertest.Apps["com.tinyspeck.slackmacgap"]

	if err := fireCommand(a, triggerapi.ActionTranscribe); err != nil {
		t.Fatal(err)
	}
	a.pump("the recording", func() bool { return a.sess.State == session.Recording })
	a.drain(400 * time.Millisecond)
	if err := fireCommand(a, triggerapi.ActionTranscribe); err != nil {
		t.Fatal(err)
	}
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Done || a.sess.Result != "Hello, VoxtNew." || !a.sess.Inserted || a.sess.Target.Name != "Slack" {
		t.Fatalf("end %+v", a.sess)
	}
	a.pump("the paste", func() bool { return len(a.pasted()) == 1 })
	if pasted := a.pasted()[0]; pasted.Text != "Hello, VoxtNew." || pasted.TargetPID != a.front.PID {
		t.Errorf("pasted %+v", pasted)
	}
}

// TestTriggerCancelsAndRefusesHiddenFeatures: `Voxt cancel` cancels a
// recording; a hidden feature's toggle starts nothing and says why.
func TestTriggerCancelsAndRefusesHiddenFeatures(t *testing.T) {
	set, keys, _ := fileSetup(t, "hello aura link", &fakeChat{text: "Hello, VoxtNew."})
	a := newSessionApp(t, "", func(s *settings.Settings) {
		set(s)
		s.Features.Hidden = []string{settings.HideTranslate, settings.HideNotes}
	}, keys)
	a.front = helpertest.Apps["com.tinyspeck.slackmacgap"]

	for action, reason := range map[string]string{
		triggerapi.ActionTranslate: "Translate is hidden",
		triggerapi.ActionNote:      "Notes are hidden",
	} {
		if err := fireCommand(a, action); err == nil || !strings.Contains(err.Error(), reason) {
			t.Errorf("%s: err %v, want %q", action, err, reason)
		}
	}
	a.drain(100 * time.Millisecond)
	if a.sess.State.Live() {
		t.Fatalf("a hidden feature started %+v", a.sess)
	}

	if err := fireCommand(a, triggerapi.ActionTranscribe); err != nil {
		t.Fatal(err)
	}
	a.pump("the recording", func() bool { return a.sess.State == session.Recording })
	if err := fireCommand(a, triggerapi.ActionCancel); err != nil {
		t.Fatal(err)
	}
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Cancelled || len(a.pasted()) != 0 {
		t.Fatalf("end %+v, pasted %+v", a.sess, a.pasted())
	}
}
