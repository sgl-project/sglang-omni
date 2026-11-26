package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// detectApp is the app over the fake helper process (the real supervisor,
// real pipes), where the helper's mic.watch names Zoom as recording in a
// meeting (its aomhost runs, unless opts say hosts=), with the speech model
// svc (nil: none installed). Its cues are silent.
func detectApp(t *testing.T, opts string, set func(*settings.Settings), svc local.Services) *sessionApp {
	t.Helper()
	if !strings.Contains(opts, "hosts=") {
		opts = "hosts=aomhost," + opts
	}
	s := newSessionApp(t, "recording=us.zoom.xos,"+opts, func(st *settings.Settings) {
		st.Features.Meeting.NoFinalPass, st.Features.Meeting.NoAutoSummary = true, true
		if set != nil {
			set(st)
		}
	}, svc)
	useTestSounds(t, s.app)
	s.playSound = func(sounds.Target) {}
	shortHold(s)
	return s
}

// shortHold makes s ask about an app after a moment of recording instead
// of meetdetect.Hold, on real timers (a test app has none of its own), and
// decides again about what the helper said before.
func shortHold(s *sessionApp) {
	s.detect.Hold = 20 * time.Millisecond
	s.laterHook = func(d time.Duration, fn func()) { time.AfterFunc(d, func() { s.post(fn) }) }
	s.post(s.syncMeetingDetect)
}

func hearsMeeting() *fakeLocal {
	return &fakeLocal{ASRP: fakeStreamer{final: "let us begin the weekly meeting now", partial: "let us begin"}}
}

func micLine(afterMS int, ids ...string) string {
	users := proto.MicUsers{Apps: []proto.App{}}
	for i, id := range ids {
		users.Apps = append(users.Apps, proto.App{PID: 7000 + i, BundleID: id})
	}
	return string(mustJSON(helpertest.Script{AfterMS: afterMS, Method: proto.MethodMicChanged, Params: mustJSON(users)}))
}

// The helper says Zoom records: the card shows, its button starts a real
// meeting over the helper's captures, and the card is gone while it runs.
func TestMeetingDetectEndToEnd(t *testing.T) {
	s := detectApp(t, "", nil, hearsMeeting())
	s.pump("the card", func() bool { return s.ask.view.Visible })
	if s.ask.view.App != "Zoom" || s.meetingActive() {
		t.Fatalf("the card %+v, meeting active %v", s.ask.view, s.meetingActive())
	}
	s.startDetectedMeeting()
	s.pump("the meeting", func() bool { return s.meetSnap.State == meeting.StateRecording })
	if s.ask.view.Visible {
		t.Fatal("the card stayed while Voxt records")
	}
	s.pump("its live text", func() bool { return len(s.meetSnap.Live) > 0 || len(s.meetSnap.Segments) > 0 })
	s.meetingFinish()
	s.pump("its end", func() bool { return !s.meetingActive() })
	list, err := s.lib.store.Meetings().ListMeetings(context.Background(), store.MeetingQuery{})
	if err != nil || len(list) != 1 {
		t.Fatalf("meetings kept: %+v %v", list, err)
	}
	if s.ask.view.Visible || s.notice != nil && !s.notice.ok { // a done notice ("Meeting saved") may show
		t.Fatalf("after the meeting, Zoom still recording: card %+v, notice %+v", s.ask.view, s.notice)
	}
}

// Zoom on the microphone outside a meeting (its settings' microphone test)
// brings no card; the card comes with its meeting process (11-25).
func TestMeetingDetectZoomNeedsItsMeetingEndToEnd(t *testing.T) {
	joined := proto.MicUsers{Apps: []proto.App{{PID: 7000, BundleID: "us.zoom.xos"}}, Hosts: []proto.MicHost{{PID: 8000, Name: "aomhost", App: 7000}}}
	line := string(mustJSON(helpertest.Script{AfterMS: 600, Method: proto.MethodMicChanged, Params: mustJSON(joined)}))
	s := detectApp(t, "hosts=,"+script(t, line), nil, nil)
	s.pump("Zoom's microphone test", func() bool { return len(s.micUsers) == 1 && s.micUsers[0].Hosts != nil })
	s.drain(150 * time.Millisecond) // well past the test's Hold
	if s.ask.view.Visible {
		t.Fatalf("a card for Zoom's microphone test: %+v", s.ask.view)
	}
	s.pump("the card, Zoom in a meeting", func() bool { return s.ask.view.Visible })
	if s.ask.view.App != "Zoom" {
		t.Fatalf("the card %+v", s.ask.view)
	}
}

// With Start recording automatically the meeting starts with no card.
func TestMeetingDetectAutoStartEndToEnd(t *testing.T) {
	s := detectApp(t, "", func(st *settings.Settings) { st.Features.Meeting.AutoStart = true }, hearsMeeting())
	s.pump("the meeting", func() bool { return s.meetSnap.State == meeting.StateRecording })
	if s.ask.view.Visible {
		t.Fatal("automatic, and the card asks")
	}
	// Words first: one ended before any would ask "Started by mistake?".
	s.pump("its live text", func() bool { return len(s.meetSnap.Live) > 0 || len(s.meetSnap.Segments) > 0 })
	s.meetingFinish()
	s.pump("its end", func() bool { return !s.meetingActive() })
}

// A start that fails says why in the overlay, where the user is, with the
// page that fixes it; the main window's toast alone may never be seen.
func TestMeetingDetectStartFailsEndToEnd(t *testing.T) {
	s := detectApp(t, "", nil, nil)
	s.pump("the card", func() bool { return s.ask.view.Visible })
	s.startDetectedMeeting()
	s.pump("the notice", func() bool { return s.notice != nil })
	want := meetingStartProblem(local.NotInstalledError(local.ASR, ""))
	if s.notice.text != want || s.notice.action != actionModels || s.ask.view.Visible || s.meetingActive() {
		t.Fatalf("no speech model: notice %+v, card %+v", s.notice, s.ask.view)
	}
	if s.overlayModeNow() != ovNotice {
		t.Fatalf("the overlay shows %q", s.overlayModeNow())
	}
}

// The meeting ends in its app: the helper says nobody records, and the
// card goes by itself.
func TestMeetingDetectEndsEndToEnd(t *testing.T) {
	s := detectApp(t, script(t, micLine(400)), nil, hearsMeeting())
	s.pump("the card", func() bool { return s.ask.view.Visible })
	s.pump("the meeting's end", func() bool { return !s.ask.view.Visible })
	if s.meetingActive() {
		t.Fatal("a meeting started by itself")
	}
}

// A helper that restarts is watched again, and the card stays: the
// meeting it asks about goes on.
func TestMeetingDetectHelperRestartEndToEnd(t *testing.T) {
	s := detectApp(t, "", nil, hearsMeeting())
	s.pump("the card", func() bool { return s.ask.view.Visible })
	first := s.helperStatus().PID
	sup := s.host.sup
	go sup.Restart()
	s.pump("the new helper", func() bool {
		st := s.helperStatus()
		return st.State == helper.Ready && st.PID != first && s.micWatch
	})
	s.drain(0)
	if !s.ask.view.Visible || s.ask.view.App != "Zoom" {
		t.Fatalf("after the helper restarted: %+v", s.ask.view)
	}
}

// The card's button pressed while a dictation records: the dictation has
// the microphone (a second recording of it fails, "Could not record the
// microphone"), so the meeting starts when the dictation ends.
func TestMeetingDetectDuringDictationEndToEnd(t *testing.T) {
	s := detectApp(t, script(t,
		hotkeyLine(300, "transcribe/0", proto.PhaseTrigger),
		hotkeyLine(700, "transcribe/0", proto.PhaseTrigger),
	), nil, hearsMeeting())
	s.pump("the card", func() bool { return s.ask.view.Visible })
	s.pump("the dictation", func() bool { return s.sess.State == session.Recording })
	if !s.ask.view.Visible {
		t.Fatal("the dictation took the card away")
	}
	s.startDetectedMeeting()
	if s.ask.view.Visible || s.meetingActive() || len(s.toasts) != 0 {
		t.Fatalf("pressed during the dictation: card %+v, meeting %s, toasts %q", s.ask.view, s.meetSnap.State, s.toasts)
	}
	s.pump("the dictation's end and the meeting", func() bool {
		return !s.sess.State.Live() && s.meetSnap.State == meeting.StateRecording
	})
	if len(s.toasts) != 0 || s.ask.startAfter != "" {
		t.Fatalf("the meeting after the dictation: toasts %q, waiting for %q", s.toasts, s.ask.startAfter)
	}
	s.pump("its live text", func() bool { return len(s.meetSnap.Live) > 0 || len(s.meetSnap.Segments) > 0 })
	s.meetingFinish()
	s.pump("its end", func() bool { return !s.meetingActive() })
}

// Voxt records the meeting, and its capture (silence here, not the fake
// helper's tone) hears no sound: the card asks whether the meeting is
// over, and its button ends the recording.
func TestMeetingDetectEndedEndToEnd(t *testing.T) {
	quiet := filepath.Join(t.TempDir(), "quiet.pcm")
	if err := os.WriteFile(quiet, make([]byte, 2), 0o644); err != nil {
		t.Fatal(err)
	}
	s := detectApp(t, script(t, micLine(1500))+","+helpertest.Opt("pcm", quiet)+","+helpertest.Opt("system", quiet), nil, hearsMeeting())
	s.recorded.quietAsk = 300 * time.Millisecond
	s.pump("the card", func() bool { return s.ask.view.Visible })
	s.startDetectedMeeting()
	s.pump("the meeting", func() bool { return s.meetSnap.State == meeting.StateRecording })
	s.pump("the card that asks whether it ended", func() bool { return s.ask.view.Visible && s.ask.view.End })
	if view := s.ask.view; view.Title != "Meeting ended?" || view.App != "Nobody spoke for 5 minutes" || view.StartLabel != "End recording" || len(view.Menu) != 0 {
		t.Fatalf("the card: %+v", view)
	}
	if !s.meetingActive() {
		t.Fatal("the recording ended by itself")
	}
	s.startDetectedMeeting()
	// A silent meeting this short heard no words: it asks "Started by
	// mistake?" (a real one is 5 minutes old by now, past MistakeWithin).
	s.pump("the question", func() bool { return s.meetSnap.State == meeting.StateAsking })
	if !s.meetSnap.Mistake {
		t.Fatalf("a silent short meeting: %+v", s.meetSnap)
	}
	s.meetingSave()
	s.pump("the recording's end", func() bool { return !s.meetingActive() })
	list, err := s.lib.store.Meetings().ListMeetings(context.Background(), store.MeetingQuery{})
	if err != nil || len(list) != 1 || s.ask.view.Visible {
		t.Fatalf("ended from the card: meetings %+v %v, card %+v", list, err, s.ask.view)
	}
}
