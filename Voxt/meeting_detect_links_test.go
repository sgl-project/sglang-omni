package main

// How the detection of meetings meets the rest of the app: the hidden
// Meetings feature, the setup guide, the language, a dictation that
// records, and a start that fails.

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meetdetect"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// With Meetings hidden the watch stops and nothing asks; shown again, the
// watch is back.
func TestMeetingDetectFollowsHiddenMeetings(t *testing.T) {
	a, _, _ := newTestApp(t)
	micUsers(t, a, "us.zoom.xos")
	if !a.ask.view.Visible || !a.micWatch {
		t.Fatalf("before: card %+v, watching %v", a.ask.view, a.micWatch)
	}
	shown := a.prefs
	hidden := shown
	hidden.Features.Hidden = []string{settings.HideMeetings}
	a.settingsChanged(shown, hidden)
	if a.ask.view.Visible || a.micWatch {
		t.Fatalf("Meetings hidden: card %+v, watching %v", a.ask.view, a.micWatch)
	}
	micUsers(t, a, "com.microsoft.teams2")
	if a.ask.view.Visible {
		t.Fatal("asked with Meetings hidden")
	}
	a.settingsChanged(hidden, shown)
	if !a.micWatch {
		t.Fatal("Meetings shown again, and nothing watches")
	}
}

// The setup guide is not interrupted: a meeting found while it is open is
// asked about when it closes.
func TestMeetingDetectWaitsForTheGuide(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.ob.open = true
	micUsers(t, a, "us.zoom.xos")
	if a.ask.view.Visible {
		t.Fatal("the card came over the setup guide")
	}
	a.closeOnboarding()
	if !a.ask.view.Visible || a.ask.view.App != "Zoom" {
		t.Fatalf("after the guide: %+v", a.ask.view)
	}
}

// The card is written in the language now, also when it changed under it.
func TestMeetingAskFollowsLanguage(t *testing.T) {
	a, _, _ := newTestApp(t)
	micUsers(t, a, "com.tencent.meeting")
	if got := a.meetingAskViewNow(); got.Title != "Meeting detected" || got.App != "Tencent Meeting" {
		t.Fatalf("in English: %+v", got)
	}
	i18n.SetLocale("zh-Hans")
	if got := a.meetingAskViewNow(); got.Title != "检测到会议" || got.App != "腾讯会议" || got.StartLabel != "开始记录" || len(got.Menu) != 3 || got.Menu[0].Label != "2 分钟后再问" {
		t.Fatalf("in Chinese: %+v", got)
	}
}

// The button pressed during a dictation waits for its end; a meeting that
// ended meanwhile is not recorded.
func TestMeetingAskStartWaitsForDictation(t *testing.T) {
	a, _, _ := newTestApp(t)
	micUsers(t, a, "us.zoom.xos")
	a.sess = session.Snapshot{ID: "s1", State: session.Recording}
	a.startDetectedMeeting()
	if a.ask.view.Visible || a.ask.startAfter != "us.zoom.xos" || len(a.toasts) != 0 {
		t.Fatalf("pressed during a dictation: card %+v, waiting for %q, toasts %q", a.ask.view, a.ask.startAfter, a.toasts)
	}
	end := func() {
		was := a.sess
		a.sess = session.Snapshot{ID: was.ID, State: session.Done}
		for _, fn := range a.afterSession {
			fn(was, a.sess)
		}
	}
	end()
	// No speech model here: the start runs and says why no meeting began.
	if a.ask.startAfter != "" || !a.meetingActive() && len(a.toasts) != 1 {
		t.Fatalf("after the dictation: waiting for %q, toasts %q", a.ask.startAfter, a.toasts)
	}

	b, _, _ := newTestApp(t)
	micUsers(t, b, "us.zoom.xos")
	b.sess = session.Snapshot{ID: "s1", State: session.Recording}
	b.startDetectedMeeting()
	micUsers(t, b) // the meeting ended during the dictation
	a = b
	end()
	if b.ask.startAfter != "" || b.meetingActive() || len(b.toasts) != 0 {
		t.Fatalf("the meeting ended first: waiting for %q, toasts %q", b.ask.startAfter, b.toasts)
	}
}

// A start that fails names the page that fixes it.
func TestMeetingStartAction(t *testing.T) {
	denied := &meeting.StartError{Source: meeting.SourceMic, Err: &proto.Error{Code: proto.CodePermissionDenied}, Guidance: meeting.GuidanceMic}
	for name, c := range map[string]struct {
		err  error
		want string
	}{
		"no speech model":      {local.NotInstalledError(local.ASR, ""), actionModels},
		"microphone refused":   {denied, actionPermissions},
		"microphone gave none": {&meeting.StartError{Source: meeting.SourceMic, Err: errors.New("no input"), Guidance: meeting.GuidanceMic}, actionNone},
		"another meeting runs": {meeting.ErrBusy, actionNone},
		"anything else":        {errors.New("disk full"), actionNone},
	} {
		if got := meetingStartAction(c.err); got != c.want {
			t.Errorf("%s: action %q, want %q", name, got, c.want)
		}
	}
}

// An app that has the microphone for a moment (a microphone test, a voice
// clip) is no meeting: no card, and nothing starts by itself.
func TestMeetingDetectHoldsBeforeAsking(t *testing.T) {
	a, _, _ := newTestApp(t)
	timers := detectClock(t, a)
	a.updateSettings(func(s *settings.Settings) { s.Features.Meeting.AutoStart = true })
	micSays(t, a, "us.zoom.xos")
	timers.advance(meetdetect.Hold - time.Second)
	micSays(t, a) // Zoom's microphone test is over
	timers.advance(time.Minute)
	if a.ask.view.Visible || a.meetingActive() || len(a.toasts) != 0 {
		t.Fatalf("two seconds of Zoom: card %+v, meeting %s, toasts %q", a.ask.view, a.meetSnap.State, a.toasts)
	}
	micSays(t, a, "us.zoom.xos")
	timers.advance(meetdetect.Hold - time.Millisecond)
	if len(a.toasts) != 0 {
		t.Fatalf("started before Zoom held the microphone: %q", a.toasts)
	}
	timers.advance(time.Millisecond)
	// No speech model here: the start runs and says why no meeting began.
	if !a.meetingActive() && len(a.toasts) != 1 {
		t.Fatalf("a meeting that holds the microphone, automatic: toasts %q", a.toasts)
	}
}

// "Ask again in 2 minutes" asks nothing when Voxt recorded the meeting
// meanwhile.
func TestMeetingAskLaterAfterRecording(t *testing.T) {
	a, _, _ := newTestApp(t)
	timers := detectClock(t, a)
	micUsers(t, a, "us.zoom.xos")
	a.chooseMeetingAskMenu(meetingAskSnooze)
	a.meetingChanged(meeting.Snapshot{ID: "m1", State: meeting.StateRecording}) // started from the menu bar
	a.meetingChanged(meeting.Snapshot{})                                        // and ended
	timers.advance(meetingAskLater)
	if a.ask.view.Visible {
		t.Fatalf("asked again about a meeting Voxt recorded: %+v", a.ask.view)
	}
}

// A helper that refuses mic.watch is not watched, and the log says so; a
// change of the setting asks it again.
func TestMeetingDetectWatchRefused(t *testing.T) {
	a, fs, _ := newTestApp(t)
	if !a.micWatch {
		t.Fatal("the watch is not on")
	}
	a.setMeetingDetect(false)
	fs.mu.Lock()
	fs.handlers[proto.MethodMicWatch] = func(json.RawMessage) (any, error) { return nil, proto.Errorf(proto.CodeInternal, "no process list") }
	fs.mu.Unlock()
	before := len(fs.methods())
	a.setMeetingDetect(true)
	logged := false
	for _, e := range a.logs.Tail(50, meetingLogSource) {
		logged = logged || strings.Contains(e.Message, "mic.watch") && strings.Contains(e.Message, "no process list")
	}
	if a.micWatch || !logged || len(fs.methods()) == before {
		t.Fatalf("refused: watching %v, logged %v, %d calls", a.micWatch, logged, len(fs.methods())-before)
	}
	fs.mu.Lock()
	fs.handlers[proto.MethodMicWatch] = func(json.RawMessage) (any, error) { return proto.MicUsers{Apps: []proto.App{}}, nil }
	fs.mu.Unlock()
	a.setMeetingDetect(false)
	a.setMeetingDetect(true)
	if !a.micWatch {
		t.Fatal("asked again, and still not watching")
	}
}
