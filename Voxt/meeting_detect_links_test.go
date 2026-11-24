package main

// How the detection of meetings meets the rest of the app: the hidden
// Meetings feature, the setup guide, the language, a dictation that
// records, and a start that fails.

import (
	"errors"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
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
