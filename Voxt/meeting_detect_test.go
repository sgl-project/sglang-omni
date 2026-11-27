package main

import (
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meetdetect"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// detectClock gives a fake timers and their clock, once: the tests move
// the time an app has recorded by hand.
func detectClock(t *testing.T, a *app) *fakeTimers {
	t.Helper()
	if f, ok := detectClocks.Load(a); ok {
		return f.(*fakeTimers)
	}
	f := (&fakeTimers{}).install(a)
	a.now = f.clock
	detectClocks.Store(a, f)
	t.Cleanup(func() { detectClocks.Delete(a) })
	return f
}

var detectClocks sync.Map // *app → *fakeTimers

// micUsers sends the app a mic.changed naming the bundle ids that record,
// and lets them record for meetdetect.Hold.
func micUsers(t *testing.T, a *app, ids ...string) {
	t.Helper()
	timers := detectClock(t, a)
	micSays(t, a, ids...)
	timers.advance(meetdetect.Hold)
}

// micSays sends the mic.changed alone: the apps have only just started.
func micSays(t *testing.T, a *app, ids ...string) {
	t.Helper()
	users := proto.MicUsers{Apps: []proto.App{}}
	for i, id := range ids {
		users.Apps = append(users.Apps, proto.App{PID: 700 + i, BundleID: id})
	}
	m, err := proto.NewNotification(proto.MethodMicChanged, users)
	if err != nil {
		t.Fatal(err)
	}
	a.helperNotified(m)
}

// A meeting app that starts recording is asked about once, with the card;
// the card goes when the app lets go of the microphone and, unanswered,
// comes back at its next use.
func TestMeetingDetectedAsksOnce(t *testing.T) {
	a, fs, _ := newTestApp(t)
	if !slices.Contains(fs.methods(), proto.MethodMicWatch) {
		t.Fatalf("the watch was not turned on: %v", fs.methods())
	}
	micUsers(t, a, "com.apple.CoreSpeech", "com.tencent.xinWeChat")
	if a.ask.view.Visible {
		t.Fatalf("asked about apps that are no meeting: %+v", a.ask.view)
	}
	micUsers(t, a, "com.apple.CoreSpeech", "us.zoom.xos")
	view := a.ask.view
	if !view.Visible || view.Title != "Meeting detected" || view.App != "Zoom" || view.StartLabel != "Start recording" || view.Compact || view.MenuOpen {
		t.Fatalf("Zoom records: the card is %+v", view)
	}
	if a.meetingActive() || len(a.toasts) != 0 {
		t.Fatal("a meeting started without the user's click")
	}
	logged := false
	for _, e := range a.logs.Tail(20, meetingLogSource) {
		logged = logged || e.Message == "detected us.zoom.xos, on the microphone for 1s"
	}
	if !logged {
		t.Fatalf("no line says when the card came: %+v", a.logs.Tail(20, meetingLogSource))
	}

	micUsers(t, a, "com.apple.CoreSpeech")
	if a.ask.view.Visible {
		t.Fatal("the card stayed after Zoom let go of the microphone")
	}
	micUsers(t, a, "us.zoom.xos")
	if !a.ask.view.Visible {
		t.Fatal("a mute before the card was answered lost the question")
	}
	a.dismissMeetingAsk()
	micUsers(t, a, "com.apple.CoreSpeech")
	micUsers(t, a, "us.zoom.xos")
	if a.ask.view.Visible {
		t.Fatal("an unmute after the card was closed asked again")
	}
}

// The card's button starts a meeting; its close button only hides it, for
// this meeting.
func TestMeetingAskStartAndDismiss(t *testing.T) {
	a, _, _ := newTestApp(t)
	micUsers(t, a, "us.zoom.xos")
	a.startDetectedMeeting()
	// The start itself is meeting_link.go's: here it runs, and without a
	// speech model it says why no meeting began.
	if a.ask.view.Visible || !a.meetingActive() && len(a.toasts) != 1 {
		t.Fatalf("the button did not start a meeting: card %+v, toasts %q", a.ask.view, a.toasts)
	}

	b, _, _ := newTestApp(t)
	micUsers(t, b, "us.zoom.xos")
	b.dismissMeetingAsk()
	micUsers(t, b, "us.zoom.xos", "com.apple.CoreSpeech")
	if b.ask.view.Visible || b.meetingActive() || len(b.toasts) != 0 {
		t.Fatalf("dismissed: card %+v, toasts %q", b.ask.view, b.toasts)
	}
}

// While Voxt records a meeting, or the user dictates, nothing asks; a
// meeting found during a dictation is asked about when it ends.
func TestMeetingDetectedWaits(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.meetSnap = meeting.Snapshot{ID: "m1", State: meeting.StateRecording}
	micUsers(t, a, "us.zoom.xos")
	if a.ask.view.Visible {
		t.Fatal("asked while Voxt records the meeting")
	}
	a.meetingChanged(meeting.Snapshot{})
	if a.ask.view.Visible {
		t.Fatal("asked when Voxt's meeting ended, about the app it recorded")
	}

	b, _, _ := newTestApp(t)
	b.sess = session.Snapshot{ID: "s1", State: session.Recording}
	micUsers(t, b, "com.microsoft.teams2.helper")
	if b.ask.view.Visible {
		t.Fatal("asked in the middle of a dictation")
	}
	was := b.sess
	b.sess = session.Snapshot{ID: "s1", State: session.Done}
	for _, fn := range b.afterSession {
		fn(was, b.sess)
	}
	if !b.ask.view.Visible || b.ask.view.App != "Microsoft Teams" {
		t.Fatalf("after the dictation: %+v", b.ask.view)
	}
	b.meetingChanged(meeting.Snapshot{ID: "m2", State: meeting.StateStarting})
	if b.ask.view.Visible {
		t.Fatal("the card stayed while Voxt records")
	}
}

// A browser that records is a meeting only on a meeting's page.
func TestMeetingDetectedInBrowser(t *testing.T) {
	a, fs, _ := newTestApp(t)
	browserURLs(fs, map[string]string{
		"com.google.Chrome": "https://meet.google.com/abc-defg-hij",
		"com.apple.Safari":  "https://chatgpt.com/",
	})
	micUsers(t, a, "com.apple.WebKit.GPU")
	if a.ask.view.Visible {
		t.Fatalf("a voice chat in Safari asked: %+v", a.ask.view)
	}
	micUsers(t, a, "com.apple.WebKit.GPU", "com.google.Chrome.helper")
	if !a.ask.view.Visible || a.ask.view.App != "Google Meet" {
		t.Fatalf("Google Meet in Chrome: %+v", a.ask.view)
	}
}

// Settings → Meetings → Detect meetings off: the watch stops, the card
// goes, and nothing asks.
func TestMeetingDetectOff(t *testing.T) {
	a, fs, _ := newTestApp(t)
	micUsers(t, a, "us.zoom.xos")
	a.setMeetingDetect(false)
	if a.ask.view.Visible || !a.prefs.Features.Meeting.NoDetect {
		t.Fatalf("off: card %+v, setting %+v", a.ask.view, a.prefs.Features.Meeting)
	}
	fs.mu.Lock()
	last := fs.calls[len(fs.calls)-1]
	fs.mu.Unlock()
	var p proto.MicWatchParams
	if last.Method != proto.MethodMicWatch || last.DecodeParams(&p) != nil || p.On {
		t.Fatalf("the watch was not turned off: %s %s", last.Method, last.Params)
	}
	micUsers(t, a, "com.microsoft.teams2")
	if a.ask.view.Visible {
		t.Fatal("asked with the detection off")
	}
	state := buildSettingsMeetingViewState(i18n.NewTranslator("en"), settingsMeetingStateInput{})
	if !state.DetectMeetings || state.AutoStart {
		t.Fatalf("by default the detection asks: %+v", state)
	}
}

// A browser read on another tab is read again while it records: every
// meetingTabAgain, and as it comes to the front. Its meeting tab, shown
// later, asks (11-25: it was never read again).
func TestMeetingDetectReadsTheTabAgain(t *testing.T) {
	a, fs, _ := newTestApp(t)
	browserURLs(fs, map[string]string{"com.google.Chrome": "https://mail.google.com/"})
	micUsers(t, a, "com.google.Chrome.helper")
	if a.ask.view.Visible {
		t.Fatalf("mail asked: %+v", a.ask.view)
	}
	browserURLs(fs, map[string]string{"com.google.Chrome": "https://meet.google.com/abc-defg-hij"})
	detectClock(t, a).advance(time.Second)
	if !a.ask.view.Visible || a.ask.view.App != "Google Meet" {
		t.Fatalf("its meeting tab, read again: %+v", a.ask.view)
	}

	b, fsb, _ := newTestApp(t)
	browserURLs(fsb, map[string]string{"com.google.Chrome": "https://mail.google.com/"})
	micUsers(t, b, "com.google.Chrome.helper")
	browserURLs(fsb, map[string]string{"com.google.Chrome": "https://meet.google.com/abc-defg-hij"})
	b.frontChanged(proto.App{PID: 900, BundleID: "com.google.Chrome"})
	if !b.ask.view.Visible {
		t.Fatalf("Chrome in front with its meeting: %+v", b.ask.view)
	}
}
