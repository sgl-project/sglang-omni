package main

// Meetings detected (Settings → Meetings → Detect meetings): the helper's
// mic.watch says which apps record from the microphone, internal/meetdetect
// picks the meeting apps among them, and a card asks once per meeting
// (meeting_ask_view.go, its window in meeting_ask_window.go): its button
// starts a meeting in Voxt. Nothing starts by itself unless the user chose
// so (the card's menu, or Meetings → settings): recording other people is
// their decision.

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meetdetect"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// The card's times: it turns compact after meetingAskRest without the
// pointer over it, and "Ask again in 2 minutes" is meetingAskLater.
const (
	meetingAskRest  = 8 * time.Second
	meetingAskLater = 2 * time.Minute
)

// meetingDetectState is the detection as the main thread knows it.
type meetingDetectState struct {
	detect meetdetect.Detector
	// micWatch: mic.watch is on in the helper that runs now.
	micWatch bool
	micUsers []meetdetect.User
	// reading is the browser whose tab is being read.
	reading string
	ask     meetingAsk
}

// meetingAsk is the card on screen: the app it asks about (key is the
// detector's, "" for no card), its compact and menu states, and seq, which
// numbers its timer so that a later card's is not this one's.
type meetingAsk struct {
	key, app         string
	compact, menu    bool
	seq              int
	armed            time.Time
	meetingAskWindow // meeting_ask_window.go
}

// detectMeetings reports whether the user wants meetings detected.
func (a *app) detectMeetings() bool {
	return a.meetMgr != nil && !a.prefs.Features.Meeting.NoDetect
}

// syncMicWatch turns the helper's watch on or off, as the setting and the
// helper say; a new helper has none.
func (a *app) syncMicWatch() {
	want := a.detectMeetings() && a.helperReady() && a.hstatus.Has(proto.MethodMicWatch)
	if want == a.micWatch {
		return
	}
	a.micWatch = want
	if !want {
		a.micUsers = nil
		a.syncMeetingDetect()
		if !a.helperReady() {
			return
		}
	}
	helperCall(a, proto.MethodMicWatch, proto.MicWatchParams{On: want}, 5*time.Second, func(users proto.MicUsers, err error) {
		if err == nil && want && a.micWatch {
			a.micChanged(users)
		}
	})
}

// micHelperChanged follows the helper: one that went away took its watch
// with it.
func (a *app) micHelperChanged(st helper.Status) {
	if st.State != helper.Ready {
		a.micWatch = false
	}
	a.syncMicWatch()
}

// micChanged takes the apps recording now (mic.watch, mic.changed).
func (a *app) micChanged(users proto.MicUsers) {
	a.micUsers = a.micUsers[:0]
	for _, u := range users.Apps {
		a.micUsers = append(a.micUsers, meetdetect.User{PID: u.PID, BundleID: u.BundleID})
	}
	a.syncMeetingDetect()
}

// syncMeetingDetect asks about a meeting app that records and was not
// asked about yet. It runs when the apps recording change, and when what
// held the question back is over (a dictation, Voxt's own meeting).
func (a *app) syncMeetingDetect() {
	var browsers []string
	for _, b := range appctx.Browsers(a.prefs.AppEnhancement.Browsers) {
		browsers = append(browsers, b.BundleID)
	}
	found := a.detect.Update(a.micUsers, browsers, time.Now())
	if a.ask.key != "" && !a.detect.Recording(a.ask.key) {
		a.closeMeetingAsk() // the meeting is over
	}
	if len(found) == 0 || !a.detectMeetings() {
		return
	}
	if a.meetingActive() {
		for _, c := range found {
			a.detect.Settle(c.Key) // Voxt records it already
		}
		return
	}
	if a.sess.State.Live() {
		return // a dictation: afterSession asks once it ended
	}
	for _, c := range found {
		if c.App != "" {
			a.askMeeting(c.Key, c.App)
			return
		}
	}
	a.readMeetingTab(found[0])
}

// readMeetingTab reads the front tab of a browser that records: it is a
// meeting only on a meeting's page.
func (a *app) readMeetingTab(c meetdetect.Candidate) {
	if a.reading != "" {
		return
	}
	a.reading = c.Key
	helperCall(a, proto.MethodBrowserURL, proto.BrowserURLParams{BundleID: c.Browser}, 2*time.Second, func(tab proto.BrowserURLResult, err error) {
		a.reading = ""
		app, ok := meetdetect.Web(tab.URL)
		switch {
		case err != nil || !ok || !a.detect.Recording(c.Key):
			a.detect.Settle(c.Key) // not a meeting: this use asks nothing
		case a.sess.State.Live() || a.meetingActive():
			// Held back meanwhile: syncMeetingDetect decides again.
		default:
			a.askMeeting(c.Key, app)
			return
		}
		a.syncMeetingDetect()
	})
}

// askMeeting offers to record the meeting in app with the card, or starts
// it at once when the user chose so.
func (a *app) askMeeting(key, app string) {
	a.detect.Settle(key)
	if a.prefs.Features.Meeting.AutoStart {
		a.startMeeting()
		return
	}
	a.ask.key, a.ask.app, a.ask.compact, a.ask.menu = key, app, false, false
	a.syncMeetingAsk()
	a.restMeetingAsk()
}

// restMeetingAsk makes the card compact once meetingAskRest passed with
// neither the pointer over it nor its menu open.
func (a *app) restMeetingAsk() {
	a.ask.seq++
	seq := a.ask.seq
	a.ask.armed = time.Now()
	a.later(meetingAskRest, func() {
		switch {
		case a.ask.seq != seq || a.ask.key == "" || a.ask.compact:
		case a.ask.menu:
			a.restMeetingAsk()
		default:
			a.ask.compact = true
			a.syncMeetingAsk()
		}
	})
}

// closeMeetingAsk hides the card.
func (a *app) closeMeetingAsk() {
	if a.ask.key == "" {
		return
	}
	a.ask.key, a.ask.app, a.ask.menu = "", "", false
	a.ask.seq++
	a.syncMeetingAsk()
}

// meetingAskHovered is the pointer over the card: a compact card opens,
// and it stays open while the pointer moves over it. The view calls it
// while it builds, so the window changes in a post.
func (a *app) meetingAskHovered() {
	if a.ask.key == "" {
		return
	}
	if a.ask.compact {
		a.ask.compact = false
		a.post(a.syncMeetingAsk)
		a.restMeetingAsk()
	} else if time.Since(a.ask.armed) > time.Second {
		a.restMeetingAsk()
	}
}

func (a *app) startDetectedMeeting() {
	a.closeMeetingAsk()
	a.startMeeting()
}

func (a *app) dismissMeetingAsk() { a.closeMeetingAsk() }

func (a *app) toggleMeetingAskMenu() {
	a.ask.menu = !a.ask.menu
	a.syncMeetingAsk()
}

func (a *app) chooseMeetingAskMenu(id string) {
	switch id {
	case meetingAskSnooze:
		key := a.ask.key
		a.closeMeetingAsk()
		a.later(meetingAskLater, func() {
			a.detect.Forget(key) // still recording: syncMeetingDetect asks again
			a.syncMeetingDetect()
		})
	case meetingAskAuto:
		a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.AutoStart = true })
		a.startDetectedMeeting()
	case meetingAskSettings:
		a.ask.menu = false
		a.syncMeetingAsk()
		a.showMainWindow()
		a.navigate("/meetings")
		a.mvUI.setOpen = true
	}
}
