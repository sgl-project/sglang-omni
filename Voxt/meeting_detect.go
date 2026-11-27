package main

// Meetings detected (Settings → Meetings → Detect meetings): the helper's
// mic.watch says which apps record from the microphone, internal/meetdetect
// picks the meeting apps among them, and a card asks once per meeting
// (meeting_ask_view.go, its window in meeting_ask_window.go): its button
// starts a meeting in Voxt. Nothing starts by itself unless the user chose
// so (the card's menu, or Meetings → settings): recording other people is
// their decision.

import (
	"errors"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meetdetect"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// The card's times: it turns compact after meetingAskRest without the
// pointer over it, and "Ask again in 2 minutes" is meetingAskLater.
const (
	meetingAskRest  = 8 * time.Second
	meetingAskLater = 2 * time.Minute
	// meetingTabAgain is the first retry; repeated misses back off to 5 s.
	meetingTabAgain = time.Second
)

// meetingDetectState is the detection as the main thread knows it.
type meetingDetectState struct {
	detect meetdetect.Detector
	// micWatch: mic.watch is on in the helper that runs now.
	micWatch bool
	micUsers []meetdetect.User
	// reading is the browser whose tab is being read; tree the browser
	// that records and needs its page tree turned on for that
	// (appenhance.go: syncTrees), "" for none.
	reading, tree string
	// tabAgain has the browsers that record whose tab was no meeting's
	// (or could not be read), each read again from its time on.
	tabAgain map[string]meetingTabRetry
	ask      meetingAsk
	recorded meetingRecorded // meeting_detect_end.go
}

type meetingTabRetry struct {
	next  time.Time
	delay time.Duration
}

// meetingAsk is the card on screen: the app it asks about (key is the
// detector's, "" for no card), its compact and menu states, and seq, which
// numbers its timer so that a later card's is not this one's.
type meetingAsk struct {
	key, app      string
	compact, menu bool
	// end: the card asks whether the meeting Voxt records is over
	// (meeting_detect_end.go), not whether to record one.
	end   bool
	seq   int
	armed time.Time
	// snoozed is the app to ask about again when "Ask again in 2 minutes"
	// is up ("": none, or Voxt recorded it meanwhile).
	snoozed string
	// startAfter is the app whose meeting starts when the dictation that
	// records now ends ("": none): its button was pressed meanwhile.
	startAfter string

	meetingAskWindow // meeting_ask_window.go
}

// detectMeetings reports whether the user wants meetings detected.
func (a *app) detectMeetings() bool {
	f := a.prefs.Features
	return a.meetMgr != nil && !f.Meeting.NoDetect && !f.IsHidden(settings.HideMeetings)
}

// meetingDetectChanged follows the settings (shell_settings_controller.go):
// the watch stops and the card goes when meetings are no longer detected
// or Meetings is hidden, and the card goes when its question is decided.
func (a *app) meetingDetectChanged() {
	switch {
	case !a.detectMeetings():
		a.closeMeetingAsk()
	case a.prefs.Features.Meeting.AutoStart && a.ask.key != "" && !a.ask.end && !meetdetect.IsChat(a.ask.app):
		a.startDetectedMeeting() // the card asked what is now decided: yes
	}
	a.syncMicWatch()
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
	params := proto.MicWatchParams{On: want}
	if want {
		params.Hosts = meetdetect.HostNames() // Zoom's meeting processes
	}
	helperCall(a, proto.MethodMicWatch, params, 5*time.Second, func(users proto.MicUsers, err error) {
		switch {
		case err != nil && want:
			// Not watching after all: the next change of the helper or of
			// the settings asks again.
			a.micWatch = false
			a.logs.Append(core.LogEntry{Time: a.now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: "mic.watch: " + err.Error()})
		case err == nil && want && a.micWatch:
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
	a.micUsers = make([]meetdetect.User, 0, len(users.Apps)) // not nil: the helper answered
	for _, u := range users.Apps {
		user := meetdetect.User{PID: u.PID, BundleID: u.BundleID}
		if users.Hosts != nil {
			user.Hosts = []string{} // the helper looked: none, or these
			for _, h := range users.Hosts {
				if h.App == u.PID || h.PID == u.PID {
					user.Hosts = append(user.Hosts, h.Name)
				}
			}
		}
		a.micUsers = append(a.micUsers, user)
	}
	a.syncMeetingDetect()
}

// syncMeetingDetect asks about a meeting app that records and was not
// asked about yet. It runs when the apps recording change, and when what
// held the question back is over (a dictation, Voxt's own meeting, or its
// final pass and question).
func (a *app) syncMeetingDetect() {
	var browsers []string
	for _, b := range appctx.Browsers(a.prefs.AppEnhancement.Browsers) {
		browsers = append(browsers, b.BundleID)
	}
	found, wait := a.detect.Update(a.micUsers, browsers, a.now())
	if a.ask.key != "" && !a.ask.end && !a.detect.Recording(a.ask.key) {
		a.detect.Forget(a.ask.key) // unanswered: a mute may have let go of the microphone
		a.closeMeetingAsk()        // the app let go of the microphone
	}
	if a.tree != "" && !a.detect.Recording(a.tree) {
		a.tree = ""
		a.syncTrees() // appenhance.go: the browser no longer records
	}
	maps.DeleteFunc(a.tabAgain, func(key string, _ meetingTabRetry) bool {
		return !a.detect.Recording(key) // read again only while it records
	})
	if wait > 0 {
		// An app took the microphone a moment ago: a test of it is over
		// before meetdetect.Hold, a meeting is not.
		a.later(wait, a.syncMeetingDetect)
	}
	if len(found) == 0 || !a.detectMeetings() {
		return
	}
	if a.meetingActive() {
		if !a.meetingRecords() {
			return // Voxt's last meeting is being filed: asked once it is
		}
		for _, c := range found {
			a.detect.Settle(c.Key) // Voxt records it already
		}
		return
	}
	if a.sess.State.Live() || a.ob.open {
		return // a dictation, the setup guide: asked once it is over
	}
	for _, c := range found {
		if c.App != "" {
			a.askMeeting(c.Key, c.App)
			return
		}
	}
	for _, c := range found {
		if retry, ok := a.tabAgain[c.Key]; !ok || !a.now().Before(retry.next) {
			a.readMeetingTab(c) // a browser not read yet, or whose time came
			return
		}
	}
}

// readMeetingTab reads the front tab of a browser that records: it is a
// meeting only on a meeting's page.
func (a *app) readMeetingTab(c meetdetect.Candidate) {
	if a.reading != "" {
		return
	}
	a.reading = c.Key
	if a.tree == "" && slices.Contains(appctx.TreeOnDemand(a.prefs.AppEnhancement.Browsers), c.Browser) {
		// A Chromium browser builds a page's tree only when asked (Arc
		// and Dia show no address without it): asked while it records.
		a.tree = c.Browser
	}
	a.sendTrees(func() { a.readMeetingTabNow(c) })
}

func (a *app) readMeetingTabNow(c meetdetect.Candidate) {
	helperCall(a, proto.MethodBrowserURL, proto.BrowserURLParams{BundleID: c.Browser}, 2*time.Second, func(tab proto.BrowserURLResult, err error) {
		a.reading = ""
		app, ok := meetdetect.Web(tab.URL)
		switch {
		case !a.detect.Recording(c.Key):
			delete(a.tabAgain, c.Key) // it let go of the microphone
		case err != nil || !ok:
			// Not a meeting's page now, or none read (no Accessibility
			// yet): read again while it records, when it comes to the
			// front and on a bounded backoff. Settled, it was never asked
			// again (11-25: the meeting tab came to the front later).
			if a.tabAgain == nil {
				a.tabAgain = map[string]meetingTabRetry{}
			}
			delay := meetingTabAgain
			if prior := a.tabAgain[c.Key].delay; prior > 0 {
				delay = 2 * time.Second
				if prior >= delay {
					delay = 5 * time.Second
				}
			}
			a.tabAgain[c.Key] = meetingTabRetry{next: a.now().Add(delay), delay: delay}
			a.later(delay, func() { a.readMeetingTabAgain(c.Key) })
			// Another browser that records may be read meanwhile.
		case a.sess.State.Live() || a.meetingActive() || a.ob.open:
			// Held back meanwhile: syncMeetingDetect decides again.
		default:
			delete(a.tabAgain, c.Key)
			a.askMeeting(c.Key, app)
			return
		}
		a.syncMeetingDetect()
	})
}

// readMeetingTabAgain reads the tab of the browser with key again, if its
// tab was no meeting's: on its timer, or as it came to the front
// (meetingTabFront).
func (a *app) readMeetingTabAgain(key string) {
	retry, ok := a.tabAgain[key]
	if !ok || a.now().Before(retry.next) {
		return
	}
	retry.next = time.Time{}
	a.tabAgain[key] = retry
	a.syncMeetingDetect()
}

// meetingTabFront follows the app in front (appenhance.go: frontChanged):
// a browser that records, its tab no meeting's, may show its meeting now.
func (a *app) meetingTabFront(front proto.App) {
	for key := range a.tabAgain {
		if strings.EqualFold(front.BundleID, key) {
			a.tabAgain[key] = meetingTabRetry{} // foreground changes bypass and reset the backoff
			a.readMeetingTabAgain(key)
			return
		}
	}
}

// meetingAfterDictation runs when a dictation ended (events.go): the
// meeting whose button was pressed during it starts, if its app still
// records; else a meeting found during it is asked about.
func (a *app) meetingAfterDictation() {
	key := a.ask.startAfter
	a.ask.startAfter = ""
	if key != "" && a.detect.Recording(key) && a.detectMeetings() && !a.meetingActive() {
		a.startMeetingFor(key)
		return
	}
	a.syncMeetingDetect()
}

// askMeeting offers to record the meeting in app with the card, or starts
// it at once when the user chose so.
func (a *app) askMeeting(key, app string) {
	a.detect.Settle(key)
	// Where a slow card's time went: the app's own start, or the wait here.
	msg := "detected " + key
	if since, ok := a.detect.Since(key); ok {
		msg += ", on the microphone for " + a.now().Sub(since).Round(10*time.Millisecond).String()
	}
	a.logs.Append(core.LogEntry{Time: a.now(), Source: meetingLogSource, Level: core.SeverityInfo, Message: msg})
	if a.prefs.Features.Meeting.AutoStart && !meetdetect.IsChat(app) {
		a.startMeetingFor(key)
		return
	}
	a.showMeetingAsk(key, app, false)
}

// showMeetingAsk shows the card about app: whether to record its meeting,
// or (end) whether the meeting Voxt records is over.
func (a *app) showMeetingAsk(key, app string, end bool) {
	a.ask.key, a.ask.app, a.ask.end, a.ask.compact, a.ask.menu = key, app, end, false, false
	a.syncMeetingAsk()
	a.restMeetingAsk()
}

// restMeetingAsk makes the card compact once meetingAskRest passed with
// neither the pointer over it nor its menu open.
func (a *app) restMeetingAsk() {
	a.ask.seq++
	seq := a.ask.seq
	a.ask.armed = a.now()
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
	a.ask.key, a.ask.app, a.ask.menu, a.ask.end = "", "", false, false
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
	} else if a.now().Sub(a.ask.armed) > time.Second {
		a.restMeetingAsk()
	}
}

// startDetectedMeeting starts a meeting for the one detected. The user
// is in the meeting's app, and Voxt's window may be closed: a start that
// fails says so in the overlay too, with the way to fix it.
func (a *app) startDetectedMeeting() {
	key, end := a.ask.key, a.ask.end
	a.closeMeetingAsk()
	if end {
		a.meetingFinish() // "End recording": the meeting it asked about is over
		return
	}
	a.startMeetingFor(key)
}

// startMeetingFor starts a meeting for the app the detector knows as key.
func (a *app) startMeetingFor(key string) {
	if a.sess.State.Live() && key != "" {
		// The dictation has the microphone, and a second recording of it
		// fails: the meeting starts when the dictation ends.
		a.ask.startAfter = key
		return
	}
	a.startMeetingThen(func(msg string, err error) {
		a.showNotice(sessionNotice{text: msg, action: meetingStartAction(err)})
	})
}

// meetingStartAction is where a meeting that did not start is fixed: the
// speech models when none is downloaded, the permissions when macOS
// refused the recording.
func meetingStartAction(err error) string {
	var start *meeting.StartError
	switch {
	case errors.Is(err, local.ErrNotInstalled):
		return actionModels
	case errors.As(err, &start) && errors.Is(start.Err, &proto.Error{Code: proto.CodePermissionDenied}):
		return actionPermissions
	}
	return actionNone
}

func (a *app) dismissMeetingAsk() {
	a.closeMeetingAsk() // the end question: asked again after a sound
}

func (a *app) toggleMeetingAskMenu() {
	a.ask.menu = !a.ask.menu
	a.syncMeetingAsk()
}

func (a *app) chooseMeetingAskMenu(id string) {
	switch id {
	case meetingAskSnooze:
		key := a.ask.key
		a.closeMeetingAsk()
		a.ask.snoozed = key
		a.later(meetingAskLater, func() {
			if a.ask.snoozed != key {
				return // Voxt recorded it meanwhile, or it was asked later again
			}
			a.ask.snoozed = ""
			a.detect.Forget(key) // still recording: syncMeetingDetect asks again
			a.syncMeetingDetect()
		})
	case meetingAskAuto:
		a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.AutoStart = true })
		a.meetingDetectChanged() // starts this one, unless the saved setting already did
	case meetingAskSettings:
		a.ask.menu = false
		a.syncMeetingAsk()
		a.showMainWindow()
		a.navigate("/meetings")
		a.mvUI.setOpen = true
	}
}
