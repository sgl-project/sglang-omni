package main

import (
	"net/url"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

const meetingSpeakerPoll = 750 * time.Millisecond

type meetingSpeakerReader struct {
	id         string
	generation int
	target     proto.App
}

func (a *app) syncMeetingSpeakers(previous meeting.Snapshot) {
	r := &a.speakerReader
	if previous.ID != a.meetSnap.ID || previous.State != a.meetSnap.State {
		r.generation++
		r.id = ""
		if previous.ID != a.meetSnap.ID {
			r.target = proto.App{}
		}
	}
	if !a.liveMeetingSpeakers() || a.meetSnap.State != meeting.StateRecording || !a.helperReady() || !a.hstatus.Has(proto.MethodMeetingWindow) || r.id != "" {
		return
	}
	r.id = a.meetSnap.ID
	a.readMeetingSpeakers(r.id, r.generation)
}

func (a *app) readMeetingSpeakers(id string, generation int) {
	r := &a.speakerReader
	valid := func() bool {
		return r.id == id && r.generation == generation && a.meetSnap.ID == id && a.meetSnap.State == meeting.StateRecording
	}
	if !valid() {
		return
	}
	observedAt := a.meetMgr.Current().Elapsed
	again := func(observation meeting.SpeakerObservation) {
		if !valid() {
			return
		}
		observation.At = &observedAt
		a.meetMgr.ObserveSpeakers(id, observation)
		a.later(meetingSpeakerPoll, func() { a.readMeetingSpeakers(id, generation) })
	}
	if !a.helperReady() || !a.hstatus.Has(proto.MethodMeetingWindow) {
		again(meeting.SpeakerObservation{})
		return
	}
	if r.target.PID == 0 {
		// Name matching starts from a supported foreground window, then
		// retains that target for this meeting. Acoustic IDs need no window.
		front := a.previewApp()
		if front.BundleID == "us.zoom.xos" || appctx.IsBrowser(front.BundleID, a.prefs.AppEnhancement.Browsers) {
			r.target = front
		}
	}
	target := r.target
	if target.PID == 0 {
		again(meeting.SpeakerObservation{})
		return
	}
	read := func(platform string) {
		helperCall(a, proto.MethodMeetingWindow, proto.MeetingWindowParams{PID: target.PID, Screenshot: true}, 6*time.Second, func(result proto.MeetingWindowResult, err error) {
			if err != nil {
				again(meeting.SpeakerObservation{})
				return
			}
			again(meeting.ScreenSpeakers(result, platform))
		})
	}
	if target.BundleID == "us.zoom.xos" {
		read("zoom")
		return
	}
	helperCall(a, proto.MethodBrowserURL, proto.BrowserURLParams{PID: target.PID}, time.Second, func(result proto.BrowserURLResult, err error) {
		if !valid() {
			return
		}
		page, parseErr := url.Parse(result.URL)
		if err != nil || parseErr != nil || !strings.EqualFold(page.Hostname(), "meet.google.com") {
			again(meeting.SpeakerObservation{})
			return
		}
		read("meet")
	})
}
