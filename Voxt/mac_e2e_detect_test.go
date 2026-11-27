package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meetdetect"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// envSystemFile is the helper's test-only file input for capture.system.
const envSystemFile = "VOXTNEW_HELPER_SYSTEM_FILE"

// noTap is the real helper without its event tap: the app's shortcuts are
// refused, so no key the person at the Mac presses reaches the test.
type noTap struct{ helper.Supervisor }

func (n noTap) Call(ctx context.Context, id, method string, params any, notify func(proto.Message)) (json.RawMessage, error) {
	if method == proto.MethodHotkeyConfigure {
		return nil, proto.Errorf(proto.CodeMethodNotFound, "the test installs no event tap")
	}
	return n.Supervisor.Call(ctx, id, method, params, notify)
}

// newMacDetectApp is the whole app over the real helper at bin, which
// records the speech sample instead of the microphone and the system's
// sound (no permission, nothing played), with a speech model that hears
// a sentence. No window, no event tap, silent cues.
func newMacDetectApp(t *testing.T, bin string) *sessionApp {
	t.Helper()
	lib, err := openLibrary(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(lib.close)
	if err := lib.settings.Update(context.Background(), func(st *settings.Settings) {
		st.Features.Meeting.NoFinalPass, st.Features.Meeting.NoAutoSummary = true, true
	}); err != nil {
		t.Fatal(err)
	}
	old := newLocalServices
	newLocalServices = func(*app) local.Services { return hearsMeeting() }
	t.Cleanup(func() { newLocalServices = old })
	a := newApp(lib, &helperHost{})
	a.measureHook = guessText
	s := &sessionApp{app: a, t: t, queue: make(chan func(), 4096)}
	a.copyText = func(string) error { return nil }
	a.postQueue = func(fn func()) { s.queue <- fn }
	a.muteJournal = outputJournal(t.TempDir(), "")
	useTestSounds(t, a)
	a.playSound = func(sounds.Target) {}
	sample := speechSample()
	a.host.sup = noTap{helper.New(helper.Config{
		Bin: bin, AppVersion: "mac-e2e", Bus: a.bus, Log: a.logs.Append,
		Env:      []string{envCaptureFile + "=" + sample, envSystemFile + "=" + sample, envCaptureSpeed + "=4"},
		OnNotify: func(m proto.Message) { a.post(func() { a.helperNotified(m) }) },
	})}
	a.connectHelper()
	a.startHelper()
	t.Cleanup(func() {
		a.quit()
		for {
			select {
			case fn := <-s.queue:
				fn()
			case <-time.After(50 * time.Millisecond):
				return
			}
		}
	})
	s.pump("the helper", func() bool { return a.helperStatus().State == helper.Ready })
	shortHold(s)
	return s
}

// TestMacE2EMeetingDetect runs the detection in the whole app over the
// real helper: the app turns the helper's watch on and takes Core Audio's
// answer, a meeting app that records brings the card, and its button
// records a meeting through the helper's two captures.
func TestMacE2EMeetingDetect(t *testing.T) {
	s := newMacDetectApp(t, macE2E(t))
	s.pump("Core Audio's answer", func() bool { return s.micWatch && s.micUsers != nil })
	t.Logf("recording on this Mac now: %d process(es)", len(s.micUsers))
	// Zoom joins a meeting, as mic.changed says, beside whoever records.
	users := proto.MicUsers{Apps: []proto.App{{PID: 704, BundleID: "us.zoom.xos"}}}
	for _, u := range s.micUsers {
		users.Apps = append(users.Apps, proto.App{PID: u.PID, BundleID: u.BundleID})
	}
	s.micChanged(users)
	s.pump("the card", func() bool { return s.ask.view.Visible })
	if s.meetingActive() {
		t.Fatalf("the card %+v, meeting %s", s.ask.view, s.meetSnap.State)
	}
	s.startDetectedMeeting()
	s.pump("the meeting", func() bool { return s.meetSnap.State == meeting.StateRecording })
	id := s.meetSnap.ID
	s.pump("its live text", func() bool { return len(s.meetSnap.Live) > 0 || len(s.meetSnap.Segments) > 0 })
	if s.ask.view.Visible || len(s.toasts) != 0 {
		t.Fatalf("while Voxt records: card %+v, toasts %q", s.ask.view, s.toasts)
	}
	s.meetingFinish()
	// Ended on its first words, the meeting asks "Started by mistake?"
	// (fewer than meeting.MistakeWords): Keep files it.
	s.pump("its end", func() bool {
		return s.meetSnap.State != meeting.StateRecording && s.meetSnap.State != meeting.StateFinishing
	})
	if s.meetSnap.State == meeting.StateAsking && s.meetSnap.Mistake {
		s.meetingSave()
	}
	s.pump("its filing", func() bool { return !s.meetingActive() })
	list, err := s.lib.store.Meetings().ListMeetings(context.Background(), store.MeetingQuery{})
	if err != nil || len(list) != 1 || list[0].ID != id {
		t.Fatalf("meetings kept: %+v %v", list, err)
	}
	for _, name := range []string{"mic.wav", "system.wav"} {
		info, err := os.Stat(filepath.Join(s.dataDir(), "meetings", id, name))
		if err != nil || info.Size() <= 44 {
			t.Errorf("%s of the meeting: %v %v", name, info, err)
		} else {
			t.Logf("%s: %d bytes", name, info.Size())
		}
	}
}

// The person running this test prepares the real app's microphone and
// front tab (docs/manual-test.md, 15.3a). No mic.changed is fabricated.
func TestMacE2EMeetingDetectLive(t *testing.T) {
	scenario := os.Getenv("VOXTNEW_MEETING_CASE")
	if scenario == "" {
		t.Skip("prepare a real meeting or microphone test; see docs/manual-test.md, 15.3a")
	}
	var bundle, want string
	switch scenario {
	case "zoom-test":
		bundle = "us.zoom.xos"
	case "zoom-meeting":
		bundle, want = "us.zoom.xos", "Zoom"
	case "chrome-other":
		bundle = "com.google.Chrome"
	case "chrome-meet":
		bundle, want = "com.google.Chrome", "Google Meet"
	default:
		t.Fatalf("unknown meeting case %q", scenario)
	}
	s := newMacDetectApp(t, macE2E(t))
	s.detect.Hold = meetdetect.Hold
	s.pump("Core Audio's answer", func() bool { return s.micWatch && s.micUsers != nil })
	recording := false
	for _, user := range s.micUsers {
		if user.BundleID == bundle || strings.HasPrefix(user.BundleID, bundle+".") {
			recording = true
			t.Logf("real input: %s, meeting processes %v", user.BundleID, user.Hosts)
		}
	}
	if !recording {
		t.Fatalf("%s is not using the microphone: prepare %s first", bundle, scenario)
	}
	if want == "" {
		s.drain(2*meetdetect.Hold + 5*time.Second)
		if s.ask.view.Visible {
			t.Fatalf("%s brought a card: %+v", scenario, s.ask.view)
		}
	} else {
		started := time.Now()
		s.pump("the real meeting's card", func() bool { return s.ask.view.Visible })
		if s.ask.view.App != want {
			t.Fatalf("%s: card for %q, want %q", scenario, s.ask.view.App, want)
		}
		t.Logf("%s detected in %s after the initial mic snapshot", scenario, time.Since(started).Round(time.Millisecond))
	}
	if s.meetingActive() {
		t.Fatal("detecting the meeting started recording without a click")
	}
}
