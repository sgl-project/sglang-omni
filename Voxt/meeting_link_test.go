package main

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// The meeting manager is made before the helper starts, so whether the
// helper can capture system audio must be asked at each start, not then.
func TestSystemCaptureOKFollowsHelper(t *testing.T) {
	a, fs, _ := newTestApp(t)
	if !a.systemCaptureOK() {
		t.Fatal("helper with capture.system: not OK")
	}
	fs.Stop()
	if a.systemCaptureOK() {
		t.Fatal("stopped helper: OK")
	}
	if err := fs.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !a.systemCaptureOK() {
		t.Fatal("restarted helper: not OK")
	}
}

func TestMeetingStartCaptureFailureNotice(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mode   string
		method string
		err    error
		want   string
	}{
		{
			name: "system output unavailable", mode: "subtitles", method: proto.MethodCaptureSystem,
			err:  proto.Errorf(proto.CodeDeviceUnavailable, "No output device to record from"),
			want: "Could not record system audio. Try again, or record the microphone only.",
		},
		{
			name: "system capture timed out", mode: "subtitles", method: proto.MethodCaptureSystem,
			err:  proto.Errorf(proto.CodeTimeout, "Capture did not start"),
			want: "Could not record system audio. Try again, or record the microphone only.",
		},
		{
			name: "microphone unavailable", mode: "recording", method: proto.MethodCaptureStart,
			err:  proto.Errorf(proto.CodeDeviceUnavailable, "No input gave audio"),
			want: "Could not record the microphone. Check your input device and try again.",
		},
		{
			name: "microphone denied", mode: "recording", method: proto.MethodCaptureStart,
			err:  &proto.Error{Code: proto.CodePermissionDenied, Data: json.RawMessage(`{"permission":"microphone"}`)},
			want: "Could not record the microphone. Allow Voxt in System Settings → Privacy & Security → Microphone.",
		},
		{
			name: "system audio denied", mode: "subtitles", method: proto.MethodCaptureSystem,
			err:  &proto.Error{Code: proto.CodePermissionDenied, Data: json.RawMessage(`{"permission":"systemAudio"}`)},
			want: "Could not record system audio. Allow system audio recording for Voxt in System Settings → Privacy & Security → Screen & System Audio Recording, or record the microphone only.",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, fs, _ := newTestApp(t)
			fs.handlers[tc.method] = func(json.RawMessage) (any, error) { return nil, tc.err }
			a.meetMgr = meeting.NewManager(meeting.Config{
				Mic:    session.HelperAudio{H: fs},
				System: session.HelperAudio{H: fs, Method: proto.MethodCaptureSystem},
				Store:  a.lib.store.Meetings(),
				Dir:    t.TempDir(),
			})
			a.prefs.Features.Meeting.Mode = tc.mode

			a.startMeeting()

			if len(a.toasts) != 1 || a.toasts[0] != tc.want {
				t.Fatalf("start notices = %q, want %q", a.toasts, tc.want)
			}
		})
	}
}
