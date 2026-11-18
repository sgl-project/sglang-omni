package helper

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// recordingWAV writes n samples of a ramp (step apart) as a 16 kHz WAV and
// returns its path and its PCM.
func recordingWAV(t *testing.T, name string, n int, step int16) (string, []byte) {
	t.Helper()
	pcm := make([]byte, 2*n)
	for i := range n {
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(int16(i)*step))
	}
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, helpertest.WAV(pcm, 16000), 0o644); err != nil {
		t.Fatal(err)
	}
	return path, pcm
}

// played checks that a job's audio is the recording byte for byte, then
// silence.
func played(t *testing.T, what string, got, want []byte) {
	t.Helper()
	if len(got) <= len(want) || !bytes.Equal(got[:len(want)], want) {
		t.Errorf("%s: %d bytes that are not the recording's %d", what, len(got), len(want))
		return
	}
	if tail := got[len(want):]; !bytes.Equal(tail, make([]byte, len(tail))) {
		t.Errorf("%s: not silence after the recording", what)
	}
}

// TestCaptureSystemBesideMicrophone: a capture.system job records beside a
// capture.start job, each playing its own recording (system=, pcm=); while
// they run a second job of either kind is busy; stopping the system's
// recording leaves the microphone's running and frees its place.
func TestCaptureSystemBesideMicrophone(t *testing.T) {
	micPath, micPCM := recordingWAV(t, "mic.wav", 8000, 3)
	sysPath, sysPCM := recordingWAV(t, "system.wav", 8000, -7)
	h := start(t, "frame-ms=5,"+helpertest.Opt("pcm", micPath)+","+helpertest.Opt("system", sysPath))
	h.ready(t)
	ctx := context.Background()
	mic := startCapture(ctx, h, "M1", proto.CaptureParams{})
	sys := startRecording(ctx, h, proto.MethodCaptureSystem, "Y1", proto.SystemParams{})
	wait(t, "both recordings", func() bool { return mic.frames() >= 7 && sys.frames() >= 7 })
	if got := slices.Sorted(slices.Values(h.sup.Status().Running)); !slices.Equal(got, []string{"M1", "Y1"}) {
		t.Errorf("running %v", got)
	}
	busy := &proto.Error{Code: proto.CodeBusy}
	if _, err := request[proto.CaptureResult](t, h, proto.MethodCaptureSystem, proto.SystemParams{}); !errors.Is(err, busy) {
		t.Errorf("second capture.system: %v", err)
	}
	if _, err := request[proto.CaptureResult](t, h, proto.MethodCaptureStart, proto.CaptureParams{}); !errors.Is(err, busy) {
		t.Errorf("second capture.start: %v", err)
	}

	if stop, err := request[proto.CaptureStopResult](t, h, proto.MethodCaptureStop, proto.CaptureStopParams{Job: "Y1"}); err != nil || !stop.Stopped {
		t.Fatalf("stop Y1: %+v %v", stop, err)
	}
	<-sys.done
	if sys.err != nil || sys.res.Reason != proto.CaptureStopped || sys.res.Samples*2 != len(sys.pcm) {
		t.Fatalf("system: %+v %v (%d bytes)", sys.res, sys.err, len(sys.pcm))
	}
	if s := sys.started; s == nil || s.Device == "" || s.Device != sys.res.Device || s.DeviceName != "" || s.FellBack ||
		s.DeviceChannels != 1 || s.SampleRate != 16000 || slices.ContainsFunc(helpertest.Devices(), func(d proto.CaptureDevice) bool { return d.ID == s.Device }) {
		t.Errorf("system started %+v", s)
	}
	played(t, "system", sys.pcm, sysPCM)

	// The microphone records on, and the system's place is free again.
	n := mic.frames()
	wait(t, "the microphone after the stop", func() bool { return mic.frames() >= n+3 })
	again := startRecording(ctx, h, proto.MethodCaptureSystem, "Y2", proto.SystemParams{})
	wait(t, "a new system recording", func() bool { return again.frames() >= 1 })
	for _, job := range []string{"Y2", "M1"} {
		request[proto.CaptureStopResult](t, h, proto.MethodCaptureStop, proto.CaptureStopParams{Job: job})
	}
	<-again.done
	<-mic.done
	if again.err != nil || mic.err != nil || mic.res.Reason != proto.CaptureStopped {
		t.Fatalf("Y2 %v; microphone %+v %v", again.err, mic.res, mic.err)
	}
	played(t, "microphone", mic.pcm, micPCM)
}

// TestCaptureSystemWithoutMicrophone: the system's sound is not the
// microphone: capture.system records while the microphone is denied.
func TestCaptureSystemWithoutMicrophone(t *testing.T) {
	h := start(t, "deny-mic,frame-ms=5")
	h.ready(t)
	c := startRecording(context.Background(), h, proto.MethodCaptureSystem, "Y3", proto.SystemParams{FrameMS: 50})
	wait(t, "a frame", func() bool { return c.frames() >= 2 })
	request[proto.CaptureStopResult](t, h, proto.MethodCaptureStop, proto.CaptureStopParams{Job: "Y3"})
	<-c.done
	if c.err != nil || c.res.Reason != proto.CaptureStopped || len(c.pcm) != c.res.Frames*800*2 {
		t.Errorf("system without the microphone: %+v %v (%d bytes)", c.res, c.err, len(c.pcm))
	}
}

// TestCaptureSystemOldHelper: an older helper, without capture.system in
// hello's methods (no-system), is told apart by Status.Has and does not
// know the method.
func TestCaptureSystemOldHelper(t *testing.T) {
	if st := start(t, "").ready(t); !st.Has(proto.MethodCaptureSystem) {
		t.Errorf("the helper does not list capture.system: %v", st.Hello.Methods)
	}
	h := start(t, "no-system")
	st := h.ready(t)
	if st.Has(proto.MethodCaptureSystem) || !st.Has(proto.MethodCaptureStart) {
		t.Errorf("an older helper lists %v", st.Hello.Methods)
	}
	if _, err := request[proto.CaptureResult](t, h, proto.MethodCaptureSystem, proto.SystemParams{}); !errors.Is(err, &proto.Error{Code: proto.CodeMethodNotFound}) {
		t.Errorf("capture.system on an older helper: %v", err)
	}
}
