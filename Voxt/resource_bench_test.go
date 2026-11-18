package main

import (
	"encoding/json"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

func logW0ResourceSample(t *testing.T, phase string) {
	t.Helper()
	goroutines := runtime.NumGoroutine()
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	memory, err := macx.Memory(os.Getpid())
	if err != nil {
		t.Fatalf("read process memory: %v", err)
	}
	t.Logf("MEASURE resource phase=%s resident_bytes=%d heap_alloc_bytes=%d goroutines=%d", phase, memory.Resident, stats.HeapAlloc, goroutines)
}

func installW0Hotkeys(tb testing.TB, a *app, fake *fakeSupervisor) {
	tb.Helper()
	// The render helper omits hotkey.configure; answer like the fake helper so meeting updates do not retry a failing call.
	fake.handlers[proto.MethodHotkeyConfigure] = func(raw json.RawMessage) (any, error) {
		var params proto.HotkeyConfigureParams
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, err
		}
		installed := len(params.Bindings) > 0 || params.Cancel != nil
		return proto.HotkeyStatus{Installed: installed, Accessibility: true, Bindings: len(params.Bindings)}, nil
	}
	a.syncHotkeys()
	if !a.hotkeys.Installed {
		tb.Fatal("the fake hotkey tap is not installed")
	}
}

func TestW0ResourceBaseline(t *testing.T) {
	if os.Getenv("VOXTNEW_W0_RESOURCES") != "1" {
		t.Skip("set VOXTNEW_W0_RESOURCES=1 to measure resources")
	}
	if runtime.GOOS != "darwin" {
		t.Skip("resident memory measurement requires macOS")
	}

	t.Run("startup", func(t *testing.T) {
		started := time.Now()
		a, fake, tester := newRenderBenchmarkApp(t)
		firstFrame := time.Since(started)
		if !tester.HasText("Today") {
			t.Fatalf("the first frame is not settled: %q", tester.Texts())
		}
		t.Logf("MEASURE resource phase=startup_to_first_frame duration_ms=%.3f", float64(firstFrame)/float64(time.Millisecond))

		installW0Hotkeys(t, a, fake)
		hotkeyReady := time.Since(started)
		if !a.hotkeys.Installed {
			t.Fatal("startup finished without a usable hotkey tap")
		}
		t.Logf("MEASURE resource phase=startup_to_hotkey_ready duration_ms=%.3f", float64(hotkeyReady)/float64(time.Millisecond))
	})

	t.Run("session", func(t *testing.T) {
		set, services, _ := benchmarkFileSetup(t, "resource baseline", &fakeChat{text: "Resource baseline"})
		events := benchmarkScript(t,
			hotkeyLine(1000, "transcribe/0", proto.PhaseTrigger),
			hotkeyLine(2000, "transcribe/0", proto.PhaseTrigger),
		)
		a := newBenchmarkSessionApp(t, events, set, services)
		a.pump("the hotkey tap", func() bool { return a.hotkeys.Installed })
		if a.sess.State != session.Idle {
			t.Fatalf("idle sample requested during session state %s", a.sess.State)
		}
		logW0ResourceSample(t, "idle")
		idleGoroutines := runtime.NumGoroutine()

		a.pump("the recording", func() bool { return a.sess.State == session.Recording })
		if a.sess.State != session.Recording {
			t.Fatalf("recording sample requested in state %s", a.sess.State)
		}
		logW0ResourceSample(t, "recording")

		a.pump("the session end", func() bool { return a.sess.State.Ended() })
		if a.sess.State != session.Done {
			t.Fatalf("session ended in state %s with problem %+v", a.sess.State, a.sess.Problem)
		}
		deadline := time.After(2 * time.Second)
		for runtime.NumGoroutine() > idleGoroutines {
			select {
			case fn := <-a.queue:
				fn()
			case <-deadline:
				t.Fatalf("%d goroutines remain after the session, want %d", runtime.NumGoroutine(), idleGoroutines)
			default:
				runtime.Gosched()
			}
		}
	drain:
		for {
			select {
			case fn := <-a.queue:
				fn()
			default:
				break drain
			}
		}
		logW0ResourceSample(t, "after_session")
	})

	t.Run("meeting_1h_synthetic", func(t *testing.T) {
		a, fake, _ := newRenderBenchmarkApp(t)
		installW0Hotkeys(t, a, fake)
		snapshot := meeting.Snapshot{ID: "resource-hour", Mode: meeting.ModeMeeting, State: meeting.StateRecording}
		a.meetSnap = snapshot
		model := a.meetingOverlayModel()
		tester := hudTester(a.meetingOverlayBenchmarkView, model.Layout.Width, meetingOvMaxFaceH+2*layout.Pad)
		frames(tester)

		// No real one-hour meeting or model runs; two synthetic tracks commit at the production 8-second cap.
		segments := make([]meeting.Segment, meetingUpdatesPerHour)
		for index := range segments {
			trackIndex := index / 2
			source, speaker := meeting.SourceMic, "me"
			if index%2 != 0 {
				source, speaker = meeting.SourceSystem, "s1"
			}
			start := time.Duration(trackIndex) * meetingCommitInterval
			segments[index] = meeting.Segment{
				ID: "segment-" + strconv.Itoa(index), Source: source, Speaker: speaker,
				Start: start, End: start + meetingCommitInterval,
				Text: "Resource segment " + strconv.Itoa(index) + ".", Final: true,
			}
		}
		for _, segment := range segments {
			snapshot.Segments = append(snapshot.Segments, segment)
			if len(snapshot.Segments) > meetingSnapshotSegmentCap {
				snapshot.Segments = snapshot.Segments[len(snapshot.Segments)-meetingSnapshotSegmentCap:]
			}
			snapshot.Elapsed = segment.End
			a.meetingChanged(snapshot)
			tester.Frame()
		}
		if len(a.meetSnap.Segments) != meetingSnapshotSegmentCap {
			t.Fatalf("snapshot retains %d segments, want %d", len(a.meetSnap.Segments), meetingSnapshotSegmentCap)
		}
		if a.meetSnap.Elapsed != time.Hour {
			t.Fatalf("synthetic meeting elapsed time is %v, want %v", a.meetSnap.Elapsed, time.Hour)
		}
		newest := segments[len(segments)-1]
		if !tester.HasText(newest.Text) {
			t.Fatalf("newest segment %q is missing from the frame: %q", newest.Text, tester.Texts())
		}
		logW0ResourceSample(t, "meeting_1h_synthetic")
	})
}
