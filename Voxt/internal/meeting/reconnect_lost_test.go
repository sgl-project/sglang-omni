package meeting_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// droppingAudio loses its first capture on its own, the way a
// microphone unplugged or lost on wake from sleep does.
type droppingAudio struct {
	*fakeAudio
	dropped     chan struct{}
	reconnected chan struct{}
}

func (a *droppingAudio) Record(ctx context.Context, id string, params proto.CaptureParams, fn func(session.AudioEvent)) (proto.CaptureResult, error) {
	if len(a.jobIDs()) > 0 {
		if len(a.jobIDs()) == 1 {
			close(a.reconnected)
		}
		return a.fakeAudio.Record(ctx, id, params, fn)
	}
	a.fakeAudio.mu.Lock()
	a.fakeAudio.jobs = append(a.fakeAudio.jobs, id)
	a.fakeAudio.mu.Unlock()
	fn(session.AudioEvent{Started: &proto.CaptureStartedParams{Job: id, SampleRate: asr.SampleRate}})
	fn(session.AudioEvent{PCM: &proto.PCMParams{Job: id, Data: []byte{9}}})
	<-a.dropped
	return proto.CaptureResult{}, errors.New("device lost")
}

// A microphone lost while its streaming ASR connection hangs still
// reconnects: the hung Final must not leave the meeting recording nothing.
func TestLostMicReconnectsWhileFinalHangs(t *testing.T) {
	t.Cleanup(meeting.SetLostFinalWaitForTest(50 * time.Millisecond))
	mic := &droppingAudio{fakeAudio: newFakeAudio([]byte{9}), dropped: make(chan struct{}), reconnected: make(chan struct{})}
	streamer := newFinishGateStreamer() // the first stream's Final never returns until cleanup
	m, ctx := testManagerStreams(t, mic, newFakeAudio([]byte{8}), streamer, &meetTestStreamer{})
	t.Cleanup(streamer.release)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Finish(ctx, id)
	close(mic.dropped)
	select {
	case <-streamer.finishStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("the lost track did not reach its Final")
	}
	select {
	case <-mic.reconnected:
	case <-time.After(3 * time.Second):
		t.Fatal("the lost microphone did not reconnect behind its hung Final")
	}
}

// flakyAudio loses its first capture, fails to start the next, then
// records, keeping that last start's params.
type flakyAudio struct {
	*fakeAudio
	starts      atomic.Int32
	dropped     chan struct{} // closed by the test: the first capture is lost
	reconnected chan struct{} // closed at the third start, after last is set
	last        proto.CaptureParams
}

func (a *flakyAudio) Record(ctx context.Context, id string, params proto.CaptureParams, fn func(session.AudioEvent)) (proto.CaptureResult, error) {
	switch a.starts.Add(1) {
	case 1:
		fn(session.AudioEvent{Started: &proto.CaptureStartedParams{Job: id, SampleRate: asr.SampleRate}})
		<-a.dropped
		return proto.CaptureResult{}, errors.New("device lost")
	case 2:
		return proto.CaptureResult{}, errors.New("no input could start")
	case 3:
		a.last = params
		close(a.reconnected)
	}
	return a.fakeAudio.Record(ctx, id, params, fn)
}

// A mode chosen while a lost track waits to reconnect is the one it
// reconnects in: a microphone back in Recording mode records without
// echo cancellation, which only a meeting with the system's sound asks for.
func TestReconnectAfterSetModeUsesTheNewMode(t *testing.T) {
	waiting, release := make(chan struct{}), make(chan time.Time)
	var once sync.Once
	t.Cleanup(meeting.SetReconnectAfterForTest(func(time.Duration) <-chan time.Time {
		once.Do(func() { close(waiting) })
		return release
	}))
	mic := &flakyAudio{fakeAudio: newFakeAudio([]byte{9}), dropped: make(chan struct{}), reconnected: make(chan struct{})}
	m, ctx := testManagerStreams(t, mic, newFakeAudio([]byte{8}), &meetTestStreamer{}, &meetTestStreamer{})
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Finish(ctx, id)
	close(mic.dropped)
	select {
	case <-waiting:
	case <-time.After(3 * time.Second):
		t.Fatal("the lost microphone did not wait to reconnect")
	}
	if err := m.SetMode(ctx, id, meeting.ModeRecording); err != nil {
		t.Fatal(err)
	}
	close(release)
	select {
	case <-mic.reconnected:
	case <-time.After(3 * time.Second):
		t.Fatal("the microphone did not reconnect")
	}
	if mic.last.EchoCancel {
		t.Error("the microphone reconnected in Recording mode with echoCancel: the mode from before the wait")
	}
}
