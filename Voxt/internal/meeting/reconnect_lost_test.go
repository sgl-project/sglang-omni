package meeting_test

import (
	"context"
	"errors"
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
