package meeting_test

import (
	"context"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock(start time.Time) *fakeClock {
	return &fakeClock{t: start}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

type fakeAudio struct {
	startErr   error
	pcm        []byte
	levels     []float64 // dBFS levels sent after the audio
	levelsSent chan struct{}
	mu         sync.Mutex
	stops      map[string]chan struct{}
	jobs       []string
}

func newFakeAudio(pcm []byte) *fakeAudio {
	return &fakeAudio{pcm: pcm, stops: map[string]chan struct{}{}}
}

func (f *fakeAudio) jobIDs() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]string, len(f.jobs))
	copy(out, f.jobs)
	return out
}

func (f *fakeAudio) Record(ctx context.Context, id string, _ proto.CaptureParams, fn func(session.AudioEvent)) (proto.CaptureResult, error) {
	if f.startErr != nil {
		return proto.CaptureResult{}, f.startErr
	}
	stop := make(chan struct{})
	f.mu.Lock()
	f.stops[id] = stop
	f.jobs = append(f.jobs, id)
	f.mu.Unlock()
	fn(session.AudioEvent{Started: &proto.CaptureStartedParams{Job: id, SampleRate: asr.SampleRate}})
	chunk := f.pcm
	if len(chunk) == 0 {
		chunk = make([]byte, asr.BytesPerSecond/10)
	}
	fn(session.AudioEvent{PCM: &proto.PCMParams{Job: id, Data: chunk}})
	for _, db := range f.levels {
		fn(session.AudioEvent{Level: &proto.LevelParams{Job: id, RMS: db}})
	}
	if f.levelsSent != nil {
		close(f.levelsSent)
	}
	select {
	case <-ctx.Done():
		return proto.CaptureResult{}, ctx.Err()
	case <-stop:
		return proto.CaptureResult{Reason: proto.CaptureStopped}, nil
	}
}

func (f *fakeAudio) Stop(ctx context.Context, id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if ch, ok := f.stops[id]; ok {
		close(ch)
		delete(f.stops, id)
	}
	return nil
}
