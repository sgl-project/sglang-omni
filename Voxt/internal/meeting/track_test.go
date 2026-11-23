package meeting_test

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// helperLine stands in for the helper's one stdout reader: it runs the
// capture callbacks, and it alone answers calls, as Apple Speech's
// speech.audio is a helper call.
type helperLine struct {
	calls  chan chan struct{}
	chunks int
	mu     sync.Mutex
	stops  map[string]chan struct{}
}

func (h *helperLine) Record(ctx context.Context, id string, _ proto.CaptureParams, fn func(session.AudioEvent)) (proto.CaptureResult, error) {
	stop := make(chan struct{})
	h.mu.Lock()
	h.stops[id] = stop
	h.mu.Unlock()
	fn(session.AudioEvent{Started: &proto.CaptureStartedParams{Job: id, SampleRate: asr.SampleRate}})
	for i := 0; ; i++ {
		if i < h.chunks {
			fn(session.AudioEvent{PCM: &proto.PCMParams{Job: id, Data: make([]byte, 3200)}})
		}
		select {
		case c := <-h.calls:
			close(c)
		case <-stop:
			return proto.CaptureResult{Reason: proto.CaptureStopped}, nil
		case <-ctx.Done():
			return proto.CaptureResult{}, ctx.Err()
		case <-time.After(time.Millisecond):
		}
	}
}

func (h *helperLine) Stop(_ context.Context, id string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if ch, ok := h.stops[id]; ok {
		close(ch)
		delete(h.stops, id)
	}
	return nil
}

type helperStream struct {
	line   *helperLine
	mu     sync.Mutex
	bytes  int
	stuck  bool
	events chan asr.Event
}

func (s *helperStream) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (s *helperStream) Open(context.Context, asr.StreamRequest) (asr.Stream, error) { return s, nil }

func (s *helperStream) Send(pcm []byte) error {
	c := make(chan struct{})
	s.line.calls <- c
	select {
	case <-c:
	case <-time.After(time.Second):
		s.mu.Lock()
		s.stuck = true
		s.mu.Unlock()
		return errors.New("no answer: the reader is busy in this callback")
	}
	s.mu.Lock()
	s.bytes += len(pcm)
	s.mu.Unlock()
	return nil
}

func (s *helperStream) Finish() error {
	close(s.events)
	return nil
}

func (s *helperStream) Close() {}

func (s *helperStream) Events() <-chan asr.Event { return s.events }

func TestTrackPCMNeverWaitsOnTheHelperReader(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	line := &helperLine{calls: make(chan chan struct{}, 64), chunks: 5, stops: map[string]chan struct{}{}}
	stream := &helperStream{line: line, events: make(chan asr.Event)}
	m := meeting.NewManager(meeting.Config{
		Mic:     line,
		Store:   s.Meetings(),
		Dir:     filepath.Join(dir, "meetings"),
		Privacy: func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) {
			return meeting.DiskMinStartBytes() + 1, nil
		},
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: stream.Caps(), Streamer: stream}, nil
		},
	})
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	stream.mu.Lock()
	defer stream.mu.Unlock()
	if stream.stuck {
		t.Fatal("a PCM callback waited on a helper call the reader could not answer")
	}
	if stream.bytes != 5*3200 {
		t.Fatalf("the stream got %d bytes, want all %d", stream.bytes, 5*3200)
	}
}

// hangingStart never reports the capture started, like a helper that
// does not answer.
type hangingStart struct{}

func (hangingStart) Record(ctx context.Context, _ string, _ proto.CaptureParams, _ func(session.AudioEvent)) (proto.CaptureResult, error) {
	<-ctx.Done()
	return proto.CaptureResult{}, ctx.Err()
}

func (hangingStart) Stop(context.Context, string) error { return nil }

func TestFinishCancelsAHangingStart(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	m := meeting.NewManager(meeting.Config{
		Mic:     hangingStart{},
		Store:   s.Meetings(),
		Dir:     filepath.Join(dir, "meetings"),
		Privacy: func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) {
			return meeting.DiskMinStartBytes() + 1, nil
		},
	})
	errc := make(chan error, 1)
	go func() {
		_, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
		errc <- err
	}()
	deadline := time.Now().Add(time.Second)
	for m.Current().State != meeting.StateStarting && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if err := m.Finish(ctx, m.Current().ID); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-errc:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Start: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Start still hangs after Finish")
	}
	if st := m.Current().State; st != meeting.StateIdle {
		t.Fatalf("state %q after a cancelled start", st)
	}
	if list, err := s.Meetings().ListMeetings(ctx, store.MeetingQuery{Limit: 5}); err != nil || len(list) != 0 {
		t.Fatalf("a cancelled start left %d meetings (%v)", len(list), err)
	}
}
