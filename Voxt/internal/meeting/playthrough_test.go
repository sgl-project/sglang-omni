package meeting_test

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// playedAudio is a microphone that plays a recording in 100 ms chunks, as
// the helper sends them, 50 times faster than it was spoken, and then
// stays quiet until it is stopped.
type playedAudio struct {
	pcm   []byte
	mu    sync.Mutex
	stops map[string]chan struct{}
}

func (p *playedAudio) Record(ctx context.Context, id string, _ proto.CaptureParams, fn func(session.AudioEvent)) (proto.CaptureResult, error) {
	stop := make(chan struct{})
	p.mu.Lock()
	p.stops[id] = stop
	p.mu.Unlock()
	fn(session.AudioEvent{Started: &proto.CaptureStartedParams{Job: id, SampleRate: asr.SampleRate}})
	const chunk = asr.BytesPerSecond / 10
	for off := 0; off < len(p.pcm); off += chunk {
		fn(session.AudioEvent{PCM: &proto.PCMParams{Job: id, Data: p.pcm[off:min(off+chunk, len(p.pcm))]}})
		// The model's text for a chunk reaches the captions before the next chunk, as it does live.
		time.Sleep(2 * time.Millisecond)
	}
	select {
	case <-ctx.Done():
		return proto.CaptureResult{}, ctx.Err()
	case <-stop:
		return proto.CaptureResult{Reason: proto.CaptureStopped}, nil
	}
}

func (p *playedAudio) Stop(_ context.Context, id string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if ch, ok := p.stops[id]; ok {
		close(ch)
		delete(p.stops, id)
	}
	return nil
}

// spokenStreamer is a streaming speech model that hears the recording's
// sentences: sentence i is said from i*every on, and a stream's text is
// every sentence said since that stream opened.
type spokenStreamer struct {
	sentences []string
	every     int // bytes of audio from one sentence's start to the next one's
	mu        sync.Mutex
	heard     int // bytes of audio all streams got
}

func (s *spokenStreamer) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (s *spokenStreamer) Open(context.Context, asr.StreamRequest) (asr.Stream, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return &spokenStream{model: s, from: s.heard, events: make(chan asr.Event, 256)}, nil
}

type spokenStream struct {
	model  *spokenStreamer
	from   int
	events chan asr.Event
}

func (s *spokenStream) text() string {
	var said []string
	for i, sentence := range s.model.sentences {
		if start := i * s.model.every; start >= s.from && start < s.model.heard {
			said = append(said, sentence)
		}
	}
	return strings.Join(said, " ")
}

func (s *spokenStream) Send(pcm []byte) error {
	s.model.mu.Lock()
	s.model.heard += len(pcm)
	text := s.text()
	s.model.mu.Unlock()
	select {
	case s.events <- asr.Event{Kind: asr.EventPartial, Text: text}:
	default:
	}
	return nil
}

func (s *spokenStream) Finish() error {
	s.model.mu.Lock()
	text := s.text()
	s.model.mu.Unlock()
	s.events <- asr.Event{Kind: asr.EventFinal, Text: text, Result: asr.Result{Text: text}}
	close(s.events)
	return nil
}

func (s *spokenStream) Close() {}

func (s *spokenStream) Events() <-chan asr.Event { return s.events }

// TestMeetingPlaythrough plays a recording of three sentences, each
// followed by a pause, through a whole meeting: start, live captions,
// finish, the saved meeting. The captions commit in the order said,
// each sentence once, and the saved transcript is those lines.
func TestMeetingPlaythrough(t *testing.T) {
	sentences := []string{"First we review the release plan.", "Then we assign the open bugs.", "Last we pick the next meeting date."}
	// 1.5 s of speech and a 1.9 s pause: the speech detector holds on after
	// the speech ends, and what is left of the pause must still pass 0.75 s.
	said := append(liveTestPCM(t), make([]byte, asr.BytesPerSecond)...)
	var pcm []byte
	for range sentences {
		pcm = append(pcm, said...)
	}
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	streamer := &spokenStreamer{sentences: sentences, every: len(said)}
	m := meeting.NewManager(meeting.Config{
		Mic:       &playedAudio{pcm: pcm, stops: map[string]chan struct{}{}},
		Store:     s.Meetings(),
		Dir:       filepath.Join(dir, "meetings"),
		Privacy:   func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) { return meeting.DiskMinStartBytes() + 1, nil },
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: streamer.Caps(), Streamer: streamer}, nil
		},
	})
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	texts := func(segs []meeting.Segment) []string {
		var out []string
		for _, g := range segs {
			out = append(out, g.Text)
		}
		return out
	}
	for deadline := time.Now().Add(10 * time.Second); len(m.Current().Segments) < len(sentences); time.Sleep(5 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("live captions %q, want %q", texts(m.Current().Segments), sentences)
		}
	}
	live := m.Current().Segments
	if got := texts(live); !slices.Equal(got, sentences) {
		t.Fatalf("live captions %q, want %q", got, sentences)
	}
	for i := 1; i < len(live); i++ {
		if live[i].Start < live[i-1].End {
			t.Errorf("caption %d starts at %v, before caption %d ends at %v", i, live[i].Start, i-1, live[i-1].End)
		}
	}
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	_, stored, err := s.Meetings().Meeting(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	var saved []string
	for _, g := range stored {
		saved = append(saved, g.Text)
	}
	if !slices.Equal(saved, sentences) {
		t.Fatalf("saved transcript %q, want %q", saved, sentences)
	}
}
