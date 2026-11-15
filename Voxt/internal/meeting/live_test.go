package meeting

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

type liveFakeStreamer struct {
	mu      sync.Mutex
	streams []*liveFakeStream
}

func (f *liveFakeStreamer) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (f *liveFakeStreamer) Open(context.Context, asr.StreamRequest) (asr.Stream, error) {
	s := &liveFakeStream{events: make(chan asr.Event, 8)}
	f.mu.Lock()
	f.streams = append(f.streams, s)
	f.mu.Unlock()
	return s, nil
}

type liveFakeStream struct {
	mu     sync.Mutex
	data   []byte
	closed bool
	done   bool
	events chan asr.Event
}

func (s *liveFakeStream) Send(pcm []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.done {
		return &asr.Error{Kind: asr.KindProtocol, Detail: "ended"}
	}
	s.data = append(s.data, pcm...)
	select {
	case s.events <- asr.Event{Kind: asr.EventPartial, Text: "line"}:
	default:
	}
	return nil
}

func (s *liveFakeStream) Finish() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		return nil
	}
	s.done = true
	s.events <- asr.Event{Kind: asr.EventFinal, Text: "line", Result: asr.Result{Text: "line"}}
	close(s.events)
	return nil
}

func (s *liveFakeStream) Close() {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
}

func (s *liveFakeStream) Events() <-chan asr.Event { return s.events }

type tickingClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *tickingClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *tickingClock) advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

func speechPCM(ms time.Duration, rate int) []byte {
	n := int(ms.Seconds() * float64(rate))
	out := make([]byte, n*2)
	for i := 0; i < n; i++ {
		if i%80 < 40 {
			out[i*2], out[i*2+1] = 0x40, 0
		}
	}
	return out
}

func silencePCM(ms time.Duration, rate int) []byte {
	n := int(ms.Seconds() * float64(rate))
	return make([]byte, n*2)
}

func TestLiveStreamRotateAtTenMinutes(t *testing.T) {
	fs := &liveFakeStreamer{}
	clk := &tickingClock{t: time.Unix(0, 0)}
	var live Live
	live.Clock = clk.Now
	live.Streamer = fs
	live.Req = asr.StreamRequest{SampleRate: 16000}
	live.Speech = NewSpeech(16000, nil)
	ctx := context.Background()
	if err := live.Open(ctx); err != nil {
		t.Fatal(err)
	}
	rate := 16000
	frame := speechPCM(time.Second, rate)
	for i := 0; i < 600; i++ {
		clk.advance(time.Second)
		if err := live.Push(ctx, frame); err != nil {
			t.Fatal(err)
		}
	}
	// One more phrase then silence to commit; rotation runs at the commit point.
	clk.advance(800 * time.Millisecond)
	if err := live.Push(ctx, silencePCM(800*time.Millisecond, rate)); err != nil {
		t.Fatal(err)
	}
	fs.mu.Lock()
	n := len(fs.streams)
	first := append([]byte(nil), fs.streams[0].data...)
	var rest []byte
	for i := 1; i < n; i++ {
		rest = append(rest, fs.streams[i].data...)
	}
	fs.mu.Unlock()
	if n < 2 {
		t.Fatalf("want stream rotation at 10m commit, streams=%d", n)
	}
	if len(first) == 0 || len(rest) == 0 {
		t.Fatal("expected audio on both streams")
	}
}

func TestLiveStreamRotateStalePartial(t *testing.T) {
	clk := &tickingClock{t: time.Unix(0, 0)}
	var live Live
	live.Clock = clk.Now
	live.streamOpen = clk.Now()
	if live.stalePartial(true) {
		t.Fatal("too early")
	}
	clk.advance(stalePartialLive + time.Second)
	live.gotPartial = false
	if !live.stalePartial(true) {
		t.Fatal("expected stale partial while speaking")
	}
}

func TestLiveStreamContinuityOnRotate(t *testing.T) {
	fs := &liveFakeStreamer{}
	clk := &tickingClock{t: time.Unix(0, 0)}
	var live Live
	live.Clock = clk.Now
	live.Streamer = fs
	live.Req = asr.StreamRequest{SampleRate: 16000}
	live.Speech = NewSpeech(16000, nil)
	ctx := context.Background()
	if err := live.Open(ctx); err != nil {
		t.Fatal(err)
	}
	rate := 16000
	var sent []byte
	frame := speechPCM(50*time.Millisecond, rate)
	for i := 0; i < 20; i++ {
		clk.advance(50 * time.Millisecond)
		sent = append(sent, frame...)
		if err := live.Push(ctx, frame); err != nil {
			t.Fatal(err)
		}
	}
	if err := live.rotate(ctx, false); err != nil {
		t.Fatal(err)
	}
	clk.advance(50 * time.Millisecond)
	sent = append(sent, frame...)
	if err := live.Push(ctx, frame); err != nil {
		t.Fatal(err)
	}
	fs.mu.Lock()
	var got []byte
	for _, s := range fs.streams {
		got = append(got, s.data...)
	}
	fs.mu.Unlock()
	if len(got) != len(sent) {
		t.Fatalf("audio bytes: sent %d got %d", len(sent), len(got))
	}
}

// TestLiveRotateStartsFreshText: a new stream's text is all new, even when it
// starts like the old stream's committed text.
func TestLiveRotateStartsFreshText(t *testing.T) {
	old := &liveFakeStream{events: make(chan asr.Event), done: true}
	close(old.events)
	live := Live{Clock: time.Now, Streamer: &liveFakeStreamer{}, stream: old}
	live.committer.Partial("今天下午开会")
	if _, ok := live.committer.Tick(time.Second, silenceFinalizeLive); !ok {
		t.Fatal("expected commit")
	}
	if err := live.rotate(context.Background(), false); err != nil {
		t.Fatal(err)
	}
	live.committer.Partial("今天晚上")
	if g := live.LiveText(); g != "今天晚上" {
		t.Fatalf("live after rotate %q", g)
	}
}
