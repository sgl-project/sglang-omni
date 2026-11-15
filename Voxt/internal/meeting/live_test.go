package meeting

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

type liveFakeStreamer struct {
	mu      sync.Mutex
	streams []*liveFakeStream
	partial string // partial text each Send reports; "" means "line"
}

func (f *liveFakeStreamer) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (f *liveFakeStreamer) Open(context.Context, asr.StreamRequest) (asr.Stream, error) {
	s := &liveFakeStream{events: make(chan asr.Event, 8), partial: f.partial}
	if s.partial == "" {
		s.partial = "line"
	}
	f.mu.Lock()
	f.streams = append(f.streams, s)
	f.mu.Unlock()
	return s, nil
}

type liveFakeStream struct {
	mu      sync.Mutex
	data    []byte
	closed  bool
	done    bool
	events  chan asr.Event
	partial string
}

func (s *liveFakeStream) Send(pcm []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.done {
		return &asr.Error{Kind: asr.KindProtocol, Detail: "ended"}
	}
	s.data = append(s.data, pcm...)
	select {
	case s.events <- asr.Event{Kind: asr.EventPartial, Text: s.partial}:
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
	if err := live.rotate(ctx, false, true); err != nil {
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
	if err := live.rotate(context.Background(), false, true); err != nil {
		t.Fatal(err)
	}
	live.committer.Partial("今天晚上")
	if g := live.LiveText(); g != "今天晚上" {
		t.Fatalf("live after rotate %q", g)
	}
}

// TestLiveRotateFinalDoesNotRepeat replays #3983: the old stream's Final
// recognises its whole audio again, worded differently from the preview,
// and rotating must not commit the already committed minutes again.
func TestLiveRotateFinalDoesNotRepeat(t *testing.T) {
	tests := []struct {
		name, final, want string
	}{
		{"reworded", "今天下午我们开会，先说 Voxt 的进度。然后讲会议功能。还有一点", "还有一点"},
		{"continues", "今天下午开会先说Voxt的进度然后讲会议功能还有一点吗", "还有一点吗"},
		{"empty", "", "还有一点"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			old := &liveFakeStream{events: make(chan asr.Event, 1), done: true}
			old.events <- asr.Event{Kind: asr.EventFinal, Text: tc.final}
			close(old.events)
			var got []string
			live := Live{Clock: time.Now, Streamer: &liveFakeStreamer{}, stream: old,
				OnCommit: func(s Segment) { got = append(got, s.Text) }}
			for i, p := range []string{"今天下午开会", "今天下午开会先说Voxt的进度", "今天下午开会先说Voxt的进度然后讲会议功能"} {
				live.committer.Partial(p)
				if _, ok := live.committer.Tick(time.Duration(i+1)*time.Second, silenceFinalizeLive); !ok {
					t.Fatalf("commit %d", i)
				}
			}
			live.committer.Partial("今天下午开会先说Voxt的进度然后讲会议功能还有一点")
			if err := live.rotate(context.Background(), false); err != nil {
				t.Fatal(err)
			}
			if len(got) != 1 || got[0] != tc.want {
				t.Fatalf("rotate committed %q, want [%q]", got, tc.want)
			}
		})
	}
}

// levelModel is a VAD that calls any non-zero frame speech.
type levelModel struct{}

func (levelModel) Feed(pcm []byte) ([]float64, error) {
	for _, b := range pcm {
		if b != 0 {
			return []float64{1}, nil
		}
	}
	return []float64{0}, nil
}
func (levelModel) Chunk() int         { return 160 }
func (levelModel) Threshold() float64 { return 0.5 }
func (levelModel) Close()             {}

// TestLiveSegmentEndsAtBreath replays #3987: a speaker who only takes short
// breaths got a segment every 8 s, each cut mid-sentence and ending in "。".
func TestLiveSegmentEndsAtBreath(t *testing.T) {
	const rate = 16000
	ctx := context.Background()
	open := func(partial string) (*Live, *[]Segment) {
		var got []Segment
		live := &Live{
			Clock:    (&tickingClock{t: time.Unix(0, 0)}).Now,
			Streamer: &liveFakeStreamer{partial: partial},
			Req:      asr.StreamRequest{SampleRate: rate},
			Speech:   NewSpeech(rate, levelModel{}),
			OnCommit: func(s Segment) { got = append(got, s) },
		}
		if err := live.Open(ctx); err != nil {
			t.Fatal(err)
		}
		return live, &got
	}
	push := func(live *Live, pcm []byte) {
		for off := 0; off < len(pcm); off += rate / 10 * 2 { // 100 ms chunks, like the recorder
			if err := live.Push(ctx, pcm[off:min(off+rate/10*2, len(pcm))]); err != nil {
				t.Fatal(err)
			}
		}
	}

	live, got := open("我想问一下，那个插件，它就是因为我对它的理解。")
	push(live, speechPCM(5*time.Second, rate))
	push(live, silencePCM(400*time.Millisecond, rate))
	if len(*got) != 1 {
		t.Fatalf("want a commit at the 0.4 s breath after 5 s of speech, got %d", len(*got))
	}
	if end := (*got)[0].End; end > 5500*time.Millisecond {
		t.Fatalf("segment ended at %v, want the breath near 5.4 s", end)
	}
	if text := (*got)[0].Text; text != "我想问一下，那个插件，它就是因为我对它的理解。" {
		t.Fatalf("pause commit keeps the model's full stop, got %q", text)
	}

	// A short breath early in a phrase is not a boundary yet.
	live, got = open("然后")
	push(live, speechPCM(time.Second, rate))
	push(live, silencePCM(400*time.Millisecond, rate))
	if len(*got) != 0 {
		t.Fatalf("0.4 s breath after 1 s committed %q", (*got)[0].Text)
	}

	// No pause at all: the 8 s cap cuts, without the model's full stop.
	for _, partial := range []string{"然后Typeless就是一个，然后微信书。", "so I think the."} {
		live, got = open(partial)
		push(live, speechPCM(8*time.Second, rate))
		if len(*got) != 1 {
			t.Fatalf("%q: want one forced cut at 8 s, got %d", partial, len(*got))
		}
		if text := (*got)[0].Text; strings.HasSuffix(text, "。") || strings.HasSuffix(text, ".") {
			t.Fatalf("forced cut mid-sentence ends in a full stop: %q", text)
		}
	}
}
