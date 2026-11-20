package meeting

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

func TestCollapseLoops(t *testing.T) {
	sentence := "But it is said that if it is really necessary to battle on the screen, it is impossible to say. "
	cases := []struct{ in, want string }{
		{"", ""},
		{"我们今天开会。", "我们今天开会。"},
		{"我我我其实", "我我我其实"},
		{"对对对对对", "对对对对对"},
		{"this, this, this, right", "this, this, this, right"},
		{"一亿是100000000。", "一亿是100000000。"},
		{"……………………………………", "……………………………………"},
		{"做成一个，" + strings.Repeat("那个，", 300) + "那", "做成一个，那个，"},
		{"李，" + strings.Repeat("我", 500), "李，我"},
		{"从我们。On the results, " + strings.Repeat("on the results, ", 200) + "on the res", "从我们。On the results, on the results,"},
		{strings.Repeat(sentence, 40), strings.TrimSpace(sentence)},
		{strings.Repeat("その交換が仕上がった時に、", 60) + "我们继续", "その交換が仕上がった時に、我们继续"},
		{"ha " + strings.Repeat("ha ", 9) + "ok", "ha ok"},
	}
	for _, c := range cases {
		if got := collapseLoops(c.in); got != c.want {
			t.Errorf("collapseLoops(%.40q…) = %.80q, want %q", c.in, got, c.want)
		}
	}
}

// loopStreamer's streams send partial for every Send and end with final,
// carrying lang; it records each Open's request.
type loopStreamer struct {
	mu       sync.Mutex
	reqs     []asr.StreamRequest
	partial  func(n int) string // the n-th stream's partial
	final    string
	lang     string
	openErrs int // Opens to fail after the first
}

func (f *loopStreamer) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (f *loopStreamer) Open(_ context.Context, req asr.StreamRequest) (asr.Stream, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.reqs) > 0 && f.openErrs > 0 {
		f.openErrs--
		return nil, errors.New("connect failed")
	}
	f.reqs = append(f.reqs, req)
	return &loopStream{f: f, n: len(f.reqs) - 1, events: make(chan asr.Event, 64)}, nil
}

type loopStream struct {
	f      *loopStreamer
	n      int
	events chan asr.Event
}

func (s *loopStream) Send([]byte) error {
	select {
	case s.events <- asr.Event{Kind: asr.EventPartial, Text: s.f.partial(s.n)}:
	default:
	}
	return nil
}

func (s *loopStream) Finish() error {
	s.events <- asr.Event{Kind: asr.EventFinal, Text: s.f.final, Result: asr.Result{Text: s.f.final, Lang: s.f.lang}}
	close(s.events)
	return nil
}

func (s *loopStream) Close()                   {}
func (s *loopStream) Events() <-chan asr.Event { return s.events }

func newLoopLive(t *testing.T, f *loopStreamer, req asr.StreamRequest) (*Live, *[]Segment) {
	t.Helper()
	clk := &tickingClock{t: time.Unix(0, 0)}
	var segs []Segment
	l := &Live{Clock: clk.Now, Streamer: f, Req: req, Speech: NewSpeech(16000, nil),
		OnCommit: func(s Segment) { segs = append(segs, s) }}
	if err := l.Open(context.Background()); err != nil {
		t.Fatal(err)
	}
	return l, &segs
}

// TestLiveLoopRestartsStream: a stream whose partial loops commits the text
// once, without its Final (which repeats it), and the next stream opens with
// the language that Final heard (#3981).
func TestLiveLoopRestartsStream(t *testing.T) {
	looped := "做成一个，" + strings.Repeat("那个，", 40)
	f := &loopStreamer{lang: "Chinese", final: "做成一个完整的转写",
		partial: func(n int) string {
			if n == 0 {
				return looped
			}
			return "接着说"
		}}
	l, segs := newLoopLive(t, f, asr.StreamRequest{SampleRate: 16000})
	ctx := context.Background()
	if err := l.Push(ctx, speechPCM(100*time.Millisecond, 16000)); err != nil {
		t.Fatal(err)
	}
	if len(f.reqs) != 2 {
		t.Fatalf("streams opened = %d, want 2 (a restart on the loop)", len(f.reqs))
	}
	if got := f.reqs[1].Lang; got != "Chinese" {
		t.Errorf("the next stream's language hint = %q, want the Final's %q", got, "Chinese")
	}
	if len(*segs) != 1 || (*segs)[0].Text != looped {
		t.Fatalf("segments = %+v, want the looped tail once (commitSegment collapses it)", *segs)
	}
	if got := collapseLoops((*segs)[0].Text); got != "做成一个，那个，" {
		t.Errorf("stored text = %q", got)
	}
	if l.LiveText() != "" {
		t.Errorf("live text after the restart = %q, want empty", l.LiveText())
	}
}

// TestLiveRotateKeepsUserHint: the user's hint wins over the language a
// Final heard.
func TestLiveRotateKeepsUserHint(t *testing.T) {
	f := &loopStreamer{lang: "Japanese", final: "x", partial: func(int) string { return "x" }}
	hints := []string{"SGLang", "Voxt"}
	l, _ := newLoopLive(t, f, asr.StreamRequest{SampleRate: 16000, Lang: "zh", Hints: hints})
	if err := l.rotate(context.Background(), false, true); err != nil {
		t.Fatal(err)
	}
	if got := f.reqs[1].Lang; got != "zh" {
		t.Errorf("hint after rotation = %q, want the user's %q", got, "zh")
	}
	if !slices.Equal(f.reqs[1].Hints, hints) {
		t.Errorf("hot words after rotation = %q, want %q", f.reqs[1].Hints, hints)
	}
}

// TestLiveRotateRetriesOpen: a rotation whose Open fails leaves the track
// live: the next Push opens a stream and sends to it.
func TestLiveRotateRetriesOpen(t *testing.T) {
	f := &loopStreamer{final: "a", openErrs: 1, partial: func(int) string { return "b" }}
	hints := []string{"SGLang", "Voxt"}
	l, _ := newLoopLive(t, f, asr.StreamRequest{SampleRate: 16000, Hints: hints})
	ctx := context.Background()
	if err := l.rotate(ctx, false, true); err == nil {
		t.Fatal("want the Open's error")
	}
	if err := l.Push(ctx, speechPCM(100*time.Millisecond, 16000)); err != nil {
		t.Fatal(err)
	}
	if len(f.reqs) != 2 || l.LiveText() != "b" {
		t.Fatalf("streams = %d, live text %q: want a reopened stream with its partial", len(f.reqs), l.LiveText())
	}
	if !slices.Equal(f.reqs[1].Hints, hints) {
		t.Errorf("hot words after reopening = %q, want %q", f.reqs[1].Hints, hints)
	}
}
