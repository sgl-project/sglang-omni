package omni

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

// drain reads a stream's events until it closes.
func drain(t *testing.T, st asr.Stream) []asr.Event {
	t.Helper()
	var evs []asr.Event
	timeout := time.After(10 * time.Second)
	for {
		select {
		case ev, ok := <-st.Events():
			if !ok {
				return evs
			}
			evs = append(evs, ev)
		case <-timeout:
			t.Fatal("the stream did not end")
		}
	}
}

// waitIdle waits until the preview's writer has nothing queued: the
// idle token only triggers a re-check, so a stale one cannot pass early.
func waitIdle(t *testing.T, s *live) {
	t.Helper()
	deadline := time.After(10 * time.Second)
	for {
		s.mu.Lock()
		q := s.queued
		s.mu.Unlock()
		if q == 0 {
			return
		}
		select {
		case <-s.idle:
		case <-deadline:
			t.Fatal("the preview's writer did not go idle")
		}
	}
}

// watchLog makes c's Log look for a line with sub; the func it returns
// waits for the first one.
func watchLog(c *Config, sub string) func(*testing.T) {
	seen, once, log := make(chan struct{}), new(sync.Once), c.Log
	c.Log = func(level, msg string) {
		log(level, msg)
		if strings.Contains(msg, sub) {
			once.Do(func() { close(seen) })
		}
	}
	return func(t *testing.T) {
		t.Helper()
		select {
		case <-seen:
		case <-time.After(10 * time.Second):
			t.Fatalf("no log line with %q", sub)
		}
	}
}

// TestAudioBytes: the PCM an append message carries, for base64 lengths
// with 0, 1 and 2 padding bytes (DecodedLen counts the padding as data).
func TestAudioBytes(t *testing.T) {
	for _, n := range []int{0, 3, 4, 5, 6, 3198, 3199, 3200} {
		msg, _ := json.Marshal(map[string]string{"type": "input_audio_buffer.append", "audio": base64.StdEncoding.EncodeToString(make([]byte, n))})
		if got := audioBytes(msg); got != n {
			t.Errorf("%d bytes of PCM: got %d", n, got)
		}
	}
}

func TestLive(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "最终文本"}, nil)
	a := f.asr(t, ASRFast)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: "zh", Hints: []string{"VoxtNew"}})
	if err != nil {
		t.Fatal(err)
	}
	pcm := tone(2500*time.Millisecond, 0.3)
	sendAll(t, st, pcm)
	// Partials come while the audio does.
	var first asr.Event
	select {
	case first = <-st.Events():
	case <-time.After(10 * time.Second):
		t.Fatal("no partial before Finish")
	}
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := append([]asr.Event{first}, drain(t, st)...)
	if len(evs) < 2 {
		t.Fatalf("events %+v", evs)
	}
	for _, ev := range evs[:len(evs)-1] {
		if ev.Kind != asr.EventPartial || !strings.HasPrefix(ev.Text, "partial ") {
			t.Errorf("before the final: %+v", ev)
		}
	}
	last := evs[len(evs)-1]
	if last.Kind != asr.EventFinal || last.Text != "最终文本" || last.Result.Model != ASRFast || last.Result.Duration != 2500*time.Millisecond {
		t.Errorf("final %+v", last)
	}
	// The Final got everything sent, with the hint and the bias.
	finals := f.records("final")
	if len(finals) != 1 || finals[0].Samples != 40000 || finals[0].Fields["prompt"] != "VoxtNew" || finals[0].Fields["language"] != "zh" {
		t.Errorf("finals %+v", finals)
	}
	f.rt.Close() // the server records its live sessions before it exits
	if l := f.records("live"); len(l) != 1 || l[0].Fields["language"] != "zh" || l[0].Fields["final_prompt"] != "VoxtNew" || l[0].Fields["prompt"] != "" || l[0].Samples == 0 {
		t.Errorf("live %+v", l)
	}
	if err := st.Send(pcm[:3200]); err == nil {
		t.Error("Send after the end")
	}
	st.Close()
	st.Close()
}

func TestLateCandidatesAreOptIn(t *testing.T) {
	for _, tc := range []struct {
		name    string
		kind    string
		request asr.StreamRequest
		want    bool
	}{
		{"Qwen3-ASR opted in", kindQwen3ASR, asr.StreamRequest{LatePartials: true}, true},
		{"Qwen3-ASR without opt-in", kindQwen3ASR, asr.StreamRequest{}, false},
		{"other realtime model", kindNemotron, asr.StreamRequest{LatePartials: true}, false},
		{"LiveResult", kindQwen3ASR, asr.StreamRequest{LatePartials: true, LiveResult: true}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rt := NewRuntime(Config{IdleTimeout: -1})
			t.Cleanup(rt.Close)
			proc := &process{}
			server := newServer(rt, Spec{ID: "candidate-test"}, "", "")
			server.proc, server.leases = proc, 1
			stream := &live{
				a: &ASR{spec: Spec{Kind: tc.kind}}, ctx: context.Background(), cancel: func() {},
				l: &lease{s: server, p: proc}, events: make(chan asr.Event, eventBuffer), finished: true,
				lateCandidates: wantsLateCandidates(tc.request, true, tc.kind),
			}
			if stream.lateCandidates != tc.want {
				t.Fatalf("late candidate eligibility %v, want %v", stream.lateCandidates, tc.want)
			}
			stream.handleSegment(realtimeEvent{Index: 1, Segment: 0, Text: "late transcript"})
			stream.end(asr.Event{Kind: asr.EventFinal, Text: "authoritative transcript"})
			events := drain(t, stream)
			if tc.want {
				if len(events) != 2 || events[0].Kind != asr.EventCandidate || events[0].Text != "late transcript" {
					t.Fatalf("candidate was not delivered before Final: %+v", events)
				}
			} else if len(events) != 1 {
				t.Fatalf("ineligible candidate was emitted: %+v", events)
			}
			if last := events[len(events)-1]; last.Kind != asr.EventFinal || last.Text != "authoritative transcript" {
				t.Fatalf("terminal event %+v", last)
			}
		})
	}
}

// note (chenye): TestFinalPrior checks that the Final carries the preview's
// text and session. Without a language hint it asks the server to propose
// the language header and text. A chunked recording carries the session only
// on the first chunk and neither kind of guess on any chunk.
func TestFinalPrior(t *testing.T) {
	for _, tc := range []struct{ lang, text, autoGuess string }{
		{"zh", "partial 5", ""},
		{"", "", "true"},
	} {
		f := newFake(t, omnitest.Opts{}, nil)
		st, err := f.asr(t, ASRFast).Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: tc.lang})
		if err != nil {
			t.Fatal(err)
		}
		// 8000 samples in one append: the fake's one partial is "partial 5".
		if err := st.Send(tone(500*time.Millisecond, 0.3)); err != nil {
			t.Fatal(err)
		}
		select {
		case <-st.Events():
		case <-time.After(10 * time.Second):
			t.Fatal("no partial")
		}
		if err := st.Finish(); err != nil {
			t.Fatal(err)
		}
		drain(t, st)
		finals := f.records("final")
		if len(finals) != 1 || finals[0].Fields["guessed_text"] != tc.text ||
			finals[0].Fields["guess_auto_language"] != tc.autoGuess || finals[0].Fields["realtime_session"] != "fake-session" {
			t.Errorf("language %q: finals %+v, want the text %q, auto guess %q and the session", tc.lang, finals, tc.text, tc.autoGuess)
		}
		f.rt.Close() // the server records its live sessions before it exits
		// Without a dictionary the session still has the Final's (empty)
		// prompt to prefill.
		if l := f.records("live"); len(l) != 1 {
			t.Errorf("language %q: live sessions %+v", tc.lang, l)
		} else if p, ok := l[0].Fields["final_prompt"]; !ok || p != "" {
			t.Errorf("language %q: final_prompt %q, sent %v, want an empty one", tc.lang, p, ok)
		}
	}

	for _, tc := range []struct {
		lang  string
		known prior
	}{
		{"zh", prior{text: "text", session: "s"}},
		{"", prior{session: "s", guessAutoLanguage: true}},
	} {
		f := newFake(t, omnitest.Opts{}, nil)
		ctx := context.Background()
		a := f.asr(t, ASRFast)
		l, err := f.rt.acquire(ctx, a.spec, a.dir, "")
		if err != nil {
			t.Fatal(err)
		}
		defer l.release()
		if _, _, err := a.final(ctx, l, tone(130*time.Second, 0.3), tc.lang, bias{}, tc.known); err != nil {
			t.Fatal(err)
		}
		finals := f.records("final")
		if len(finals) < 2 {
			t.Fatalf("130 s of audio in %d requests", len(finals))
		}
		for i, r := range finals {
			if r.Fields["guessed_text"] != "" || r.Fields["guess_auto_language"] != "" ||
				(r.Fields["realtime_session"] == "s") != (i == 0) {
				t.Errorf("language %q, chunk %d: %v", tc.lang, i, r.Fields)
			}
		}
	}
}

// TestLiveResult: with LiveResult the live socket's whole text is the
// result and no Final runs: the second pass over the whole recording made
// every dictation wait (#3644 FB7). When the socket ends without the text,
// the Final runs on the whole recording, as without LiveResult.
func TestLiveResult(t *testing.T) {
	for _, drop := range []bool{false, true} {
		f := newFake(t, omnitest.Opts{Text: "最终文本", LiveDrop: drop}, nil)
		a := f.asr(t, ASRFast)
		st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: "zh", LiveResult: true})
		if err != nil {
			t.Fatal(err)
		}
		sendAll(t, st, tone(2500*time.Millisecond, 0.3))
		if err := st.Finish(); err != nil {
			t.Fatal(err)
		}
		evs := drain(t, st)
		want, finals := "live final", 0
		if drop {
			want, finals = "最终文本", 1
		}
		if last := evs[len(evs)-1]; last.Kind != asr.EventFinal || last.Text != want || last.Result.Duration != 2500*time.Millisecond {
			t.Errorf("dropped %v: last event %+v", drop, last)
		}
		if got := f.records("final"); len(got) != finals || finals == 1 && got[0].Samples != 40000 {
			t.Errorf("dropped %v: Finals %+v", drop, got)
		}
		st.Close()
		f.rt.Close() // the server records its live sessions before it exits
		// The socket's text is the result: no Final to prefill.
		for _, l := range f.records("live") {
			if _, ok := l.Fields["final_prompt"]; ok {
				t.Errorf("dropped %v: final_prompt sent with LiveResult", drop)
			}
		}
	}
}

// TestLiveResultFullQueueSalvagesAfterFinalFails: a full preview queue
// sends LiveResult through its Final fallback. If that Final also fails,
// the queued audio still reaches the preview before Salvage chooses it.
func TestLiveResultFullQueueSalvagesAfterFinalFails(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "p{n}", FailFrom: 1, Delay: 100 * time.Millisecond}, nil)
	st, err := f.asr(t, ASRFast).Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate, LiveResult: true})
	if err != nil {
		t.Fatal(err)
	}
	s := st.(*live)
	gate := make(chan struct{})
	release := sync.OnceFunc(func() { close(gate) })
	defer release()
	s.writeGate = gate
	if err := st.Send(tone(10*time.Millisecond, 0.3)); err != nil {
		t.Fatal(err)
	}
	eventually(t, "the writer holding the first message", func() bool { return len(s.out) == 0 })
	for range cap(s.out) {
		if err := st.Send(tone(10*time.Millisecond, 0.3)); err != nil {
			t.Fatal(err)
		}
	}
	s.mu.Lock()
	queued, preview := len(s.out), s.preview
	s.mu.Unlock()
	if queued != cap(s.out) || !preview {
		t.Fatalf("preview queue has %d of %d messages, preview %v", queued, cap(s.out), preview)
	}
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	f.waitLogged(t, "the socket's queue is full")
	f.waitLogged(t, "failed after")
	s.mu.Lock()
	finalOnly := s.finalOnly
	s.mu.Unlock()
	release()
	evs := drain(t, st)
	if last := evs[len(evs)-1]; last.Kind != asr.EventError {
		t.Fatalf("last %+v", last)
	}
	if !finalOnly {
		t.Error("the LiveResult fallback did not allow the preview wrap-up")
	}
	if got := st.(asr.Salvager).Salvage(); !strings.Contains(got, "partial ") {
		t.Errorf("salvaged %q, want the queued audio's preview", got)
	}
}

// TestLiveLargeFirstSend: when the model is slow to load, the audio
// recorded meanwhile goes in the stream's first Send, a second or more of
// it. The preview must take it and send partials. The fake server
// refused a message over 32 KB (its websocket's default limit; the real
// server takes it), so the preview stopped and no partial came: the app's
// TestSessionOmni timed out whenever the model loaded slowly.
func TestLiveLargeFirstSend(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "最终文本"}, nil)
	a := f.asr(t, ASRFast)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: "zh"})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if err := st.Send(tone(2*time.Second, 0.3)); err != nil { // 64 KB of PCM in one message
		t.Fatal(err)
	}
	select {
	case ev := <-st.Events():
		if ev.Kind != asr.EventPartial || !strings.HasPrefix(ev.Text, "partial ") {
			t.Fatalf("first event %+v", ev)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("no partial: the preview did not take the first second of audio")
	}
}

// TestLiveFinalWhenTheSocketStalls: the server takes the live session,
// then hangs (it never reads the socket again), so the preview's writer
// blocks on the socket with audio still queued. Stop must still run the
// Final: a succeeding Final never waits for the stuck writer (the
// wrap-up is for Salvage only), and the socket closes under it.
func TestLiveFinalWhenTheSocketStalls(t *testing.T) {
	// A small send buffer on this end too (the fake caps its receive
	// buffer): the kernels' autotuning would otherwise take the whole
	// recording without the writer ever blocking.
	old := loopback
	loopback = &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			c, err := new(net.Dialer).DialContext(ctx, network, addr)
			if tc, ok := c.(*net.TCPConn); err == nil && ok {
				if err := tc.SetWriteBuffer(16 << 10); err != nil {
					_ = c.Close()
					return nil, err // without it the writer never blocks: no stall to test
				}
			}
			return c, err
		},
	}}
	defer func() { loopback = old }()
	f := newFake(t, omnitest.Opts{Text: "最终文本", LiveStall: true}, nil)
	a := f.asr(t, ASRFast)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: 16000})
	if err != nil {
		t.Fatal(err)
	}
	// 800 KB of PCM (over 1 MB on the socket): past the most a loopback
	// socket buffers even with the small buffers above, so the writer
	// blocks on it. The queue still takes it (maxQueued is 30 s).
	pcm := tone(25*time.Second, 0.3)
	sendAll(t, st, pcm)
	stopped := time.Now()
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := drain(t, st)
	if took := time.Since(stopped); took > previewDrain+5*time.Second {
		t.Errorf("the stream ended %v after Stop, over the %v drain bound", took, previewDrain)
	}
	if last := evs[len(evs)-1]; last.Kind != asr.EventFinal || last.Text != "最终文本" {
		t.Fatalf("final %+v", last)
	}
	if finals := f.records("final"); len(finals) != 1 || finals[0].Samples != 25*sampleRate {
		t.Errorf("finals %+v: the whole recording goes to the Final", finals)
	}
	st.Close()
}

// TestLiveFinalFailsWhenTheSocketStalls: the server answers the session
// update, then hangs, and the Final fails. The wrap-up's wait for an
// answer gives up on a quiet server (previewQuiet) instead of running
// the full previewCompleteTimeout behind a dead one, so the error ends
// the stream within the quiet bound. The queue is empty at Stop (the
// kernel took the small send), so this is the drain-succeeds path.
func TestLiveFinalFailsWhenTheSocketStalls(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "p{n}", LiveStall: true, FailFrom: 1}, nil)
	a := f.asr(t, ASRFast)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	sendAll(t, st, tone(100*time.Millisecond, 0.3))
	stopped := time.Now()
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := drain(t, st)
	if took := time.Since(stopped); took > previewQuiet+3*time.Second {
		t.Errorf("the error came %v after Stop, over the %v quiet bound", took, previewQuiet)
	}
	if last := evs[len(evs)-1]; last.Kind != asr.EventError {
		t.Fatalf("last %+v", last)
	}
}

// TestLiveFinalFailsWhenTheDrainTimesOut: the stall of
// TestLiveFinalWhenTheSocketStalls, but the Final fails: the writer is
// provably blocked (800 KB queued, a held writer, a server that never
// reads — kernel buffer sizes differ per platform, the gate does not),
// the drain gives up after previewDrain (the log says so) and closes
// the socket, the wrap-up is skipped, and the error ends the stream
// when its timer expires, without waiting on the wall clock.
func TestLiveFinalFailsWhenTheDrainTimesOut(t *testing.T) {
	old := loopback
	loopback = &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			c, err := new(net.Dialer).DialContext(ctx, network, addr)
			if tc, ok := c.(*net.TCPConn); err == nil && ok {
				if err := tc.SetWriteBuffer(16 << 10); err != nil {
					_ = c.Close()
					return nil, err // without it the writer never blocks: no stall to test
				}
			}
			return c, err
		},
	}}
	t.Cleanup(func() { loopback = old })
	f := newFake(t, omnitest.Opts{Text: "p{n}", LiveStall: true, FailFrom: 1}, nil)
	clock := new(fakeClock).install(f.rt)
	scheduled := make(chan struct{})
	after := f.rt.afterFunc
	f.rt.afterFunc = func(d time.Duration, fn func()) func() {
		stop := after(d, fn)
		if d == previewDrain {
			close(scheduled)
		}
		return stop
	}
	a := f.asr(t, ASRFast)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	// Hold the writer: the whole recording stays queued whatever the
	// kernels' buffers; released once the drain has given up.
	s := st.(*live)
	gate := make(chan struct{})
	s.writeGate = gate
	defer close(gate)
	sendAll(t, st, tone(25*time.Second, 0.3))
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	f.waitLogged(t, "failed after")
	select {
	case <-scheduled:
	case <-time.After(10 * time.Second):
		t.Fatal("the drain timer was not scheduled")
	}
	clock.advance(previewDrain - time.Nanosecond)
	select {
	case ev := <-st.Events():
		t.Fatalf("event before the drain deadline: %+v", ev)
	default:
	}
	if f.logged("did not drain") {
		t.Fatal("the drain timed out before its deadline")
	}
	clock.advance(time.Nanosecond)
	f.waitLogged(t, "did not drain")
	evs := drain(t, st)
	if last := evs[len(evs)-1]; last.Kind != asr.EventError {
		t.Fatalf("last %+v", last)
	}
}

// TestLiveResultSalvageAfterFinish: a LiveResult stream's finishLive
// owns s.out (transcription.done goes through it) until it ends the
// stream, so a Salvage after Finish must not close s.out under it
// (it panicked with send on closed channel). LiveDrop keeps finishLive
// pending on the server's answer. One goroutine at a time, a few
// rounds: the race needs the close to land before the send.
func TestLiveResultSalvageAfterFinish(t *testing.T) {
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	for _, test := range []struct {
		name string
		load time.Duration
	}{
		{name: "immediate Open"},
		{name: "delayed Open", load: 250 * time.Millisecond},
	} {
		t.Run(test.name, func(t *testing.T) {
			for range 5 {
				f := newFake(t, omnitest.Opts{LiveDrop: true, Load: test.load}, nil)
				ctx, cancel := context.WithCancel(context.Background())
				st, err := f.asr(t, ASRFast).Open(ctx, asr.StreamRequest{SampleRate: sampleRate, LiveResult: true})
				if err != nil {
					cancel()
					t.Fatal(err)
				}
				// A slow start must not spend the race's cancellation budget.
				timeout := time.AfterFunc(10*time.Second, cancel)
				sendAll(t, st, tone(500*time.Millisecond, 0.3))
				if err := st.Finish(); err != nil {
					t.Fatal(err)
				}
				_ = st.(asr.Salvager).Salvage()
				st.Close()
				timeout.Stop()
				cancel()
			}
		})
	}
}

func TestLiveClose(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, func(c *Config) { c.IdleTimeout = 200 * time.Millisecond })
	a := f.asr(t, ASRFast)
	clock := new(fakeClock).install(f.rt)
	st, err := a.Open(context.Background(), asr.StreamRequest{})
	if err != nil {
		t.Fatal(err)
	}
	_ = st.Send(tone(time.Second, 0.3))
	st.Close()
	for _, ev := range drain(t, st) {
		if ev.Kind != asr.EventPartial {
			t.Errorf("after Close: %+v", ev)
		}
	}
	// The lease is back: the idle unload happens (the idle stop returns
	// once the process exited).
	clock.advance(200 * time.Millisecond)
	if n := len(f.records("shutdown")); n != 1 {
		t.Errorf("%d shutdowns after the idle time", n)
	}
	if len(f.records("final")) != 0 {
		t.Error("Close ran a Final")
	}
}

func TestLiveCancelled(t *testing.T) {
	var finalSent func(*testing.T)
	f := newFake(t, omnitest.Opts{Delay: 5 * time.Second}, func(c *Config) { finalSent = watchLog(c, "Final request") })
	a := f.asr(t, ASRFast)
	ctx, cancel := context.WithCancel(context.Background())
	st, err := a.Open(ctx, asr.StreamRequest{})
	if err != nil {
		t.Fatal(err)
	}
	_ = st.Send(tone(time.Second, 0.3))
	_ = st.Finish()
	finalSent(t) // the cancel comes during the Final
	cancel()
	evs := drain(t, st)
	if last := evs[len(evs)-1]; last.Kind != asr.EventError || asr.KindOf(last.Err) != asr.KindCancelled {
		t.Errorf("last %+v", last)
	}
	st.Close()
}

// TestLiveStreamAll: asr.StreamAll (the end-to-end helpers) over it.
func TestLiveStreamAll(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "ok"}, nil)
	var partials []string
	res, err := asr.StreamAll(context.Background(), f.asr(t, ASRAccurate), asr.StreamRequest{}, tone(1500*time.Millisecond, 0.3),
		asr.Feed{OnPartial: func(s string) { partials = append(partials, s) }})
	if err != nil || res.Text != "ok" {
		t.Fatalf("%+v %v", res, err)
	}
}

func TestVAD(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, nil)
	v, err := NewVAD(f.rt, f.model(VADSilero))
	if err != nil {
		t.Fatal(err)
	}
	pcm := append(append(silence(time.Second), tone(time.Second, 0.3)...), silence(time.Second)...)
	segs, err := v.Speech(context.Background(), pcm, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 1 || segs[0].Start < 15000 || segs[0].Start > 16500 || segs[0].End < 31500 || segs[0].End > 33000 {
		t.Errorf("segments %+v", segs)
	}
	ts := f.records("vad-timestamps")
	if len(ts) != 1 || ts[0].Fields["threshold"] != "0.5" || ts[0].Fields["min_speech_duration_ms"] != "250" ||
		ts[0].Fields["min_silence_duration_ms"] != "100" || ts[0].Fields["speech_pad_ms"] != "30" {
		t.Errorf("request %+v", ts)
	}
	if _, err := v.Speech(context.Background(), pcm, 8000); err == nil {
		t.Error("8 kHz")
	}

	ms, err := v.Open(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var probs []float64
	for _, fr := range asr.Frames(pcm, 100*time.Millisecond, 16000) {
		p, err := ms.Feed(fr)
		if err != nil {
			t.Fatal(err)
		}
		probs = append(probs, p...)
	}
	ms.Close()
	ms.Close()
	if len(probs) != 48000/512 || ms.Chunk() != 512 || ms.Threshold() != 0.5 {
		t.Fatalf("%d probabilities", len(probs))
	}
	speech := 0
	for i, p := range probs {
		if p >= 0.5 {
			speech++
			if i*512 < 15500 || i*512 > 32000 {
				t.Errorf("speech at chunk %d", i)
			}
		}
	}
	if speech < 28 {
		t.Errorf("%d speech chunks", speech)
	}

	// A gate on the model opens at the tone, not before.
	g := vad.NewGate(16000, vad.Params{})
	ms, err = v.Open(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ms.Close()
	g.UseModel(ms)
	sent := 0
	for _, fr := range asr.Frames(pcm, 100*time.Millisecond, 16000) {
		sent += len(g.Push(fr))
	}
	if !g.Open() || !g.Heard() || g.ModelErr() != nil || g.Dropped() < 2*(16000-4800-1600) {
		t.Errorf("gate open %v heard %v err %v dropped %d", g.Open(), g.Heard(), g.ModelErr(), g.Dropped())
	}
}

func TestVADFails(t *testing.T) {
	f := newFake(t, omnitest.Opts{Mode: omnitest.ModeVADError}, nil)
	v, err := NewVAD(f.rt, f.model(VADSilero))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := v.Speech(context.Background(), tone(time.Second, 0.3), 16000); err == nil {
		t.Error("Speech of a failing server")
	}
	ms, err := v.Open(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ms.Close()
	var ferr error
	for _, fr := range asr.Frames(silence(2*time.Second), 100*time.Millisecond, 16000) {
		if _, ferr = ms.Feed(fr); ferr != nil {
			break
		}
	}
	if ferr == nil {
		t.Error("the broken stream did not fail")
	}
}

// TestLiveServerDies: the server dies while the session records (a
// crash, memory pressure). The Final runs on a new server over the whole
// recording instead of failing: the dictation is not lost.
func TestLiveServerDies(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "最终文本"}, nil)
	st, err := f.asr(t, ASRFast).Open(context.Background(), asr.StreamRequest{SampleRate: 16000})
	if err != nil {
		t.Fatal(err)
	}
	pcm := tone(2500*time.Millisecond, 0.3)
	sendAll(t, st, pcm)
	pid := f.records("launch")[0].PID
	if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
		t.Fatal(err)
	}
	f.waitLogged(t, "exited unexpectedly")
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := drain(t, st)
	if last := evs[len(evs)-1]; last.Kind != asr.EventFinal || last.Text != "最终文本" {
		t.Fatalf("final %+v", last)
	}
	if finals := f.records("final"); len(finals) != 1 || finals[0].Samples != 40000 || finals[0].PID == pid {
		t.Errorf("finals %+v: the whole recording goes to the new server", finals)
	}
}
