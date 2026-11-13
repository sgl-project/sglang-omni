package omni

import (
	"context"
	"os"
	"strings"
	"sync"
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
	if l := f.records("live"); len(l) != 1 || l[0].Fields["language"] != "zh" || l[0].Samples == 0 {
		t.Errorf("live %+v", l)
	}
	if err := st.Send(pcm[:3200]); err == nil {
		t.Error("Send after the end")
	}
	st.Close()
	st.Close()
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
	p, err := os.FindProcess(pid)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = p.Release()
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
