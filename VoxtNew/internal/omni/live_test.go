package omni

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
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

func TestLive(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "最终文本"}, nil)
	a := f.asr(t, ASRFast)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: "zh", Hints: []string{"VoxtNew"}})
	if err != nil {
		t.Fatal(err)
	}
	pcm := tone(2500*time.Millisecond, 0.3)
	for _, fr := range asr.Frames(pcm, 100*time.Millisecond, 16000) {
		if err := st.Send(fr); err != nil {
			t.Fatal(err)
		}
		time.Sleep(5 * time.Millisecond)
	}
	// Partials come while the audio does.
	eventually(t, "a partial", func() bool { return len(st.Events()) > 0 })
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := drain(t, st)
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
	eventually(t, "the live record", func() bool { return len(f.records("live")) == 1 })
	if l := f.records("live")[0]; l.Fields["language"] != "zh" || l.Samples == 0 {
		t.Errorf("live %+v", l)
	}
	if err := st.Send(pcm[:3200]); err == nil {
		t.Error("Send after the end")
	}
	st.Close()
	st.Close()
}

func TestLiveClose(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, func(c *Config) { c.IdleTimeout = 200 * time.Millisecond })
	a := f.asr(t, ASRFast)
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
	// The lease is back: the idle unload happens.
	eventually(t, "the unload", func() bool { return len(f.records("shutdown")) == 1 })
	if len(f.records("final")) != 0 {
		t.Error("Close ran a Final")
	}
}

func TestLiveCancelled(t *testing.T) {
	f := newFake(t, omnitest.Opts{Delay: 5 * time.Second}, nil)
	a := f.asr(t, ASRFast)
	ctx, cancel := context.WithCancel(context.Background())
	st, err := a.Open(ctx, asr.StreamRequest{})
	if err != nil {
		t.Fatal(err)
	}
	_ = st.Send(tone(time.Second, 0.3))
	_ = st.Finish()
	time.Sleep(100 * time.Millisecond)
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
