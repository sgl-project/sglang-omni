package omni

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
)

// withProbeOn turns Config.DecodeProbe on for the probe's tests.
func withProbeOn(t *testing.T, names ...string) func(*Config) {
	t.Helper()
	servers := withServers(t, names...)
	return func(c *Config) {
		servers(c)
		c.DecodeProbe = true
	}
}

// A live session times every Final request: the batch preview's
// provisional requests, and the Final at Finish. Each event carries the
// queue ahead of the request, never its text.
func TestDecodeProbeTimesPreviewAndFinal(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "text {n}"}, withProbeOn(t, "whisper_server"))
	a := f.asr(t, ASRWhisperTurbo)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: "zh-Hans"})
	if err != nil {
		t.Fatal(err)
	}
	sendAll(t, st, tone(2500*time.Millisecond, 0.3))
	select {
	case <-st.Events():
	case <-time.After(10 * time.Second):
		t.Fatal("no preview before Finish")
	}
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	drain(t, st)
	st.Close() // Close ends the session again; the log line must stay single
	s := st.(*live)
	events := s.probe.Events()
	if len(events) < 2 {
		t.Fatalf("decode events %+v: a preview and the Final", events)
	}
	if events[0].Kind != "preview" {
		t.Errorf("first event %+v, want the preview", events[0])
	}
	if events[len(events)-1].Kind != "final" {
		t.Errorf("last event %+v, want the Final", events[len(events)-1])
	}
	for _, ev := range events {
		if ev.FinishedMS < ev.SubmittedMS {
			t.Errorf("finished before submitted: %+v", ev)
		}
		if ev.Inflight < 0 {
			t.Errorf("negative queue: %+v", ev)
		}
	}
	lines := 0
	for _, l := range f.out {
		if strings.Contains(l, "decode_events") {
			lines++
		}
	}
	if lines != 1 {
		t.Errorf("decode_events logged %d times, want exactly one (r184 logged it twice)", lines)
	}
	f.rt.Close()
}

// A slow server makes the Final queue behind the in-flight preview: its
// event shows the one request ahead of it.
func TestDecodeProbeCountsTheQueue(t *testing.T) {
	f := newFake(t, omnitest.Opts{Delay: time.Second, Text: "text {n}"}, withProbeOn(t, "whisper_server"))
	a := f.asr(t, ASRWhisperTurbo)
	st, err := a.Open(t.Context(), asr.StreamRequest{SampleRate: sampleRate, Lang: "en"})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	sendAll(t, st, tone(time.Second, 0.3))
	eventually(t, "the preview request reached the server", func() bool {
		return len(f.records("final")) == 1
	})
	sendAll(t, st, tone(time.Second/2, 0.3))
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	drain(t, st)
	events := st.(*live).probe.Events()
	if len(events) < 2 {
		t.Fatalf("decode events %+v: a preview and the Final", events)
	}
	last := events[len(events)-1]
	if last.Kind != "final" || last.Inflight != 1 {
		t.Errorf("the Final's event %+v: it queued behind the one preview", last)
	}
	f.rt.Close()
}

// A preview that shows new text is marked changed; one over audio the
// model says nothing new about (here: the same text every time) is not,
// so pauses show as runs of false.
func TestDecodeProbeMarksUnchangedPreviews(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "same"}, withProbeOn(t, "whisper_server"))
	a := f.asr(t, ASRWhisperTurbo)
	st, err := a.Open(t.Context(), asr.StreamRequest{SampleRate: sampleRate, Lang: "en"})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	for chunks := 1; chunks <= 4; chunks++ {
		sendAll(t, st, tone(1200*time.Millisecond, 0.3))
		eventually(t, "the preview reached the server", func() bool {
			return len(f.records("final")) >= chunks
		})
	}
	select {
	case <-st.Events():
	case <-time.After(10 * time.Second):
		t.Fatal("no preview before Finish")
	}
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	drain(t, st)
	var flags []bool
	for _, ev := range st.(*live).probe.Events() {
		if ev.Kind != "preview" {
			continue
		}
		if ev.Changed == nil {
			continue // in flight at Finish: never evaluated by the assembler
		}
		flags = append(flags, *ev.Changed)
	}
	if len(flags) < 2 {
		t.Fatalf("want several previews, got %+v", flags)
	}
	if !flags[0] {
		t.Errorf("the first preview shows text: %+v", flags)
	}
	for _, changed := range flags[1:] {
		if changed {
			t.Errorf("the same text again is not a visible update: %+v", flags)
		}
	}
	f.rt.Close()
}

func TestDecodeKindNames(t *testing.T) {
	if (finalRequest{preview: true}).decodeKind() != "preview" {
		t.Error("a provisional request is the preview")
	}
	if (finalRequest{piece: true}).decodeKind() != "piece" {
		t.Error("an incremental request is a piece")
	}
	if (finalRequest{}).decodeKind() != "final" {
		t.Error("anything else is the Final")
	}
}
