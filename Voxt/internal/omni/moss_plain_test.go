package omni

// MOSS sometimes answers its plain-text instruction with its diarization
// markup anyway (a speaker turn [S01] or a timestamp [2.62], seen on
// single-speaker English dictation): the plain-text path (dictation,
// Transcribe, the meeting's mic track) drops it, so what is inserted and
// shown is plain text. The diarization path keeps everything.

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
)

const mossMarkedUp = "[S01] So,[S01] um,[S01] the first thing is we need to update the docs."

func TestMossPlainTextFinal(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: mossMarkedUp}, withServers(t, "moss_transcribe_diarize_server"))
	a := f.asr(t, ASRMoss)
	res, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3), Format: "pcm", Lang: "en"})
	if err != nil {
		t.Fatal(err)
	}
	want := "So, um, the first thing is we need to update the docs."
	if res.Text != want {
		t.Fatalf("text %q, want %q", res.Text, want)
	}
}

func TestMossPlainTextLive(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: mossMarkedUp, LiveMarkup: true}, withServers(t, "moss_transcribe_diarize_server"))
	a := f.asr(t, ASRMoss)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: "en", LiveResult: true})
	if err != nil {
		t.Fatal(err)
	}
	sendAll(t, st, tone(1500*time.Millisecond, 0.3))
	if sv, ok := st.(asr.Salvager); ok {
		if text := sv.Salvage(); strings.ContainsAny(text, "[]") {
			t.Fatalf("Salvage %q carries markup", text)
		}
	}
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := drain(t, st)
	if len(evs) == 0 || evs[len(evs)-1].Kind != asr.EventFinal {
		t.Fatalf("events %+v", evs)
	}
	for _, ev := range evs {
		if strings.ContainsAny(ev.Text, "[]") {
			t.Errorf("%v text %q carries markup", ev.Kind, ev.Text)
		}
	}
	want := "So, um, the first thing is we need to update the docs."
	if got := evs[len(evs)-1].Text; got != want {
		t.Errorf("final %q, want %q", got, want)
	}
}

// The diarization request keeps the markup: it is the segments' source.
func TestMossDiarizeKeepsMarkup(t *testing.T) {
	f := newFake(t, omnitest.Opts{Segments: true, Text: mossMarkedUp}, withServers(t, "moss_transcribe_diarize_server"))
	a := f.asr(t, ASRMoss)
	res, err := a.Diarize(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3), Format: "pcm"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != mossMarkedUp || len(res.Segments) != 2 {
		t.Fatalf("result %+v", res)
	}
}
