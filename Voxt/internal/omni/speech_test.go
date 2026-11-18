package omni

import (
	"context"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
)

// mossWithVAD is the fake's MOSS with Silero.
func mossWithVAD(t *testing.T, opts omnitest.Opts) (*fake, *ASR) {
	t.Helper()
	f := newFake(t, opts, withServers(t, "moss_transcribe_diarize_server"))
	v, err := NewVAD(f.rt, f.model(VADSilero))
	if err != nil {
		t.Fatal(err)
	}
	return f, f.asr(t, ASRMoss).WithVAD(v)
}

// TestMossSpeechOnly: MOSS's Final gets its speech with the long pause
// shortened, and no request for audio without speech.
func TestMossSpeechOnly(t *testing.T) {
	f, a := mossWithVAD(t, omnitest.Opts{})
	pcm := append(append(tone(time.Second, 0.3), silence(6*time.Second)...), tone(time.Second, 0.3)...)
	if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: pcm, Format: "pcm", SampleRate: sampleRate}); err != nil {
		t.Fatal(err)
	}
	finals := f.records("final")
	if len(finals) != 1 || finals[0].Samples >= 4*sampleRate || finals[0].Samples < 2*sampleRate {
		t.Fatalf("finals %+v", finals)
	}
	res, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: silence(5 * time.Second), Format: "pcm", SampleRate: sampleRate})
	if err != nil || res.Text != "" || len(f.records("final")) != 1 {
		t.Fatalf("silence: %q, %v, finals %+v", res.Text, err, f.records("final"))
	}
}

// TestMossLivePauses: MOSS's realtime socket does not get a long pause;
// the Final still gets all the audio to look at.
func TestMossLivePauses(t *testing.T) {
	f, a := mossWithVAD(t, omnitest.Opts{})
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	pcm := append(append(tone(time.Second, 0.3), silence(6*time.Second)...), tone(time.Second, 0.3)...)
	sendAll(t, st, pcm)
	s := st.(*live)
	written := func() (int, int) {
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.queued, s.previewed
	}
	deadline := time.Now().Add(10 * time.Second)
	for queued, _ := written(); queued > 0 && time.Now().Before(deadline); queued, _ = written() {
		time.Sleep(10 * time.Millisecond)
	}
	if _, previewed := written(); previewed >= 4*sampleRate*2 || previewed < 2*sampleRate*2 {
		t.Errorf("the socket got %d bytes of %d", previewed, len(pcm))
	}
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	drain(t, st)
	if finals := f.records("final"); len(finals) != 1 || finals[0].Samples < 2*sampleRate || finals[0].Samples >= 4*sampleRate {
		t.Errorf("finals %+v", finals)
	}
}

func TestMossDiarizeNoSpeech(t *testing.T) {
	f, a := mossWithVAD(t, omnitest.Opts{Segments: true})
	res, err := a.Diarize(context.Background(), asr.FileRequest{Audio: silence(5 * time.Second), Format: "pcm", SampleRate: sampleRate})
	if err != nil || res.Text != "" || len(res.Segments) != 0 || len(f.records("final")) != 0 {
		t.Fatalf("%+v, %v, finals %+v", res, err, f.records("final"))
	}
}

// TestMossDiarizeTags: a text the server did not split into segments
// comes without its tags.
func TestMossDiarizeTags(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "[0.25][S01]hello there[1.50]"}, withServers(t, "moss_transcribe_diarize_server"))
	res, err := f.asr(t, ASRMoss).Diarize(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3), Format: "pcm", SampleRate: sampleRate})
	if err != nil || res.Text != "hello there" {
		t.Fatalf("%q, %v", res.Text, err)
	}
	for _, s := range res.Segments {
		if s.Text != stripTags(s.Text) {
			t.Errorf("segment %+v", s)
		}
	}
}

// TestPauseCut: speech passes with its pre-roll; a pause passes only its
// first GapKeepMS/2.
func TestPauseCut(t *testing.T) {
	c := newPauseCut()
	sent := 0
	push := func(pcm []byte) {
		for _, fr := range asr.Frames(pcm, 100*time.Millisecond, sampleRate) {
			sent += len(c.push(fr))
		}
	}
	push(silence(2 * time.Second))
	if sent != 0 {
		t.Fatalf("leading silence: %d bytes", sent)
	}
	push(tone(time.Second, 0.3))
	if second := sampleRate * 2; sent < second || sent > second*3/2 {
		t.Fatalf("speech: %d bytes", sent)
	}
	before := sent
	push(silence(5 * time.Second))
	if pause := sent - before; pause == 0 || pause > sampleRate*2 {
		t.Fatalf("pause: %d bytes", pause)
	}
}
