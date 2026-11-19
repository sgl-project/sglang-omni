package omni

import (
	"context"
	"strings"
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

// TestMossLivePauses: MOSS's realtime socket does not get a long pause,
// though the preview's coverage is the whole recording; the Final still
// gets all the audio to look at.
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
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	drain(t, st)
	var got int
	for deadline := time.Now().Add(10 * time.Second); ; {
		if live := f.records("live"); len(live) == 1 {
			got = live[0].Samples
			break
		} else if len(live) > 1 || time.Now().After(deadline) {
			t.Fatalf("live records %+v", live)
		}
		time.Sleep(10 * time.Millisecond)
	}
	if got >= 4*sampleRate || got < 2*sampleRate {
		t.Errorf("the socket got %d samples of %d", got, len(pcm)/2)
	}
	if _, previewed := written(); previewed < len(pcm)-sampleRate*2 {
		t.Errorf("the preview covers %d bytes of %d", previewed, len(pcm))
	}
	if finals := f.records("final"); len(finals) != 1 || finals[0].Samples < 2*sampleRate || finals[0].Samples >= 4*sampleRate {
		t.Errorf("finals %+v", finals)
	}
}

// TestSalvageAfterPauseCut: the socket never hears the pauses the cut
// drops, but Salvage still weighs the pieces' coverage against the whole
// recording: speech, a long pause and more speech, with the Final
// failing, salvages the preview's text (it carries every stretch), not
// only the early pieces'.
func TestSalvageAfterPauseCut(t *testing.T) {
	f, a := mossWithVAD(t, omnitest.Opts{Text: "p{n}", FailFrom: 2})
	a.spec.ASR.PiecesAfter, a.spec.ASR.PieceMin, a.spec.ASR.PieceSeconds, a.spec.ASR.PieceMax = 2*time.Second, time.Second, 1500*time.Millisecond, 3*time.Second
	a.spec.ASR.PauseMin = 150 * time.Millisecond
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	pcm := append(append(tone(500*time.Millisecond, 0.3), silence(10*time.Second)...), tone(500*time.Millisecond, 0.3)...)
	sendAll(t, st, pcm)
	select { // the preview heard the speech
	case ev := <-st.Events():
		if ev.Kind != asr.EventPartial {
			t.Fatalf("before Finish: %+v", ev)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("no live partial")
	}
	f.waitLogged(t, "piece 1 of the recording")
	s := st.(*live)
	previewed := func() int {
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.previewed
	}
	for deadline := time.Now().Add(10 * time.Second); previewed() < sampleRate*2*12/10 && time.Now().Before(deadline); {
		time.Sleep(10 * time.Millisecond) // the second stretch reached the socket
	}
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := drain(t, st)
	if last := evs[len(evs)-1]; last.Kind != asr.EventError {
		t.Fatalf("last %+v", last)
	}
	if got := st.(asr.Salvager).Salvage(); !strings.HasPrefix(got, "partial") {
		t.Errorf("salvaged %q, want the preview's text, which carries every stretch of speech", got)
	}
}

// TestSalvageDrainsQueuedTail: speech still queued for the socket when
// Stop comes is delivered before the Final runs, so a failed Final
// salvages the tail's words too. Speech, a long pause and more speech;
// the writer is held, so the stretch after the pause is provably still
// queued at Stop, and its text differs from the stretch before it.
func TestSalvageDrainsQueuedTail(t *testing.T) {
	f, a := mossWithVAD(t, omnitest.Opts{Text: "p{n}", FailFrom: 2, Delay: 2 * time.Second})
	a.spec.ASR.PiecesAfter, a.spec.ASR.PieceMin, a.spec.ASR.PieceSeconds, a.spec.ASR.PieceMax = 2*time.Second, time.Second, 1500*time.Millisecond, 3*time.Second
	a.spec.ASR.PauseMin = 150 * time.Millisecond
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	s := st.(*live)
	queued := func() int {
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.queued
	}
	// The stretch before the pause reaches the server.
	sendAll(t, st, tone(500*time.Millisecond, 0.3))
	var before string
	select {
	case ev := <-st.Events():
		if ev.Kind != asr.EventPartial {
			t.Fatalf("before the pause: %+v", ev)
		}
		before = ev.Text
	case <-time.After(10 * time.Second):
		t.Fatal("no live partial")
	}
	for deadline := time.Now().Add(10 * time.Second); queued() != 0 && time.Now().Before(deadline); {
		time.Sleep(10 * time.Millisecond) // the writer is idle before it is held
	}
	if queued() != 0 {
		t.Fatal("the first stretch did not reach the socket")
	}
	// Hold the writer: the stretch after the pause is still queued at Stop.
	gate := make(chan struct{})
	s.writeGate = gate
	sendAll(t, st, silence(10*time.Second))
	f.waitLogged(t, "piece 1 of the recording") // the first Final of two
	sendAll(t, st, tone(500*time.Millisecond, 0.3))
	if queued() == 0 {
		t.Fatal("the tail was not queued for the socket")
	}
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	close(gate)
	// The drain delivers the tail; its (different) text joins the preview
	// while the Final runs, and the Final fails.
	for deadline := time.Now().Add(10 * time.Second); ; {
		s.mu.Lock()
		text := s.asm.text()
		s.mu.Unlock()
		if text != "" && text != before {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the tail's text never arrived: %q (before the pause %q)", text, before)
		}
		time.Sleep(10 * time.Millisecond)
	}
	evs := drain(t, st)
	if last := evs[len(evs)-1]; last.Kind != asr.EventError {
		t.Fatalf("last %+v", last)
	}
	if got := st.(asr.Salvager).Salvage(); !strings.HasPrefix(got, "partial ") || got == before {
		t.Errorf("salvaged %q, want the preview's text with the stretch after the pause (before it %q)", got, before)
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
			out, _ := c.push(fr)
			sent += len(out)
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
