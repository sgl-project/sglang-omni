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

// speechAndPauses is n stretches of a 1.2 s tone, each followed by a
// 0.5 s pause.
func speechAndPauses(n int) []byte {
	var pcm []byte
	for range n {
		pcm = append(append(pcm, tone(1200*time.Millisecond, 0.3)...), silence(500*time.Millisecond)...)
	}
	return pcm
}

// piecesASR is the fake's ASR with Silero and pieces at a test's scale:
// after 2 s, cut 1–3 s in, from 1.5 s pending, in pauses of 150 ms.
func piecesASR(t *testing.T, f *fake) *ASR {
	t.Helper()
	v, err := NewVAD(f.rt, f.model(VADSilero))
	if err != nil {
		t.Fatal(err)
	}
	a := f.asr(t, ASRAccurate).WithVAD(v)
	a.spec.ASR.PiecesAfter, a.spec.ASR.PieceMin, a.spec.ASR.PieceSeconds, a.spec.ASR.PieceMax = 2*time.Second, time.Second, 1500*time.Millisecond, 3*time.Second
	a.spec.ASR.PauseMin = 150 * time.Millisecond
	return a
}

// sendAll sends pcm in 100 ms frames, as fast as it goes.
func sendAll(t *testing.T, st asr.Stream, pcm []byte) {
	t.Helper()
	for _, fr := range asr.Frames(pcm, 100*time.Millisecond, sampleRate) {
		if err := st.Send(fr); err != nil {
			t.Fatal(err)
		}
	}
}

// piecesDone is how many pieces a live stream recognised.
func piecesDone(st asr.Stream) int {
	s := st.(*live)
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.pc.texts)
}

// TestLivePieces: a long live session is recognised in pieces cut in its
// pauses while it records; Finish recognises the rest only, and the text
// joins them in order.
func TestLivePieces(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "p{n}"}, nil)
	a := piecesASR(t, f)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate, Lang: "zh", Hints: []string{"VoxtNew"}})
	if err != nil {
		t.Fatal(err)
	}
	pcm := speechAndPauses(4) // 6.8 s
	sendAll(t, st, pcm)
	eventually(t, "four pieces", func() bool { return piecesDone(st) == 4 })
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := drain(t, st)
	last := evs[len(evs)-1]
	if last.Kind != asr.EventFinal || last.Text != "p1 p2 p3 p4 p5" || last.Result.Duration != asr.Duration(pcm, sampleRate) {
		t.Fatalf("final %+v", last)
	}
	finals := f.records("final")
	if len(finals) != 5 {
		t.Fatalf("%d Final requests, want 4 pieces and the rest", len(finals))
	}
	period := samplesOf(1700 * time.Millisecond)
	at := 0
	for i, r := range finals[:4] {
		if r.Fields["language"] != "zh" || r.Fields["prompt"] != "VoxtNew" {
			t.Errorf("piece %d: %+v", i+1, r.Fields)
		}
		at += r.Samples
		if in := at % period; in < samplesOf(1200*time.Millisecond) {
			t.Errorf("piece %d ends %d samples into a stretch of speech", i+1, in)
		}
	}
	if !f.logged("4 pieces while it recorded") {
		t.Error("the Final's line does not say what the pieces did")
	}
}

// TestLiveNoPiecesWhenShort: a recording shorter than PiecesAfter keeps
// one Final over everything.
func TestLiveNoPiecesWhenShort(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "p{n}"}, nil)
	a := piecesASR(t, f)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	pcm := speechAndPauses(1)
	sendAll(t, st, pcm)
	time.Sleep(200 * time.Millisecond)
	_ = st.Finish()
	evs := drain(t, st)
	if last := evs[len(evs)-1]; last.Kind != asr.EventFinal || last.Text != "p1" {
		t.Fatalf("final %+v", last)
	}
	if finals := f.records("final"); len(finals) != 1 || finals[0].Samples != len(pcm)/2 {
		t.Fatalf("finals %+v", finals)
	}
}

// TestLiveFinalFailsWithText: a Final that fails (an error, a stall)
// ends the stream with the error, and Salvage gives the text the live
// preview had, for the session to go on with.
func TestLiveFinalFailsWithText(t *testing.T) {
	for _, tc := range []struct {
		name string
		o    omnitest.Opts
		want string
	}{
		{"error", omnitest.Opts{FailFrom: 1}, "transcription_failed"},
		{"error after two pieces", omnitest.Opts{FailFrom: 3}, "transcription_failed"},
		// A piece stalls first: its server is killed, and the Final finds it gone.
		{"stall", omnitest.Opts{Mode: omnitest.ModeStall}, "exited"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFake(t, tc.o, func(c *Config) { c.StallTimeout = 300 * time.Millisecond })
			a := piecesASR(t, f)
			st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
			if err != nil {
				t.Fatal(err)
			}
			sendAll(t, st, speechAndPauses(4))
			eventually(t, "the preview's text", func() bool { return st.(asr.Salvager).Salvage() != "" })
			time.Sleep(300 * time.Millisecond)
			_ = st.Finish()
			evs := drain(t, st)
			last := evs[len(evs)-1]
			if last.Kind != asr.EventError || !strings.Contains(last.Err.Error(), tc.want) {
				t.Fatalf("last %+v", last)
			}
			if got := st.(asr.Salvager).Salvage(); !strings.HasPrefix(got, "partial") {
				t.Errorf("salvaged %q", got)
			}
			if !f.logged("Final of 6.8s of audio") || !f.logged("memory") {
				t.Error("the failed Final's line lacks the audio or the memory")
			}
		})
	}
}

// TestSalvage: the pieces' text wins when it covers at least what the
// preview heard (a preview that stopped early), else the preview's.
func TestSalvage(t *testing.T) {
	s := &live{pc: &pieces{done: 200, texts: []string{"one.", "two"}}}
	s.asm.apply(1, 0, "preview", true)
	s.previewed = 100
	if got := s.Salvage(); got != "one. two" {
		t.Errorf("%q", got)
	}
	s.previewed = 300
	if got := s.Salvage(); got != "preview" {
		t.Errorf("%q", got)
	}
	s.pc.texts = nil
	if got := s.Salvage(); got != "preview" {
		t.Errorf("%q", got)
	}
}

// TestSlowFinal: a Final that takes longer than the stall watch, with
// the server's heartbeat coming, succeeds: progress, not a wall clock.
func TestSlowFinal(t *testing.T) {
	f := newFake(t, omnitest.Opts{Delay: time.Second, Text: "slow"}, func(c *Config) { c.StallTimeout = 300 * time.Millisecond })
	res, err := f.asr(t, ASRFast).Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3)})
	if err != nil || res.Text != "slow" {
		t.Fatalf("%+v %v", res, err)
	}
}

func TestLongestPause(t *testing.T) {
	for _, c := range []struct {
		segs   []vad.Segment
		n, min int
		at     int
		ok     bool
	}{
		{nil, 100, 10, 50, true}, // no speech
		{[]vad.Segment{{Start: 0, End: 40}, {Start: 60, End: 100}}, 100, 10, 50, true},                                            // between
		{[]vad.Segment{{Start: 0, End: 40}, {Start: 50, End: 100}}, 100, 20, 0, false},                                            // too short
		{[]vad.Segment{{Start: 30, End: 40}, {Start: 45, End: 80}}, 100, 10, 15, true},                                            // before the first (30) beats after the last (20)
		{[]vad.Segment{{Start: 0, End: 10}, {Start: 20, End: 30}, {Start: 35, End: 60}, {Start: 90, End: 100}}, 100, 5, 75, true}, // the longest of three
	} {
		at, ok := longestPause(c.segs, c.n, c.min)
		if at != c.at || ok != c.ok {
			t.Errorf("%+v: %d %v", c, at, ok)
		}
	}
}
