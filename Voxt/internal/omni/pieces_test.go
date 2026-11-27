package omni

import (
	"bytes"
	"context"
	"encoding/base64"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
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

type pieceGateRequest struct {
	eventKind   string
	sampleCount int
	release     chan struct{}
}

type pieceTestGate struct {
	server    *httptest.Server
	pending   chan pieceGateRequest
	cancelled chan pieceGateRequest
	shutdown  chan struct{}
}

func newPieceTestGate(t *testing.T) *pieceTestGate {
	t.Helper()
	gate := &pieceTestGate{
		pending:   make(chan pieceGateRequest, 16),
		cancelled: make(chan pieceGateRequest, 16),
		shutdown:  make(chan struct{}),
	}
	gateHandler := http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		sampleCount, _ := strconv.Atoi(request.Header.Get("X-Omnitest-Samples"))
		gateRequest := pieceGateRequest{
			eventKind: request.Header.Get("X-Omnitest-Event"), sampleCount: sampleCount, release: make(chan struct{}),
		}
		select {
		case gate.pending <- gateRequest:
		case <-request.Context().Done():
			gate.cancelled <- gateRequest
			return
		case <-gate.shutdown:
			return
		}
		select {
		case <-gateRequest.release:
			w.WriteHeader(http.StatusNoContent)
		case <-request.Context().Done():
			gate.cancelled <- gateRequest
		case <-gate.shutdown:
			return
		}
	})
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gate.server = httptest.NewUnstartedServer(gateHandler)
	gate.server.Listener = listener
	gate.server.Start()
	t.Cleanup(func() {
		close(gate.shutdown)
		gate.server.Close()
	})
	return gate
}

func (gate *pieceTestGate) next(t *testing.T) pieceGateRequest {
	t.Helper()
	select {
	case request := <-gate.pending:
		return request
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for a gated request")
		return pieceGateRequest{}
	}
}

func (gate *pieceTestGate) waitForFinal(t *testing.T) pieceGateRequest {
	t.Helper()
	for {
		request := gate.next(t)
		if request.eventKind == "final" {
			return request
		} else if request.eventKind == "vad" {
			request.allow()
		} else {
			t.Fatalf("unexpected gated request %q", request.eventKind)
		}
	}
}

func (gate *pieceTestGate) waitForCancellation(t *testing.T, eventKind string) {
	t.Helper()
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	for {
		select {
		case request := <-gate.cancelled:
			if request.eventKind == eventKind {
				return
			}
		case <-deadline.C:
			t.Fatalf("timed out waiting for gated %s cancellation", eventKind)
			return
		}
	}
}

func (request pieceGateRequest) allow() { close(request.release) }

// sendAll sends pcm in 100 ms frames, as fast as it goes.
func sendAll(t *testing.T, st asr.Stream, pcm []byte) {
	t.Helper()
	for _, fr := range asr.Frames(pcm, 100*time.Millisecond, sampleRate) {
		if err := st.Send(fr); err != nil {
			t.Fatal(err)
		}
	}
}

// lookedAtAll returns once a live stream's pieces loop has looked at all
// the audio sent. The loop takes one wake at a time, at the start of a
// round: the second wake goes in once it took the first (a round that
// begins with all the audio), the third once that round ended.
func lookedAtAll(t *testing.T, st asr.Stream) {
	t.Helper()
	pc := st.(*live).pc
	for range 3 {
		select {
		case pc.wake <- struct{}{}:
		case <-time.After(10 * time.Second):
			t.Fatal("the pieces loop stopped")
		}
	}
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
	pcm := speechAndPauses(5) // 8.5 s: the closing pause is no cut, so a fifth stretch of speech for a fourth piece
	sendAll(t, st, pcm)
	f.waitLogged(t, "piece 4 of the recording")
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
	// The rest starts after the recording: it gets the session for what its
	// server prefilled, and not the preview's text.
	if r := finals[4]; r.Fields["realtime_session"] != "fake-session" || r.Fields["guessed_text"] != "" {
		t.Errorf("the rest: %+v", r.Fields)
	}
	for i, r := range finals[:4] {
		if r.Fields["realtime_session"] != "" || r.Fields["guessed_text"] != "" {
			t.Errorf("piece %d: %+v", i+1, r.Fields)
		}
	}
	// Each piece's end is where the server starts the Final's prefill.
	st.Close()
	f.rt.Close() // the server records its live sessions before it exits
	var ends []string
	at = 0
	for _, r := range finals[:4] {
		at += r.Samples
		ends = append(ends, strconv.Itoa(at))
	}
	if l := f.records("live"); len(l) != 1 || l[0].Fields["final_start"] != strings.Join(ends, ",") {
		t.Errorf("live %+v, want the Final's starts %v", l, ends)
	}
	if !f.logged("4 pieces while it recorded") {
		t.Error("the Final's line does not say what the pieces did")
	}
}

// TestMossPiecesSkipHotwords: intermediate MOSS pieces keep the plain-text
// instruction; only the post-stop tail adds the dictionary.
func TestMossPiecesSkipHotwords(t *testing.T) {
	f, a := mossWithVAD(t, omnitest.Opts{Text: "p{n}"})
	a.spec.ASR.PiecesAfter, a.spec.ASR.PieceMin, a.spec.ASR.PieceSeconds, a.spec.ASR.PieceMax = 2*time.Second, time.Second, 1500*time.Millisecond, 3*time.Second
	a.spec.ASR.PauseMin = 150 * time.Millisecond
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate, Lang: "en", Hints: []string{"Zephyria"}})
	if err != nil {
		t.Fatal(err)
	}
	sendAll(t, st, speechAndPauses(2))
	f.waitLogged(t, "piece 1 of the recording")
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	if evs := drain(t, st); len(evs) == 0 || evs[len(evs)-1].Kind != asr.EventFinal {
		t.Fatalf("events %+v", evs)
	}
	finals := f.records("final")
	instruction := again(ASRMoss).ASR.Instruction
	if len(finals) != 2 || finals[0].Fields["prompt"] != instruction || finals[1].Fields["prompt"] != instruction+" Hotwords: Zephyria" {
		t.Errorf("finals %+v", finals)
	}
}

// TestLivePiecesEvents: every piece recognised while recording reports the
// final text so far (asr.EventPiece), growing, before the Final,
// each a prefix of it, so the session can start its clean-up of the
// finished stretches.
func TestLivePiecesEvents(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "p{n}"}, nil)
	a := piecesASR(t, f)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate, Lang: "zh"})
	if err != nil {
		t.Fatal(err)
	}
	pcm := speechAndPauses(5)
	sendAll(t, st, pcm)
	f.waitLogged(t, "piece 4 of the recording")
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := drain(t, st)
	final := evs[len(evs)-1]
	if final.Kind != asr.EventFinal {
		t.Fatalf("last %+v", final)
	}
	var pieces []string
	for _, ev := range evs[:len(evs)-1] {
		if ev.Kind == asr.EventPiece {
			pieces = append(pieces, ev.Text)
		}
	}
	if want := []string{"p1", "p1 p2", "p1 p2 p3", "p1 p2 p3 p4"}; !slices.Equal(pieces, want) {
		t.Fatalf("pieces %q, want %q", pieces, want)
	}
	for _, p := range pieces {
		if !strings.HasPrefix(final.Text, p) {
			t.Errorf("%q is not a prefix of the final %q", p, final.Text)
		}
	}
	// Without a dictionary the rest still has its windows to reuse.
	if r := f.records("final"); r[len(r)-1].Fields["realtime_session"] != "fake-session" {
		t.Errorf("the rest without a dictionary: %+v", r[len(r)-1].Fields)
	}
}

// TestLivePiecesCoverEverySample: the pieces recognised while recording
// and the Final at Finish together carry every sample that was sent, in
// order, from the first to the last: the tail left when the user stops
// (speech that lasts to the last sample) is recognised too.
func TestLivePiecesCoverEverySample(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "p{n}", RecordAudio: true}, nil)
	a := piecesASR(t, f)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	pcm := append(speechAndPauses(3), tone(1200*time.Millisecond, 0.3)...) // 6.3 s, ends in speech
	sendAll(t, st, pcm)
	f.waitLogged(t, "piece 1 of the recording")
	if err := st.Finish(); err != nil { // the user stops the moment the audio ends
		t.Fatal(err)
	}
	evs := drain(t, st)
	if last := evs[len(evs)-1]; last.Kind != asr.EventFinal {
		t.Fatalf("last %+v", last)
	}
	var got []byte
	for _, r := range f.records("final") {
		b, err := base64.StdEncoding.DecodeString(r.Audio)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, b...)
	}
	if !bytes.Equal(got, pcm) {
		t.Errorf("the Final requests carried %d samples; want every one of the recording's %d, from its first sample to its last", len(got)/2, len(pcm)/2)
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
	lookedAtAll(t, st)
	_ = st.Finish()
	evs := drain(t, st)
	if last := evs[len(evs)-1]; last.Kind != asr.EventFinal || last.Text != "p1" {
		t.Fatalf("final %+v", last)
	}
	if finals := f.records("final"); len(finals) != 1 || finals[0].Samples != len(pcm)/2 {
		t.Fatalf("finals %+v", finals)
	}
}

// note (chenye): Final must not wait for planner shutdown when no piece is active.
func TestFinishCancelsPendingPiecePlanning(t *testing.T) {
	gate := newPieceTestGate(t)
	f := newFake(t, omnitest.Opts{Text: "p{n}", RecordAudio: true, GateURL: gate.server.URL}, nil)
	st, err := piecesASR(t, f).Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(st.Close)
	pcm := speechAndPauses(3)
	if err := st.Send(pcm); err != nil {
		t.Fatal(err)
	}
	if request := gate.next(t); request.eventKind != "vad" {
		t.Fatalf("planning request is %q", request.eventKind)
	}
	stream := st.(*live)
	stream.mu.Lock()
	if stream.pc.activePieceDone != nil {
		stream.mu.Unlock()
		t.Fatal("a piece was active before Finish")
	}
	// note (chenye): Keep the planner signal pending to distinguish it from active transcription.
	plannerExitSignal := make(chan struct{})
	stream.pc.exited = plannerExitSignal
	stream.mu.Unlock()
	defer close(plannerExitSignal)
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	gate.waitForCancellation(t, "vad")
	finalRequest := gate.waitForFinal(t)
	if finalRequest.sampleCount != len(pcm)/2 {
		t.Fatalf("Final has %d samples, want all %d", finalRequest.sampleCount, len(pcm)/2)
	}
	finalRequest.allow()
	events := drain(t, st)
	if last := events[len(events)-1]; last.Kind != asr.EventFinal || last.Text != "p1" {
		t.Fatalf("final %+v", last)
	}
	select {
	case <-plannerExitSignal:
		t.Fatal("Final waited until the piece planner exited")
	default:
	}
	for _, event := range events {
		if event.Kind == asr.EventPiece {
			t.Fatalf("unclaimed planning emitted a piece: %+v", events)
		}
	}
	finals := f.records("final")
	if len(finals) != 1 {
		t.Fatalf("Final requests %+v", finals)
	}
	finalAudio, err := base64.StdEncoding.DecodeString(finals[0].Audio)
	if err != nil || !bytes.Equal(finalAudio, pcm) {
		t.Fatalf("Final audio does not cover the recording: requests=%d bytes=%d error=%v", len(finals), len(finalAudio), err)
	}
	if f.logged("looking for a pause") {
		t.Error("planning cancellation was logged as a VAD failure")
	}
	st.Close()
	f.rt.Close()
	if sessions := f.records("live"); len(sessions) != 1 || sessions[0].Fields["final_start"] != "" {
		t.Fatalf("unclaimed planning changed Final's prefill position: %+v", sessions)
	}
}

func TestClaimedPieceHandoffPreservesAudioAndFailureFallback(t *testing.T) {
	for _, testCase := range []struct {
		name      string
		failFirst bool
		wantText  string
	}{
		{name: "success", wantText: "p1 p2"},
		{name: "failure", failFirst: true, wantText: "p2"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			gate := newPieceTestGate(t)
			options := omnitest.Opts{Text: "p{n}", RecordAudio: true, GateURL: gate.server.URL}
			if testCase.failFirst {
				options.FailAt = 1
			}
			f := newFake(t, options, nil)
			st, err := piecesASR(t, f).Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(st.Close)
			pcm := speechAndPauses(3)
			if err := st.Send(pcm); err != nil {
				t.Fatal(err)
			}
			pieceRequest := gate.waitForFinal(t)
			if pieceRequest.sampleCount <= 0 || pieceRequest.sampleCount >= len(pcm)/2 {
				t.Fatalf("claimed piece has %d samples of %d", pieceRequest.sampleCount, len(pcm)/2)
			}
			if err := st.Finish(); err != nil {
				t.Fatal(err)
			}
			pieceRequest.allow()
			tailRequest := gate.waitForFinal(t)
			wantTail := len(pcm)/2 - pieceRequest.sampleCount
			if testCase.failFirst {
				wantTail = len(pcm) / 2
			}
			if tailRequest.sampleCount != wantTail {
				t.Fatalf("tail has %d samples, want %d", tailRequest.sampleCount, wantTail)
			}
			tailRequest.allow()
			events := drain(t, st)
			if last := events[len(events)-1]; last.Kind != asr.EventFinal || last.Text != testCase.wantText {
				t.Fatalf("final %+v", last)
			}
			pieceEvents := 0
			for _, event := range events {
				if event.Kind == asr.EventPiece {
					pieceEvents++
				}
			}
			wantPieceEvents := 1
			if testCase.failFirst {
				wantPieceEvents = 0
			}
			if pieceEvents != wantPieceEvents {
				t.Fatalf("committed piece events: %+v", events)
			}
			finals := f.records("final")
			if len(finals) != 2 {
				t.Fatalf("Final requests %+v", finals)
			}
			if testCase.failFirst {
				fallbackAudio, err := base64.StdEncoding.DecodeString(finals[1].Audio)
				if err != nil || !bytes.Equal(fallbackAudio, pcm) {
					t.Fatalf("fallback audio does not cover the uncommitted recording: %v", err)
				}
			} else {
				if finals[1].Fields["language"] != "Chinese" {
					t.Fatalf("detected language was not propagated: %+v", finals[1].Fields)
				}
				var transcribed []byte
				for _, request := range finals {
					requestAudio, err := base64.StdEncoding.DecodeString(request.Audio)
					if err != nil {
						t.Fatal(err)
					}
					transcribed = append(transcribed, requestAudio...)
				}
				if !bytes.Equal(transcribed, pcm) {
					t.Fatalf("piece and tail do not cover audio exactly: got %d bytes, want %d", len(transcribed), len(pcm))
				}
			}
		})
	}
}

func TestSessionCancellationAndCloseReleasePiecePlanningVAD(t *testing.T) {
	for _, testCase := range []struct {
		name        string
		closeStream bool
	}{
		{name: "session cancellation"},
		{name: "Close", closeStream: true},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			gate := newPieceTestGate(t)
			f := newFake(t, omnitest.Opts{GateURL: gate.server.URL}, nil)
			parentContext, cancelParent := context.WithCancel(context.Background())
			defer cancelParent()
			st, err := piecesASR(t, f).Open(parentContext, asr.StreamRequest{SampleRate: sampleRate})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(st.Close)
			if err := st.Send(speechAndPauses(3)); err != nil {
				t.Fatal(err)
			}
			if request := gate.next(t); request.eventKind != "vad" {
				t.Fatalf("planning request is %q", request.eventKind)
			}
			if testCase.closeStream {
				st.Close()
			} else {
				cancelParent()
			}
			gate.waitForCancellation(t, "vad")
			stream := st.(*live)
			select {
			case <-stream.pc.exited:
			case <-time.After(10 * time.Second):
				t.Fatal("piece planner retained its VAD request")
			}
			events := drain(t, st)
			if !testCase.closeStream && (len(events) == 0 || events[len(events)-1].Kind != asr.EventError) {
				t.Fatalf("session cancellation events %+v", events)
			} else if testCase.closeStream {
				for _, event := range events {
					if event.Kind == asr.EventPiece || event.Kind == asr.EventFinal {
						t.Fatalf("Close emitted a late piece or Final: %+v", events)
					}
				}
			}
			f.rt.Close()
			if got := len(f.records("shutdown")); got != 2 {
				t.Fatalf("runtime shut down %d servers, want ASR and VAD", got)
			}
		})
	}
}

func TestEnableEarlyPiecesRechecksAudioAlreadySent(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "p{n}"}, nil)
	a := piecesASR(t, f)
	a.spec.ASR.PiecesAfter, a.spec.ASR.PieceMin, a.spec.ASR.PieceSeconds, a.spec.ASR.PieceMax = 60*time.Second, 10*time.Second, 15*time.Second, 25*time.Second
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	pcm := speechAndPauses(12) // 20.4 s: below the original policy, above EarlyPieces' threshold.
	sendAll(t, st, pcm)
	if finals := f.records("final"); len(finals) != 0 {
		t.Fatalf("the normal policy cut early: %+v", finals)
	}
	enabler, ok := st.(asr.EarlyPiecesEnabler)
	if !ok || !enabler.EnableEarlyPieces() {
		t.Fatal("Qwen3 stream refused safe late EarlyPieces activation")
	}
	f.waitLogged(t, "piece 1 of the recording")
	if a.spec.ASR.PiecesAfter != 60*time.Second {
		t.Fatalf("late activation changed the shared ASR policy to %v", a.spec.ASR.PiecesAfter)
	}
}

func TestEnableEarlyPiecesKeepsClaimedPieceBoundaries(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "p{n}"}, nil)
	a := piecesASR(t, f)
	a.spec.ASR.PiecesAfter, a.spec.ASR.PieceMin, a.spec.ASR.PieceSeconds, a.spec.ASR.PieceMax = 25*time.Second, 10*time.Second, 15*time.Second, 25*time.Second
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	sendAll(t, st, speechAndPauses(15)) // 25.5 s reaches the existing policy.
	f.waitLogged(t, "piece 1 of the recording")
	enabler, ok := st.(asr.EarlyPiecesEnabler)
	if !ok {
		t.Fatal("Qwen3 stream does not expose EarlyPieces activation")
	}
	if enabler.EnableEarlyPieces() {
		t.Fatal("late activation changed segmentation after a piece had been claimed")
	}
}

func TestEnableEarlyPiecesIsUnavailableForLiveResult(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, nil)
	a := f.asr(t, ASRAccurate)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate, LiveResult: true})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	enabler, ok := st.(asr.EarlyPiecesEnabler)
	if !ok || enabler.EnableEarlyPieces() {
		t.Fatal("LiveResult stream accepted an unsupported EarlyPieces policy")
	}
}

func TestEnableEarlyPiecesRejectsOtherModelKinds(t *testing.T) {
	stream := &live{a: &ASR{spec: Spec{Kind: kindWhisper}}, pc: &pieces{}}
	if stream.EnableEarlyPieces() {
		t.Fatal("Whisper accepted Qwen3's EarlyPieces policy")
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
		// A piece stalls first: its server is killed, the Final finds it gone and stalls again on a new one.
		{"stall", omnitest.Opts{Mode: omnitest.ModeStall}, "stopped answering"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFake(t, tc.o, func(c *Config) { c.StallTimeout = 300 * time.Millisecond })
			a := piecesASR(t, f)
			st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
			if err != nil {
				t.Fatal(err)
			}
			sendAll(t, st, speechAndPauses(4))
			select { // the preview's text
			case ev := <-st.Events():
				if ev.Kind != asr.EventPartial {
					t.Fatalf("before Finish: %+v", ev)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("no live partial")
			}
			f.waitLogged(t, "a piece of the recording failed")
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
		atEnd  bool
		at     int
		ok     bool
	}{
		{nil, 100, 10, false, 50, true}, // no speech
		{[]vad.Segment{{Start: 0, End: 40}, {Start: 60, End: 100}}, 100, 10, false, 50, true},                                            // between
		{[]vad.Segment{{Start: 0, End: 40}, {Start: 50, End: 100}}, 100, 20, false, 0, false},                                            // too short
		{[]vad.Segment{{Start: 30, End: 40}, {Start: 45, End: 80}}, 100, 10, false, 15, true},                                            // before the first (30) beats after the last (20)
		{[]vad.Segment{{Start: 0, End: 10}, {Start: 20, End: 30}, {Start: 35, End: 60}, {Start: 90, End: 100}}, 100, 5, false, 75, true}, // the longest of three
		{[]vad.Segment{{Start: 0, End: 40}, {Start: 45, End: 60}}, 100, 3, false, 80, true},                                              // after the last, with audio after it
		{[]vad.Segment{{Start: 0, End: 40}, {Start: 45, End: 60}}, 100, 3, true, 42, true},                                               // at the end: the closing pause is not a cut
		{nil, 100, 10, true, 0, false}, // no speech yet at the end: wait
	} {
		at, ok := longestPause(c.segs, c.n, c.min, c.atEnd)
		if at != c.at || ok != c.ok {
			t.Errorf("%+v: %d %v", c, at, ok)
		}
	}
}
