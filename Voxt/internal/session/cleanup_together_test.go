package session

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// numbered is a dictation of n sentences, none like another.
func numbered(n int) string {
	var b strings.Builder
	for i := range n {
		fmt.Fprintf(&b, "第%d件事，我们下周三下午三点开个会，讨论一下预算。", i+1)
	}
	return b.String()
}

// gatedLLM answers the pieces of a dictation when the test lets them.
type gatedLLM struct {
	pieces  []compose.Piece
	answers []string
	errs    []error
	begun   chan int
	ended   chan int        // a request returned
	let     []chan struct{} // closed: the piece answers

	mu             sync.Mutex
	inFlight, most int
}

func newGatedLLM(pieces []compose.Piece, answers []string) *gatedLLM {
	g := &gatedLLM{pieces: pieces, answers: answers, errs: make([]error, len(pieces)), begun: make(chan int, len(pieces)), ended: make(chan int, len(pieces)), let: make([]chan struct{}, len(pieces))}
	for i := range g.let {
		g.let[i] = make(chan struct{})
	}
	return g
}

func (g *gatedLLM) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	last := req.Messages[len(req.Messages)-1].Content
	i := slices.IndexFunc(g.pieces, func(p compose.Piece) bool { return last == compose.UserMessage(p.Text) })
	g.mu.Lock()
	g.inFlight++
	g.most = max(g.most, g.inFlight)
	g.mu.Unlock()
	defer func() {
		g.mu.Lock()
		g.inFlight--
		g.mu.Unlock()
		g.ended <- i
	}()
	g.begun <- i
	select {
	case <-g.let[i]:
	case <-ctx.Done():
		return llm.Response{}, ctx.Err()
	}
	if g.errs[i] != nil {
		return llm.Response{}, g.errs[i]
	}
	if req.OnDelta != nil {
		req.OnDelta(g.answers[i])
	}
	return llm.Response{Text: g.answers[i]}, nil
}

// TestLongDictationTogether: a remote service is asked for the pieces side
// by side. Whichever answers first, the draft shows them in the dictation's
// order, the first piece before the second.
func TestLongDictationTogether(t *testing.T) {
	long := numbered(16)
	pieces := compose.Pieces(long)
	if len(pieces) != 2 {
		t.Fatalf("%d pieces", len(pieces))
	}
	for _, c := range []struct {
		name    string
		fail    error
		result  string
		problem bool
	}{
		{"the second answers first", nil, "一。二。", false},
		{"the second fails", errors.New("503"), "一。" + pieces[1].Text, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			g := newGatedLLM(pieces, []string{"一。", "二。"})
			g.errs[1] = c.fail
			h := newHarness(t, &fakeFile{text: long}, g, nil)
			h.remote = true
			h.speak(h.start(), 5)
			got := []int{<-g.begun, <-g.begun} // both asked while neither is answered
			slices.Sort(got)
			if !slices.Equal(got, []int{0, 1}) {
				t.Fatalf("pieces %v asked", got)
			}
			close(g.let[1])
			<-g.ended // the second answered before the first
			close(g.let[0])
			end := h.rec.wait(t)
			if end.Result != c.result || (end.Problem != nil) != c.problem {
				t.Errorf("result %q, problem %+v", end.Result, end.Problem)
			}
			for _, s := range h.rec.all() {
				if s.State == Composing && s.Draft != "" && !strings.HasPrefix(s.Draft, "一") {
					t.Errorf("a draft without the first piece: %q", s.Draft)
				}
			}
		})
	}
}

// TestLongDictationTogetherLimit: a very long dictation's pieces are asked
// partsTogether at a time.
func TestLongDictationTogetherLimit(t *testing.T) {
	long := numbered(120)
	pieces := compose.Pieces(long)
	if len(pieces) <= partsTogether {
		t.Fatalf("%d pieces, want more than %d", len(pieces), partsTogether)
	}
	g := newGatedLLM(pieces, make([]string, len(pieces)))
	for i := range g.answers {
		g.answers[i] = "答。"
	}
	h := newHarness(t, &fakeFile{text: long}, g, nil)
	h.remote = true
	h.speak(h.start(), 5)
	for range partsTogether {
		<-g.begun
	}
	select {
	case i := <-g.begun:
		t.Fatalf("piece %d asked beside %d others", i, partsTogether)
	case <-time.After(50 * time.Millisecond):
	}
	for _, ch := range g.let {
		close(ch)
	}
	end := h.rec.wait(t)
	if end.Result != strings.Repeat("答。", len(pieces)) {
		t.Errorf("result %q", end.Result)
	}
	if g.most != partsTogether {
		t.Errorf("%d at once, want %d", g.most, partsTogether)
	}
}

// connectLLM is a fakeLLM on a server (llm.Connector).
type connectLLM struct {
	*fakeLLM
	connected chan struct{}
}

func (c *connectLLM) Connect(context.Context) error {
	c.connected <- struct{}{}
	return nil
}

// TestConnectLLM: while the user speaks, the session opens the connection
// to a remote language model it is going to use, and to no other.
func TestConnectLLM(t *testing.T) {
	for _, tc := range []struct {
		name    string
		kind    Kind
		enhance bool
		remote  bool
		want    bool
	}{
		{"transcribe", store.KindTranscribe, true, true, true},
		{"transcribe, no clean-up", store.KindTranscribe, false, true, false},
		{"translate speech", store.KindTranslate, true, true, true},
		{"rewrite", store.KindRewrite, true, true, true},
		{"a server at home", store.KindTranscribe, true, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &connectLLM{&fakeLLM{text: `{"title":"Hi","content":"Hello."}`}, make(chan struct{}, 2)}
			h := newHarness(t, &fakeFile{text: "hello aura link"}, model, func(_ *Config, s *settings.Settings) {
				s.Features.Transcribe.Enhance = tc.enhance
			})
			h.remote = tc.remote
			id, err := h.m.Start(StartRequest{Kind: tc.kind, Target: slack})
			if err != nil {
				t.Fatal(err)
			}
			h.rec.waitFor(t, "recording", func(s Snapshot) bool { return s.State == Recording })
			if tc.want {
				<-model.connected // while the user speaks
			}
			h.speak(id, 3)
			h.rec.wait(t)
			if len(model.connected) > 0 {
				t.Error("connected again, or for no reason")
			}
		})
	}
}

func TestSplitTranscribeConnectsBeforeBlockedASROpenReturns(t *testing.T) {
	openRelease, releaseOpen := controlledRelease(t)
	provider := &fakeStreamer{final: "hello aura link", openGate: openRelease}
	model := &connectLLM{fakeLLM: &fakeLLM{text: "Cleaned."}, connected: make(chan struct{}, 2)}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
	})
	signals := observeSessionSignals(h)
	h.m.cfg.ResolveASR = func(context.Context, Kind, settings.Settings) ASRReadiness {
		return ASRReadiness{ASR: provider, ASRInfo: Model{ID: "fake-asr"}}
	}
	h.m.cfg.ResolveLLM = func(context.Context, Kind, settings.Settings) LLMReadiness {
		return LLMReadiness{LLM: model, LLMInfo: Model{ID: "remote-llm", Remote: true}, Outcome: LLMAvailable}
	}
	id := h.start()
	waitInitializationSignal(t, signals.asrOpenStarted, "split ASR Open to start")
	select {
	case <-model.connected:
	case <-time.After(5 * time.Second):
		t.Fatal("remote connection warmup waited for ASR Open to return")
	}
	select {
	case <-signals.asrOpenDone:
		t.Fatal("ASR Open returned while its gate was still closed")
	default:
	}
	releaseOpen()
	waitInitializationSignal(t, signals.asrOpenDone, "split ASR Open to complete")
	h.speak(id, 4)
	if end := h.rec.wait(t); end.State != Done || end.Result != "Cleaned." {
		t.Fatalf("end %+v", end)
	}
	if len(model.connected) != 0 {
		t.Fatal("remote connection warmup ran more than once")
	}
}

type tagGateLLM struct {
	started chan struct{}
	release <-chan struct{}
}

func (m *tagGateLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	request.OnDelta("<output>")
	close(m.started)
	select {
	case <-m.release:
	case <-ctx.Done():
		return llm.Response{}, ctx.Err()
	}
	request.OnDelta("cleaned output</output>")
	return llm.Response{Text: "<output>cleaned output</output>"}, nil
}

type controlledCleanupLLM struct {
	started  chan llm.Request
	finish   <-chan struct{}
	response string
	err      error
}

func (m *controlledCleanupLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	m.started <- request
	select {
	case <-m.finish:
		return llm.Response{Text: m.response}, m.err
	case <-ctx.Done():
		return llm.Response{}, ctx.Err()
	}
}

func TestCleanupTagOnlyDeltaDoesNotPublishDraft(t *testing.T) {
	release, releaseLLM := controlledRelease(t)
	model := &tagGateLLM{started: make(chan struct{}), release: release}
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, nil)
	id := h.start()
	h.speak(id, 4)
	select {
	case <-model.started:
	case <-time.After(5 * time.Second):
		t.Fatal("Cleanup did not emit its tag-only prefix")
	}
	if draft := h.m.Current().Draft; draft != "" {
		t.Fatalf("tag-only delta published a draft: %q", draft)
	}
	releaseLLM()
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "cleaned output" || len(h.text.pastes()) != 1 {
		t.Fatalf("end=%+v deliveries=%q", end, h.text.pastes())
	}
}

func TestFailedCleanupKeepsRawTextAndDeliversOnce(t *testing.T) {
	text := strings.Repeat("First sentence should be cleaned. Second sentence should also be cleaned. ", 30)
	model := &fakeLLM{
		texts: []string{"<output>First sentence cleaned.</output>", "<output>Second sentence cleaned.</output>"},
		err:   errors.New("second piece failed"), failOn: 2,
	}
	h := newHarness(t, &fakeFile{text: text}, model, nil)
	id := h.start()
	h.speak(id, 4)
	end := h.rec.wait(t)
	if end.State != Done || end.Problem == nil || end.Problem.Code != ProblemLLM || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("failed Cleanup result=%+v deliveries=%q", end, h.text.pastes())
	}
}

func TestFallbackCleanupFencesLatePrimaryDraft(t *testing.T) {
	clock := newSoftDeadlineClock()
	primaryFinish, fallbackFinish := make(chan struct{}), make(chan struct{})
	primary := &controlledCleanupLLM{
		started: make(chan llm.Request, 1), finish: primaryFinish,
		response: "primary answer", err: errors.New("primary failed after streaming"),
	}
	fallback := &controlledCleanupLLM{started: make(chan llm.Request, 1), finish: fallbackFinish, response: "Fallback result."}
	h := newHarness(t, &fakeFile{text: "spoken words"}, primary, cleanupLatencyConfig(clock, false))
	h.fallback = fallback
	id := h.start()
	h.speak(id, 4)
	primaryRequest := <-primary.started
	clock.Advance(200 * time.Millisecond)
	primaryRequest.OnDelta("<output>Primary draft.</output>")
	if snapshot := h.m.Current(); snapshot.Draft != "Primary draft." {
		t.Fatalf("primary draft was not published: %q", snapshot.Draft)
	}
	close(primaryFinish)
	fallbackRequest := <-fallback.started
	if snapshot := h.m.Current(); snapshot.Draft != "" {
		t.Fatalf("fallback did not clear the failed attempt's draft: %q", snapshot.Draft)
	}
	primaryRequest.OnDelta(" stale primary callback")
	if snapshot := h.m.Current(); snapshot.Draft != "" {
		t.Fatalf("late primary callback modified the snapshot: %q", snapshot.Draft)
	}
	fallbackRequest.OnDelta("Fallback cleaned text.")
	close(fallbackFinish)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "Fallback result." || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("end=%+v deliveries=%q", end, h.text.pastes())
	}
	if snapshot := h.m.Current(); snapshot.Result != "Fallback result." || snapshot.Draft != "Fallback result." {
		t.Fatalf("late primary callback changed final output: %+v", snapshot)
	}
}

func TestSuccessfulUnchangedLLMOutputIsDelivered(t *testing.T) {
	const raw = "spoken words"
	h := newHarness(t, &fakeFile{text: raw}, &fakeLLM{text: raw}, nil)
	id := h.start()
	h.speak(id, 4)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != raw || len(h.text.pastes()) != 1 || h.text.pastes()[0] != raw {
		t.Fatalf("unchanged Cleanup output end=%+v deliveries=%q", end, h.text.pastes())
	}
}

func TestSoftDeadlineRawFallbackWinsOverLateCleanup(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newCleanupGateLLM("partial model output", "late output", "cleaned result")
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 4)
	waitCleanupRequest(t, model, "cleanup request before soft deadline")
	clock.Advance(cleanupSoftDeadlineTestDuration)
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "soft-deadline Raw result", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.Result != "spoken words" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("soft-deadline result=%+v deliveries=%q", end, h.text.pastes())
	}
	model.Release()
	waitCleanupSignal(t, model.returned, "late cleanup response after Raw selection")
}
