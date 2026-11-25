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
