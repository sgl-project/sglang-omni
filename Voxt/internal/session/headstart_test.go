package session

import (
	"errors"
	"fmt"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// headSession is a dictation to a remote language model. The user speaks
// until pause.
type headSession struct {
	*harness
	id       string
	streamer *fakeStreamer
	sends    chan int // the number of each chunk sent
	speaking atomic.Bool
}

func newHeadSession(t *testing.T, streamer *fakeStreamer, model LLM, set func(*headSession, *settings.Settings)) *headSession {
	t.Helper()
	h := &headSession{streamer: streamer, sends: make(chan int, 1)}
	h.speaking.Store(true)
	streamer.onSend = func(i int) {
		select {
		case <-h.sends:
		default:
		}
		h.sends <- i
	}
	h.harness = newHarness(t, streamer, model, func(_ *Config, s *settings.Settings) {
		s.Audio.VAD.Off = true
		s.Features.Transcribe.Enhance = true
		if set != nil {
			set(h, s)
		}
	})
	h.remote = true
	h.audio.speaks = func(int) bool { return h.speaking.Load() }
	return h
}

// pause starts the recording and, once a second of speech was heard, makes
// the user silent.
func (h *headSession) pause() {
	h.t.Helper()
	h.id = h.start()
	h.sent(10)
	h.speaking.Store(false)
}

// sent waits until the speech model was sent n chunks.
func (h *headSession) sent(n int) {
	h.t.Helper()
	late := time.After(5 * time.Second)
	for {
		select {
		case i := <-h.sends:
			if i >= n {
				return
			}
		case <-late:
			h.t.Fatalf("not %d chunks sent", n)
		}
	}
}

func givenUp(t *testing.T, call *controlledLiveCall, what string) {
	t.Helper()
	select {
	case <-call.ctx.Done():
	case <-time.After(5 * time.Second):
		t.Fatalf("%s was not given up", what)
	}
}

// TestHeadStart: a pause asks about the live text while the recording goes
// on. At the Final that request's answer is taken when it says the same,
// with a word put right when it nearly does, and given up when it does not.
func TestHeadStart(t *testing.T) {
	const live = "we ship the new build on friday"
	for _, tc := range []struct {
		name, final string
		// end lets the requests answer, after Stop.
		end    func(t *testing.T, h *headSession, early *controlledLiveCall)
		result string
		asks   int
	}{
		{"the same", live, func(t *testing.T, h *headSession, early *controlledLiveCall) {
			h.rec.waitFor(t, "the answer so far", func(s Snapshot) bool { return s.State == Composing && s.Draft == "We ship" })
			early.complete(t, "We ship the new build on Friday.", nil)
		}, "We ship the new build on Friday.", 1},
		{"other punctuation", "We ship the new build, on Friday.", func(t *testing.T, h *headSession, early *controlledLiveCall) {
			early.complete(t, "We ship the new build on Friday.", nil)
		}, "We ship the new build on Friday.", 1},
		{"another word, the early answer first", "we ship the blue build on friday", func(t *testing.T, h *headSession, early *controlledLiveCall) {
			fresh := waitLiveCall(t, h.llm.(*controlledLiveLLM), "the Final's request")
			early.complete(t, "We ship the new build on Friday.", nil)
			givenUp(t, fresh, "the Final's request")
		}, "We ship the blue build on Friday.", 2},
		{"another word, the Final's answer first", "we ship the blue build on friday", func(t *testing.T, h *headSession, early *controlledLiveCall) {
			fresh := waitLiveCall(t, h.llm.(*controlledLiveLLM), "the Final's request")
			fresh.complete(t, "The blue build ships on Friday.", nil)
			givenUp(t, early, "the early request")
		}, "The blue build ships on Friday.", 2},
		{"the early request fails", live + ".", func(t *testing.T, h *headSession, early *controlledLiveCall) {
			early.complete(t, "", errors.New("503"))
			fresh := waitLiveCall(t, h.llm.(*controlledLiveLLM), "the Final's request")
			fresh.complete(t, "We ship on Friday.", nil)
		}, "We ship on Friday.", 2},
		{"other words", "please call the lawyer about the lease tomorrow", func(t *testing.T, h *headSession, early *controlledLiveCall) {
			fresh := waitLiveCall(t, h.llm.(*controlledLiveLLM), "the Final's request")
			if in := inputOf(fresh.request); in != "please call the lawyer about the lease tomorrow" {
				t.Errorf("asked about %q", in)
			}
			fresh.complete(t, "Please call the lawyer about the lease tomorrow.", nil)
			givenUp(t, early, "the early request")
		}, "Please call the lawyer about the lease tomorrow.", 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := newControlledLiveLLM(false)
			t.Cleanup(model.close)
			h := newHeadSession(t, &fakeStreamer{partials: []string{live}, final: tc.final}, model, nil)
			h.pause()
			early := waitLiveCall(t, model, "a request at the pause")
			if in := inputOf(early.request); in != live {
				t.Errorf("asked about %q", in)
			}
			if s := h.rec.all(); s[len(s)-1].State != Recording {
				t.Errorf("asked in state %v", s[len(s)-1].State)
			}
			early.sendDelta(t, "We ship")
			h.m.Stop(h.id)
			tc.end(t, h, early)
			if end := h.rec.wait(t); end.Result != tc.result || end.Problem != nil {
				t.Errorf("result %q, problem %+v, want %q", end.Result, end.Problem, tc.result)
			}
			if n := model.requestCount(); n != tc.asks {
				t.Errorf("%d requests, want %d", n, tc.asks)
			}
		})
	}
}

// TestHeadStartNothingEarly: nothing is asked before Stop when the user
// turned that off, or the service limited the rate lately.
func TestHeadStartNothingEarly(t *testing.T) {
	for _, tc := range []struct {
		name  string
		rests bool
		set   func(*headSession, *settings.Settings)
	}{
		{"cleaned after Stop", false, func(_ *headSession, s *settings.Settings) { s.Features.Transcribe.CleanAfterStop = true }},
		{"the service rests", true, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "Cleaned."}
			h := newHeadSession(t, &fakeStreamer{partials: []string{"we ship"}, final: "we ship"}, model, tc.set)
			if tc.rests {
				h.m.noteRate("deepseek", llm.Response{RateLimited: true}, nil)
			}
			h.pause()
			h.sent(60)
			if n := len(model.requests()); n != 0 {
				t.Errorf("%d requests while recording", n)
			}
			h.m.Stop(h.id)
			if end := h.rec.wait(t); end.Result != "Cleaned." || len(model.requests()) != 1 {
				t.Errorf("result %q, %d requests", end.Result, len(model.requests()))
			}
		})
	}
}

// TestHeadStartAsks: a live text that keeps changing while the user is
// silent is asked about headAsks times at most, and again at Stop.
func TestHeadStartAsks(t *testing.T) {
	partials := make([]string, 400)
	for i := range partials {
		partials[i] = fmt.Sprintf("we ship build %d", max(i, 20))
	}
	model := &fakeLLM{text: "Cleaned."}
	h := newHeadSession(t, &fakeStreamer{partials: partials, final: partials[len(partials)-1]}, model, nil)
	h.pause()
	h.sent(len(partials))
	if n := len(model.requests()); n != headAsks {
		t.Errorf("%d requests while recording, want %d", n, headAsks)
	}
	h.m.Stop(h.id)
	h.rec.wait(t)
	reqs := model.requests()
	if !slices.ContainsFunc(reqs[headAsks:], func(r llm.Request) bool { return inputOf(r) == h.streamer.final }) {
		t.Errorf("%d requests, none about the final text", len(reqs))
	}
}

// newHeadRun is a run whose head start is on, outside a session.
func newHeadRun(t *testing.T, model LLM) *run {
	r := newEarlyUnitRun(t, model, nil)
	r.svc.LLMInfo.Remote = true
	if r.startHead(); !r.head.enabled {
		t.Fatal("no head start")
	}
	return r
}

// TestHeadStartKeeps: the newest headKeep texts are asked about at a time,
// and a text is not asked about twice; the calls before them, and all at
// the end, are given up.
func TestHeadStartKeeps(t *testing.T) {
	model := newControlledLiveLLM(false)
	t.Cleanup(model.close)
	r := newHeadRun(t, model)
	var calls []*controlledLiveCall
	for _, live := range []string{"we ship", "we ship it", "we ship it on", "we ship it on friday"} {
		if !r.head.ask(live) || r.head.ask(live) {
			t.Fatalf("%q: not asked about once", live)
		}
		calls = append(calls, waitLiveCall(t, model, live))
	}
	givenUp(t, calls[0], "the oldest request")
	for i, c := range calls[1:] {
		if c.ctx.Err() != nil {
			t.Errorf("request %d was given up", i+1)
		}
	}
	r.head.drop()
	for _, c := range calls[1:] {
		givenUp(t, c, "a request")
	}
}

// TestHeadStartFinalised: the text asked about is what the speech model
// finalised and the live text after it, and nothing while the live text
// does not reach there.
func TestHeadStartFinalised(t *testing.T) {
	model := newControlledLiveLLM(false)
	t.Cleanup(model.close)
	r := newHeadRun(t, model)
	r.head.hearFinal("We moved the launch to Friday.")
	if r.head.ask("we moved the") {
		t.Error("asked about a live text that does not reach the finalised one's end")
	}
	if !r.head.ask("he moved the launch to friday then we left") {
		t.Fatal("not asked")
	}
	if in := inputOf(waitLiveCall(t, model, "the request").request); in != "We moved the launch to Friday. then we left" {
		t.Errorf("asked about %q", in)
	}
}

// TestNoteRate: a service rests after it limited the rate, not after a
// retry for anything else.
func TestNoteRate(t *testing.T) {
	for _, tc := range []struct {
		name  string
		resp  llm.Response
		err   error
		rests bool
	}{
		{"answered", llm.Response{Attempts: 1}, nil, false},
		{"answered after a retry", llm.Response{Attempts: 2}, nil, false},
		{"failed", llm.Response{Attempts: 2}, llm.HTTPError("deepseek", "m", 503, "", 0), false},
		{"answered after a 429", llm.Response{Attempts: 2, RateLimited: true}, nil, true},
		{"refused with a 429", llm.Response{Attempts: 1}, llm.HTTPError("deepseek", "m", 429, "", 0), true},
		{"a 429, then another failure", llm.Response{Attempts: 2, RateLimited: true}, llm.HTTPError("deepseek", "m", 503, "", 0), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, nil, nil, nil)
			h.m.noteRate("deepseek", tc.resp, tc.err)
			if got := h.m.resting("deepseek"); got != tc.rests {
				t.Errorf("rests %v, want %v", got, tc.rests)
			}
		})
	}
}
