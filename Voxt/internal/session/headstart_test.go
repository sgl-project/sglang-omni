package session

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
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

func waitHeadCall(t *testing.T, call *headCall, what string) {
	t.Helper()
	select {
	case <-call.done:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s to finish", what)
	}
}

func watchExactAdoption(r *run) <-chan struct{} {
	selected := make(chan struct{}, 1)
	log := r.m.cfg.Log
	r.m.cfg.Log = func(level, message string) {
		log(level, message)
		if strings.Contains(message, "the language model was asked about this part ahead of the final text") {
			select {
			case selected <- struct{}{}:
			default:
			}
		}
	}
	return selected
}

func prepareLateHeadCall(t *testing.T, r *run, model *controlledLiveLLM, text string) (headPlan, *headCall, *controlledLiveCall) {
	t.Helper()
	plan, ok := r.head.prepareLate(text)
	if !ok || !r.head.submitLate(plan) {
		t.Fatal("the late candidate was not submitted")
	}
	call := waitLiveCall(t, model, "the late candidate")
	return plan, r.head.all()[0], call
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

func TestHeadStartLateCandidateAfterCapture(t *testing.T) {
	const transcript = "we ship the new build on friday"
	model := newControlledLiveLLM(false)
	t.Cleanup(model.close)
	resolveGate, finishGate := make(chan struct{}), make(chan struct{})
	var resolveOnce sync.Once
	releaseResolve := func() { resolveOnce.Do(func() { close(resolveGate) }) }
	t.Cleanup(releaseResolve)
	streamer := &fakeStreamer{
		final: transcript, silent: true, candidates: []string{transcript},
		openStarted: make(chan struct{}, 1),
		finishGate:  finishGate, finishStarted: make(chan struct{}, 1),
	}
	var finishOnce sync.Once
	releaseFinish := func() { finishOnce.Do(func() { close(finishGate) }) }
	t.Cleanup(releaseFinish)
	h := newHeadSession(t, streamer, model, nil)
	h.resolveGate = resolveGate
	id := h.start()
	h.speak(id, 10)
	h.rec.waitFor(t, "the captured recording", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.Timings.Audio > 0
	})
	releaseResolve()
	select {
	case <-streamer.openStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("the ASR stream did not start opening after capture")
	}
	select {
	case <-streamer.finishStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("the ASR stream did not finish")
	}
	call := waitLiveCall(t, model, "the late ASR candidate")
	if input := inputOf(call.request); input != transcript {
		t.Errorf("asked about %q, want %q", input, transcript)
	}
	if !streamer.latePartialsRequested() {
		t.Error("HeadStart did not opt in to late candidates")
	}
	for _, snapshot := range h.rec.all() {
		if snapshot.ID == id && snapshot.Partial != "" {
			t.Errorf("the advisory candidate became visible: %q", snapshot.Partial)
		}
	}
	call.complete(t, "We ship the new build on Friday.", nil)
	releaseFinish()
	if end := h.rec.wait(t); end.Result != "We ship the new build on Friday." || end.Problem != nil {
		t.Errorf("result %q, problem %+v", end.Result, end.Problem)
	}
	if n := model.requestCount(); n != 1 {
		t.Errorf("%d requests, want the late request adopted by Final", n)
	}
}

func TestHeadStartOrdinaryPartialAfterCapture(t *testing.T) {
	const transcript = "we ship the new build on friday"
	model := newControlledLiveLLM(false)
	t.Cleanup(model.close)
	openGate, partialGate, returnGate := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var openOnce, partialOnce, returnOnce sync.Once
	releaseOpen := func() { openOnce.Do(func() { close(openGate) }) }
	releasePartial := func() { partialOnce.Do(func() { close(partialGate) }) }
	releaseSend := func() { returnOnce.Do(func() { close(returnGate) }) }
	t.Cleanup(releaseOpen)
	t.Cleanup(releasePartial)
	t.Cleanup(releaseSend)
	streamer := &fakeStreamer{
		final: transcript, partials: []string{transcript},
		openGate: openGate, openStarted: make(chan struct{}, 1),
		partialGate: partialGate, partialReturnGate: returnGate,
		finishStarted: make(chan struct{}, 1),
	}
	h := newHeadSession(t, streamer, model, nil)
	id := h.start()
	h.speak(id, 10)
	h.rec.waitFor(t, "capture completion", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.State == Recognizing && snapshot.Timings.Audio > 0
	})
	select {
	case <-streamer.openStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("the ASR stream did not start opening")
	}
	releaseOpen()
	releasePartial()
	call := waitLiveCall(t, model, "the post-capture ordinary partial")
	select {
	case <-streamer.finishStarted:
		t.Fatal("Finish began before the ordinary partial started HeadStart")
	default:
	}
	h.rec.waitFor(t, "the visible ordinary partial", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.Partial == transcript
	})
	if !streamer.latePartialsRequested() {
		t.Error("eligible HeadStart did not opt in to late candidates")
	}
	call.complete(t, "We ship the new build on Friday.", nil)
	releaseSend()
	if end := h.rec.wait(t); end.Result != "We ship the new build on Friday." || end.Problem != nil {
		t.Errorf("result %q, problem %+v", end.Result, end.Problem)
	}
	if n := model.requestCount(); n != 1 {
		t.Errorf("%d requests, want the ordinary partial reused by Final", n)
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
			if h.streamer.latePartialsRequested() {
				t.Error("disabled HeadStart opted in to late candidates")
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
	r.startHead()
	if !r.head.enabled.Load() {
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

func TestHeadStartLateCandidateLimit(t *testing.T) {
	for _, tc := range []struct {
		name      string
		texts     []string
		candidate string
		allowed   bool
	}{
		{"one slot remains", []string{"we ship a build", "we ship a build friday"}, "we ship a build friday tomorrow", true},
		{"headKeep is full", []string{"we ship build 0", "we ship build 1", "we ship build 2"}, "we ship build 2 tomorrow", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := newControlledLiveLLM(false)
			t.Cleanup(model.close)
			r := newHeadRun(t, model)
			var calls []*controlledLiveCall
			for _, live := range tc.texts {
				if !r.head.ask(live) {
					t.Fatalf("not asked about %q", live)
				}
				calls = append(calls, waitLiveCall(t, model, live))
			}
			gen, ok := r.head.prepareLate(tc.candidate)
			if ok != tc.allowed {
				t.Fatalf("late candidate preparation %v, want %v", ok, tc.allowed)
			}
			if tc.allowed {
				if !r.head.submitLate(gen) {
					t.Fatal("the prepared generation was not submitted")
				}
				calls = append(calls, waitLiveCall(t, model, "the late generation"))
				if _, ok := r.head.prepareLate(tc.candidate + " morning"); ok {
					t.Error("a second late generation was prepared")
				}
			}
			wantCalls := len(tc.texts)
			if tc.allowed {
				wantCalls++
			}
			if n := model.requestCount(); n != wantCalls {
				t.Errorf("%d requests, want %d", n, wantCalls)
			}
			for i, call := range calls {
				if call.ctx.Err() != nil {
					t.Errorf("generation %d was evicted", i)
				}
			}
			r.head.drop()
			for _, call := range calls {
				givenUp(t, call, "a retained request")
			}
		})
	}
}

func TestHeadStartLateCandidateRejectionsPreserveRequests(t *testing.T) {
	model := newControlledLiveLLM(false)
	t.Cleanup(model.close)
	r := newHeadRun(t, model)
	if !r.head.ask("we ship on friday") {
		t.Fatal("the useful early request was not started")
	}
	call := waitLiveCall(t, model, "the useful early request")
	if _, ok := r.head.prepareLate("we ship on friday"); ok {
		t.Error("a duplicate candidate was prepared")
	}
	if _, ok := r.head.prepareLate(""); ok {
		t.Error("an empty candidate was prepared")
	}
	r.head.hearFinal("we moved the launch")
	if _, ok := r.head.prepareLate("unrelated words"); ok {
		t.Error("an incompatible candidate was prepared")
	}
	r.head.hearFinal("")
	if r.head.text != "we ship on friday" || r.head.lateAsked || len(r.head.all()) != 1 {
		t.Errorf("rejected candidates changed retained state: text %q, late %v, calls %d", r.head.text, r.head.lateAsked, len(r.head.all()))
	}
	gen, ok := r.head.prepareLate("we ship on friday tomorrow")
	if !ok || !r.head.submitLate(gen) {
		t.Fatal("rejected candidates consumed the late-generation allowance")
	}
	late := waitLiveCall(t, model, "the valid late candidate")
	if call.ctx.Err() != nil {
		t.Error("a rejected candidate discarded the useful early request")
	}
	r.head.drop()
	givenUp(t, late, "the late request")
}

func TestHeadStartLateCandidateCancellation(t *testing.T) {
	model := newControlledLiveLLM(false)
	t.Cleanup(model.close)
	r := newHeadRun(t, model)
	if !r.head.ask("we ship") {
		t.Fatal("the useful early request was not started")
	}
	waitLiveCall(t, model, "the useful early request")
	r.cancel()
	if _, ok := r.head.prepareLate("we ship tomorrow"); ok {
		t.Error("a cancelled session prepared a late candidate")
	}
	r.head.mu.Lock()
	retained := len(r.head.gens) == 1 && len(r.head.gens[0]) == 1 && r.head.gens[0][0] != nil
	r.head.mu.Unlock()
	if !retained || r.head.lateAsked {
		t.Errorf("cancellation changed retained requests or consumed the allowance: retained %v, late %v", retained, r.head.lateAsked)
	}
}

func TestHeadStartLateCandidateDoesNotDisplaceCompleted(t *testing.T) {
	for _, tc := range []struct {
		name string
		fail bool
	}{
		{"unfinished candidate", false},
		{"failed candidate", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := newControlledLiveLLM(false)
			t.Cleanup(model.close)
			r := newHeadRun(t, model)
			if !r.head.ask("we ship on friday") {
				t.Fatal("the earlier request was not started")
			}
			completed := waitLiveCall(t, model, "the earlier request")
			earlier := r.head.all()[0]
			completed.complete(t, "We ship on Friday.", nil)
			waitHeadCall(t, earlier, "the earlier HeadStart request")
			gen, ok := r.head.prepareLate("we ship, on friday")
			if !ok || !r.head.submitLate(gen) {
				t.Fatal("the late candidate was not submitted")
			}
			late := waitLiveCall(t, model, "the late candidate")
			lateHead := r.head.all()[0]
			if tc.fail {
				late.complete(t, "", errors.New("request failed"))
				waitHeadCall(t, lateHead, "the failed late request")
			} else {
				select {
				case <-lateHead.done:
					t.Fatal("the unfinished late candidate completed before selection")
				default:
				}
			}
			final, ok, _ := r.head.prepare("We ship on Friday.")
			if !ok || len(final.parts) != 1 {
				t.Fatalf("final cleanup plan %+v, %v", final, ok)
			}
			if r.head.find(final.parts[0].req) != nil {
				t.Fatal("the case unexpectedly became an exact request match")
			}
			if selected, _, changes := r.head.near(final.parts[0].req, final.parts[0].text); selected != earlier || len(changes) != 0 {
				t.Fatalf("selected call %p with %d repairs, want the completed call %p with none", selected, len(changes), earlier)
			}
			type partResult struct {
				resp llm.Response
				err  error
			}
			result := make(chan partResult, 1)
			go func() {
				resp, err := r.askPart(r.ctx, final.parts[0].req, final.parts[0].text)
				result <- partResult{resp: resp, err: err}
			}()
			select {
			case got := <-result:
				if got.err != nil || got.resp.Text != "We ship on Friday." {
					t.Errorf("cleanup result %+v, %v", got.resp, got.err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("cleanup waited for the newer candidate or started a fresh request")
			}
			if model.requestCount() != 2 {
				t.Errorf("%d requests, want only the earlier and late candidates", model.requestCount())
			}
			if !tc.fail && late.ctx.Err() != nil {
				t.Error("the unfinished late request was canceled instead of retained")
			}
			r.head.drop()
			if !tc.fail {
				givenUp(t, late, "the retained late request")
			}
		})
	}
}

func TestHeadStartLateExactFailureRecovery(t *testing.T) {
	const transcript = "we ship on friday"
	t.Run("active cleanup retries without failed deltas", func(t *testing.T) {
		model := newControlledLiveLLM(false)
		t.Cleanup(model.close)
		r := newHeadRun(t, model)
		selected := watchExactAdoption(r)
		plan, lateHead, late := prepareLateHeadCall(t, r, model, transcript)
		late.sendDelta(t, "failed speculative fragment")
		parts, err := compose.BuildParts(r.transcribeInput(plan.rest))
		if err != nil || len(parts) != 1 {
			t.Fatalf("cleanup parts: %v, %v", parts, err)
		}
		r.historyRetry = true
		drafts := make(chan string, 8)
		r.onDraft = func(draft string) { drafts <- draft }
		type cleanupResult struct {
			text string
			err  error
		}
		finished := make(chan cleanupResult, 1)
		go func() {
			text, _, err := r.cleanParts(transcript, parts, nil, cleanupRequest{ctx: r.currentLLMContext()})
			finished <- cleanupResult{text: text, err: err}
		}()
		select {
		case <-selected:
		case <-time.After(5 * time.Second):
			t.Fatal("the late exact-match request was not adopted")
		}
		late.complete(t, "", errors.New("speculative request failed"))
		waitHeadCall(t, lateHead, "the failed late exact match")
		fresh := waitLiveCall(t, model, "the authoritative cleanup retry")
		fresh.sendDelta(t, "authoritative fragment")
		select {
		case draft := <-drafts:
			if strings.Contains(draft, "failed speculative fragment") || !strings.Contains(draft, "authoritative fragment") {
				t.Errorf("Draft mixed failed and authoritative deltas: %q", draft)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("the authoritative delta did not reach Draft")
		}
		fresh.complete(t, "We ship on Friday.", nil)
		select {
		case result := <-finished:
			if result.err != nil || result.text != "We ship on Friday." {
				t.Errorf("cleanup result %q, %v", result.text, result.err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("authoritative cleanup did not finish")
		}
		if model.requestCount() != 2 {
			t.Errorf("%d requests, want one speculative and one authoritative", model.requestCount())
		}
		r.head.drop()
	})

	t.Run("cancellation does not retry", func(t *testing.T) {
		model := newControlledLiveLLM(false)
		t.Cleanup(model.close)
		r := newHeadRun(t, model)
		selected := watchExactAdoption(r)
		plan, lateHead, late := prepareLateHeadCall(t, r, model, transcript)
		late.sendDelta(t, "private speculative fragment")
		var streamed strings.Builder
		request := plan.parts[0].req
		request.OnDelta = func(delta string) { streamed.WriteString(delta) }
		type partResult struct {
			resp llm.Response
			err  error
		}
		finished := make(chan partResult, 1)
		go func() {
			resp, err := r.askPart(r.ctx, request, plan.parts[0].text)
			finished <- partResult{resp: resp, err: err}
		}()
		select {
		case <-selected:
		case <-time.After(5 * time.Second):
			t.Fatal("the late exact-match request was not adopted")
		}
		r.cancel()
		waitHeadCall(t, lateHead, "the cancelled late exact match")
		select {
		case result := <-finished:
			if result.err == nil {
				t.Error("cancelled cleanup unexpectedly succeeded")
			}
		case <-time.After(5 * time.Second):
			t.Fatal("cancelled cleanup did not return")
		}
		if model.requestCount() != 1 || streamed.Len() != 0 {
			t.Errorf("cancellation launched recovery or published speculative output: requests %d, streamed %q", model.requestCount(), streamed.String())
		}
	})
}

func TestLateDispatchReturnsReadyFinal(t *testing.T) {
	model := newControlledLiveLLM(false)
	t.Cleanup(model.close)
	r := newHeadRun(t, model)
	plan, ok := r.head.prepareLate("we ship on friday tomorrow")
	if !ok {
		t.Fatal("late candidate preparation failed")
	}
	finals := make(chan streamEnd, 1)
	finals <- streamEnd{silent: true}
	end, err, ready := r.dispatchLate(r.ctx, finals, plan)
	if !ready || err != nil || !end.silent {
		t.Errorf("terminal result %+v, %v, ready %v", end, err, ready)
	}
	if model.requestCount() != 0 || r.head.lateAsked || len(r.head.all()) != 0 {
		t.Errorf("speculative request started after Final was ready: requests %d, late %v, retained %d", model.requestCount(), r.head.lateAsked, len(r.head.all()))
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

func setBlockedRemoteResolvers(h *harness, provider asr.Provider, model LLM) (<-chan struct{}, func()) {
	started := make(chan struct{}, 1)
	release, openRelease := controlledRelease(h.t)
	h.m.cfg.ResolveASR = func(context.Context, Kind, settings.Settings) ASRReadiness {
		return ASRReadiness{ASR: provider, ASRInfo: Model{ID: "fake-asr"}}
	}
	h.m.cfg.ResolveLLM = func(context.Context, Kind, settings.Settings) LLMReadiness {
		started <- struct{}{}
		<-release
		return LLMReadiness{LLM: model, LLMInfo: Model{ID: "fake-llm", Remote: true}, Outcome: LLMAvailable}
	}
	return started, openRelease
}

func waitPauseState(t *testing.T, r *run, want bool) {
	t.Helper()
	deadline := time.NewTimer(5 * time.Second)
	tick := time.NewTicker(time.Millisecond)
	defer deadline.Stop()
	defer tick.Stop()
	for {
		if r.pauseWatch != nil && r.pauseWatch.quiet.Load() == want {
			return
		}
		select {
		case <-tick.C:
		case <-deadline.C:
			t.Fatalf("pause state did not become %v", want)
		}
	}
}

func waitHeadEnabled(t *testing.T, r *run) {
	t.Helper()
	deadline := time.NewTimer(5 * time.Second)
	tick := time.NewTicker(time.Millisecond)
	defer deadline.Stop()
	defer tick.Stop()
	for !r.head.enabled.Load() {
		select {
		case <-tick.C:
		case <-deadline.C:
			t.Fatal("Head Start did not become enabled")
		}
	}
}

func sessionRun(t *testing.T, h *harness) *run {
	t.Helper()
	h.m.mu.Lock()
	defer h.m.mu.Unlock()
	if h.m.cur == nil {
		t.Fatal("no current session")
	}
	return h.m.cur
}
