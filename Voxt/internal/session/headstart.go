package session

import (
	"context"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const (
	// headKeep is how many texts are asked about at a time: the live text's
	// end often changes once more, and the answer to the text before it may
	// be the Final's.
	headKeep = 3
	// headAsks is the most texts a recording is asked about before Stop.
	headAsks = 20
)

// headStart asks the language model to clean a dictation up before its
// Final is there, from the text known so far (guess): when the user
// pauses, and at Stop. The Final is often that text, and its requests then
// are the ones asked already: they take the answers under way (askPart)
// instead of asking, also when the Final says the same but for punctuation
// or a few words (headmatch.go). The requests the Final does not need are
// given up.
//
// What is asked is the whole text, as after Stop: a stretch cleaned alone
// does not know what is said later, and cannot take back what it wrote.
//
// The live clean-up (livepolish.go) is left out where this runs: it cleans
// the sentences one at a time, and its answers count only when the Final
// repeats the live text word for word.
//
// A server with one queue would answer these before the Final's own, so
// this is a remote service's (Model.Remote).
type headStart struct {
	r       *run
	enabled atomic.Bool
	service string // the language model's id
	under   int    // its ThinkingOffUnder

	// The session goroutine's.
	text      string // the text asked about last
	asks      int    // texts asked about while the user speaks
	lateAsked bool

	mu    sync.Mutex
	final string        // what the speech model finalised so far (asr.EventPiece)
	gens  [][]*headCall // the requests of the texts asked about last, headKeep at most, the newest last
}

type headPart struct {
	req   llm.Request
	text  string
	reuse *headCall
}

type headPlan struct {
	heard string
	rest  string
	parts []headPart
}

func (p headPlan) hasFresh() bool {
	for _, part := range p.parts {
		if part.reuse == nil {
			return true
		}
	}
	return false
}

// headCall is a request asked early. resp and err are there once done is
// closed; the answer so far goes to onDelta, or is kept until a request
// takes the call.
type headCall struct {
	req    llm.Request
	text   string // the part's
	late   bool
	cancel context.CancelFunc
	done   chan struct{}
	resp   llm.Response
	err    error

	mu      sync.Mutex
	held    strings.Builder
	onDelta func(string)
}

// prepareHeadInputs allocates partial and pause channels before ASR workers
// start when Head Start could become eligible later.
func (r *run) prepareHeadInputs(endWord bool) bool {
	r.head.r = r
	possible := r.req.Kind == store.KindTranscribe && !r.settings.Features.Transcribe.CleanAfterStop && r.shot == nil && len(r.req.Images) == 0
	if possible {
		if r.init.split && !r.init.cleanupDecisionReady {
			possible = r.settings.Features.Transcribe.Enhance || r.sc.cleanupPending
		} else {
			possible = r.transcribeCleanup()
		}
	}
	if possible && (!r.init.split || r.init.llmReady) {
		possible = r.svc.LLM != nil && r.svc.LLMInfo.Remote && !r.m.resting(r.svc.LLMInfo.ID)
	}
	if (endWord || possible) && r.heard == nil {
		r.heard, r.silent = make(chan heardText, 1), make(chan bool, 1)
	}
	return possible
}

// startHead decides eligibility only after cleanup readiness is known.
// The atomic store publishes the immutable service and Thinking policy to
// stream and cleanup workers.
func (r *run) startHead() {
	if r.head.enabled.Load() {
		return
	}
	if r.head.r == nil {
		r.head.r = r
	}
	service := r.svc.LLMInfo.ID
	if r.req.Kind != store.KindTranscribe || r.settings.Features.Transcribe.CleanAfterStop || r.shot != nil || len(r.req.Images) > 0 ||
		!r.svc.LLMInfo.Remote || r.svc.LLM == nil || !r.transcribeCleanup() || r.m.resting(service) {
		return
	}
	r.head.service = service
	r.head.under = r.thinkOffUnder(service)
	r.head.enabled.Store(true)
}

func (h *headStart) canAsk() bool {
	r := h.r
	if !h.enabled.Load() || r == nil || r.ctx.Err() != nil || r.cancelled.Load() || r.stopped.Load() || r.streamReaderOff.Load() {
		return false
	}
	r.m.mu.Lock()
	defer r.m.mu.Unlock()
	return r.cleanupDecisionOpenLocked() && !r.skipped.Load()
}

// pause asks about the live text while the user is silent.
func (h *headStart) pause(live string) {
	if h.asks < headAsks && h.ask(live) {
		h.asks++
	}
}

// hearFinal takes the final text so far of a recording that finalises
// stretches while it goes on: the head start's text begins with it.
func (h *headStart) hearFinal(text string) {
	h.mu.Lock()
	h.final = strings.TrimSpace(text)
	h.mu.Unlock()
}

// guess is the final text as far as it is known: what the speech model
// finalised, which the Final repeats, and the live text after it. False
// when the live text does not show where that ends.
func (h *headStart) guess(live string) (string, bool) {
	h.mu.Lock()
	final := h.final
	h.mu.Unlock()
	live = strings.TrimSpace(live)
	if final == "" {
		return live, live != ""
	}
	tail, ok := tailAfter(live, final)
	if !ok {
		return "", false
	}
	return joinSpeech(final, tail), true
}

// ask sends the requests of the text known so far, unless it has. It
// reports whether it asked anything.
func (h *headStart) ask(live string) bool {
	if !h.canAsk() {
		return false
	}
	heard, cleaned, ok := h.normalize(live)
	if !ok {
		return false
	}
	if heard == h.text {
		return false
	}
	h.text = heard
	return h.send(cleaned)
}

func (h *headStart) normalize(live string) (heard, cleaned string, ok bool) {
	r := h.r
	heard, ok = h.guess(live)
	if !ok {
		return "", "", false
	}
	heard, _ = withoutEndWord(heard, r.endWord())
	if heard == "" {
		return "", "", false
	}
	cleaned = heard
	if r.settings.Dictionary.Enabled {
		cleaned = dict.Replace(heard, r.terms)
	}
	return heard, cleaned, true
}

func (h *headStart) prepare(cleaned string) (headPlan, bool, bool) {
	r := h.r
	plan := headPlan{}
	rest, ok := r.earlyCleanup().rest(cleaned)
	if _, whole := snippet.Whole(cleaned, r.snippets); whole || !ok || strings.TrimSpace(rest) == "" {
		return plan, false, true
	}
	parts, err := compose.BuildParts(r.transcribeInput(rest))
	if err != nil {
		return plan, false, false
	}
	plan.rest = rest
	for _, part := range parts[:min(len(parts), partsTogether)] {
		req := r.partRequest(cleaned, part, nil, nil)
		plan.parts = append(plan.parts, headPart{req: req, text: part.Text, reuse: h.find(req)})
	}
	return plan, true, false
}

func (h *headStart) prepareLate(live string) (headPlan, bool) {
	if !h.lateAllowed() {
		return headPlan{}, false
	}
	heard, cleaned, ok := h.normalize(live)
	if !ok || heard == h.text {
		return headPlan{}, false
	}
	plan, ok, _ := h.prepare(cleaned)
	if !ok {
		return headPlan{}, false
	}
	plan.heard = heard
	if !plan.hasFresh() || !h.lateAllowed() {
		return headPlan{}, false
	}
	return plan, true
}

func (h *headStart) lateAllowed() bool {
	if !h.enabled.Load() || h.lateAsked || h.r.ctx.Err() != nil {
		return false
	}
	ctx := h.r.currentLLMContext()
	if ctx == nil || ctx.Err() != nil || !h.r.cleanupMayContinue() || h.r.m.resting(h.service) {
		return false
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.gens) < headKeep
}

func (h *headStart) submitLate(plan headPlan) bool {
	if !plan.hasFresh() || !h.lateAllowed() {
		return false
	}
	h.submit(plan, true)
	return true
}

func (h *headStart) submit(plan headPlan, late bool) {
	calls := make([]*headCall, 0, len(plan.parts))
	for _, part := range plan.parts {
		call := part.reuse
		if call == nil {
			call = h.start(part.req, part.text, late)
		}
		calls = append(calls, call)
	}
	if late {
		h.text = plan.heard
		h.lateAsked = true
	}
	h.push(calls)
	h.r.logf("debug", "asking the language model ahead about %d characters", utf8.RuneCountInString(plan.rest))
}

// send sends the requests of text, its dictionary's aliases replaced, and
// gives up those of texts before the last headKeep.
func (h *headStart) send(text string) bool {
	plan, ok, discard := h.prepare(text)
	if !ok {
		if discard {
			h.drop()
		}
		return false
	}
	h.submit(plan, false)
	return true
}

func (h *headStart) start(req llm.Request, text string, late bool) *headCall {
	r := h.r
	ctx, cancel := context.WithCancel(r.currentLLMContext())
	c := &headCall{req: req, text: text, late: late, cancel: cancel, done: make(chan struct{})}
	req.OnDelta = c.add
	go func() {
		defer close(c.done)
		defer cancel()
		c.resp, c.err = r.llmChatContext(ctx, req)
		r.m.noteRate(h.service, c.resp, c.err)
	}()
	return c
}

// all are the calls of the texts asked about, the newest text's first.
func (h *headStart) all() []*headCall {
	h.mu.Lock()
	defer h.mu.Unlock()
	var calls []*headCall
	for _, gen := range slices.Backward(h.gens) {
		for _, c := range gen {
			if !slices.Contains(calls, c) {
				calls = append(calls, c)
			}
		}
	}
	return calls
}

// find is the call that asks the same as req, nil for none.
func (h *headStart) find(req llm.Request) *headCall {
	calls := h.all()
	i := slices.IndexFunc(calls, func(c *headCall) bool { return h.sameRequest(c.req, req) })
	if i < 0 {
		return nil
	}
	return calls[i]
}

// push keeps the calls of the text asked about now, with those of the
// texts before it (headKeep of them in all), and gives up the others.
func (h *headStart) push(gen []*headCall) {
	before := h.all()
	h.mu.Lock()
	h.gens = append(slices.Clone(h.gens), gen)
	h.gens = h.gens[max(0, len(h.gens)-headKeep):]
	h.mu.Unlock()
	kept := h.all()
	for _, c := range before {
		if !slices.Contains(kept, c) {
			c.cancel()
		}
	}
}

// drop gives up every call.
func (h *headStart) drop() {
	before := h.all()
	h.mu.Lock()
	h.gens = nil
	h.mu.Unlock()
	for _, c := range before {
		c.cancel()
	}
}

// near is a call for a part that says what text says but for a few words
// at most, and the words its answer must be put right for; nil for none.
func (h *headStart) near(req llm.Request, text string) (*headCall, []spokenWord, []headChange) {
	last := len(req.Messages) - 1
	type match struct {
		call    *headCall
		old     []spokenWord
		changes []headChange
		done    bool
		err     error
	}
	var exact, repaired []match
	for _, c := range h.all() {
		if len(c.req.Messages) != last+1 || last < 0 || !slices.Equal(c.req.Messages[:last], req.Messages[:last]) || !h.sameThinking(c.req, req) ||
			strings.Replace(c.req.Messages[last].Content, c.text, text, 1) != req.Messages[last].Content {
			continue
		}
		done, err := h.callError(c)
		if c.late && done && err != nil {
			continue
		}
		old := spokenWords(c.text)
		changes, ok := headChanges(old, c.text, text)
		if !ok {
			continue
		}
		m := match{call: c, old: old, changes: changes, done: done, err: err}
		if len(changes) == 0 {
			exact = append(exact, m)
		} else {
			repaired = append(repaired, m)
		}
	}
	choose := func(matches []match) *match {
		if len(matches) == 0 {
			return nil
		}
		first := matches[0]
		if first.call.late && !first.done {
			for i := 1; i < len(matches); i++ {
				if matches[i].done && matches[i].err == nil {
					return &matches[i]
				}
			}
		}
		return &first
	}
	chosen := choose(exact)
	if chosen == nil {
		chosen = choose(repaired)
	}
	if chosen == nil {
		return nil, nil, nil
	}
	return chosen.call, chosen.old, chosen.changes
}

// askPart asks for one part's clean-up, with the answer a head start has
// under way when there is one for it. Goroutine-safe.
func (r *run) askPart(ctx context.Context, req llm.Request, text string) (llm.Response, error) {
	h := &r.head
	if c := h.find(req); c != nil && (!c.late || !h.failedLate(c)) {
		r.logf("debug", "the language model was asked about this part ahead of the final text")
		if !c.late {
			return c.join(req.OnDelta)
		}
		resp, err := c.join(nil)
		if err == nil {
			c.mu.Lock()
			deltas := c.held.String()
			c.held.Reset()
			c.mu.Unlock()
			if deltas != "" && req.OnDelta != nil {
				req.OnDelta(deltas)
			}
			return resp, nil
		}
		if ctx.Err() != nil || r.ctx.Err() != nil || !r.cleanupMayContinue() {
			return resp, err
		}
	}
	if c, old, changes := h.near(req, text); c != nil {
		if resp, err, ok := h.reuse(c, old, changes, req, text); ok {
			return resp, err
		}
	}
	resp, err := r.llmChatContext(ctx, req)
	if h.enabled.Load() {
		r.m.noteRate(h.service, resp, err)
	}
	return resp, err
}

// reuse is c's answer put right for text, and false when text must be
// asked about. An answer that cannot be put right is found out only once
// it is there: with changes to place, text is asked about meanwhile, and
// the first answer counts.
func (h *headStart) reuse(c *headCall, old []spokenWord, changes []headChange, req llm.Request, text string) (llm.Response, error, bool) {
	right := func() (llm.Response, bool) {
		if c.err != nil {
			return llm.Response{}, false
		}
		resp, ok := c.resp, false
		if resp.Text, ok = putRight(resp.Text, old, changes); ok {
			h.r.logf("debug", "the language model was asked about this part ahead of the final text, %d of its words put right", len(changes))
		}
		return resp, ok
	}
	select {
	case <-c.done:
		resp, ok := right()
		return resp, nil, ok
	default:
	}
	if len(changes) == 0 {
		<-c.done
		resp, ok := right()
		return resp, nil, ok
	}
	fresh := h.start(req, text, false)
	defer fresh.cancel()
	select {
	case <-c.done:
		if resp, ok := right(); ok {
			return resp, nil, true
		}
	case <-fresh.done:
	}
	resp, err := fresh.join(req.OnDelta)
	return resp, err, true
}

func (h *headStart) callError(c *headCall) (bool, error) {
	select {
	case <-c.done:
		return true, c.err
	default:
		return false, nil
	}
}

func (h *headStart) failedLate(c *headCall) bool {
	done, err := h.callError(c)
	return c.late && done && err != nil
}

// join streams the answer so far and the rest into onDelta, and returns
// the call's end.
func (c *headCall) join(onDelta func(string)) (llm.Response, error) {
	c.mu.Lock()
	if held := c.held.String(); held != "" && onDelta != nil {
		onDelta(held)
	}
	c.onDelta = onDelta
	c.mu.Unlock()
	<-c.done
	return c.resp, c.err
}

func (c *headCall) add(delta string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.onDelta != nil {
		c.onDelta(delta)
		return
	}
	c.held.WriteString(delta)
}

// partRequest is the request for a part of text's clean-up.
func (r *run) partRequest(text string, p compose.Part, think *llm.Thinking, onDelta func(string)) llm.Request {
	req := llm.Request{Messages: llmMessages(p.Output), Skill: llm.SkillTranscribeEnhance, Thinking: think, InputChars: utf8.RuneCountInString(text), OnDelta: onDelta}
	return r.m.cfg.llmDeadlines(req, utf8.RuneCountInString(p.Text), r.llmFloor(r.m.cfg.LLMTimeout))
}

// sameRequest says whether a and b ask the service the same: the same
// messages, with the same thinking.
func (h *headStart) sameRequest(a, b llm.Request) bool {
	return slices.Equal(a.Messages, b.Messages) && h.sameThinking(a, b)
}

// sameThinking says whether the service's setting gives a and b the same
// thinking.
func (h *headStart) sameThinking(a, b llm.Request) bool {
	return a.Thinking == nil && b.Thinking == nil && (a.InputChars < h.under) == (b.InputChars < h.under)
}

// thinkOffUnder is the length under which a service's setting turns its
// thinking off (0: never).
func (r *run) thinkOffUnder(service string) int {
	for _, provider := range r.settings.Providers.LLM {
		if provider.ID == service {
			return provider.Generation.ThinkingOffUnder
		}
	}
	return 0
}
