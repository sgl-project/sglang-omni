package session

import (
	"context"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// note (luojiaxuan): livePolish cleans a dictation up while the user speaks (behavior.md
// 5.31b): the whole sentences of the stream's live text go to the
// language model as they come, so that Stop leaves only what was said
// last, or nothing.
//
// The live text is the speech model's guess: at Stop each clean-up counts
// only when its text starts the final text word for word and ends at one
// of its sentence ends. The text is cleaned in blocks of at most
// compose.PieceChars characters, each block whole and with the prompt of a
// clean-up after Stop: a block is asked again whenever the live text adds
// a sentence to it, so that a list or a correction across its sentences is
// cleaned as at Stop. A block that is full is kept (committed), and the
// next one goes with the end of the cleaned text before it
// (compose.Input.Before, liveContext), as does the text left at Stop. A
// clean-up of a stretch takes the stretch for the whole dictation, so its
// last sentences are the least sure: when text is left at Stop, the last
// settings.Transcribe.RecleanSentences sentences of the cleaned text reused
// go back to the model with it, so that what was said last can still
// correct them ("Tuesday. No, make it Thursday."). A dictation that is a snippet's trigger alone is never sent
// (insertSnippet types the snippet), and a request that fails ends the live
// clean-up: Stop cleans the whole text, as without it.
//
// add is the stream's reader goroutine's, work the worker's, finish the
// session goroutine's (after the reader returned); mu guards the rest, and
// cond (on mu) wakes the worker.
type livePolish struct {
	r             *run
	ctx           context.Context
	cancel        context.CancelFunc
	thinkOffUnder int
	reclean       int // note (luojiaxuan): sentences cleaned again with the text left at Stop
	llm           LLM // note (luojiaxuan): the session's language model when the live clean-up started: the fallback swap after Stop does not reach the worker

	mu      sync.Mutex
	cond    *sync.Cond
	asked   int          // note (luojiaxuan): requests while recording (one per part of a clean-up)
	dropped int          // note (luojiaxuan): of them, cancelled as the live text changed
	text    string       // note (luojiaxuan): the newest live text, as the final text is cleaned (aliases replaced)
	blocks  []liveResult // note (luojiaxuan): committed, in order
	epoch   int          // note (luojiaxuan): bumped when blocks change: a clean-up of an older epoch has other text before it
	done    []liveResult // note (luojiaxuan): the clean-ups of the open block: each of a stretch that starts the text after blocks
	flight  *liveFlight  // note (luojiaxuan): the request under way, nil when none
	closed  bool         // note (luojiaxuan): the final text came, or the live clean-up gave up: no more requests

	// Test gates make completion/adoption interleavings deterministic; nil in production.
	testHooks *livePolishTestHooks
}

type livePolishTestHooks struct {
	afterAdoptSnapshot      func()
	beforeAdoptDraftPublish func()
	beforeFlightCommit      func()
	afterFinishAdopted      func()
}

func (lp *livePolish) runBeforeAdoptDraftPublish() {
	lp.mu.Lock()
	var hook func()
	if lp.testHooks != nil {
		hook = lp.testHooks.beforeAdoptDraftPublish
	}
	lp.mu.Unlock()
	if hook != nil {
		hook()
	}
}

// note (luojiaxuan): liveResult is the clean-up of src, a stretch of the live text that ends
// a sentence: the parts it was asked in (compose.BuildParts) and their
// answers.
type liveResult struct {
	src      string
	parts    []compose.Part
	answers  []string
	chars    int  // note (luojiaxuan): the dictation's characters up to the end of src
	thinkOff bool // note (luojiaxuan): asked with thinking off (chars under the service's ThinkingOffUnder)
	epoch    int
	usage    llm.Usage
}

// note (luojiaxuan): liveFlight is the request under way: done closes when it ends, ok when
// it answered.
type liveFlight struct {
	liveResult
	ctx          context.Context
	cancel       context.CancelFunc
	done         chan struct{}
	updates      chan struct{}
	parts        []compose.Part
	answers      []string
	partIndex    int
	delta        strings.Builder
	detached     bool
	adopted      bool
	streamClosed bool
	ok           bool
}

// note (luojiaxuan): hearLive takes a live text of the recording (asr.EventPartial) to the
// live clean-up. Once the speech model reports stretches of its own
// (hearPiece) the final text is made of those, and the live clean-up ends.
func (r *run) hearLive(text string) {
	if r.liveOff {
		return
	}
	r.liveOnce.Do(func() { r.live = newLivePolish(r) })
	if r.live == nil {
		return
	}
	text = strings.TrimSpace(text)
	if r.settings.Dictionary.Enabled {
		text = dict.Replace(text, r.terms)
	}
	r.live.add(text)
}

// note (luojiaxuan): endLive ends the live clean-up while the recording goes on (the
// stream's reader goroutine).
func (r *run) endLive() {
	r.liveOff = true
	if r.live != nil {
		r.live.stop()
	}
}

// note (luojiaxuan): newLivePolish starts the live clean-up of a Transcribe
// that cleans up with a language model; nil when it does not run: the user
// has it wait for Stop (settings.Transcribe.CleanAfterStop); the glossary
// depends on the text (more than compose.GlossaryAllUpTo terms: only the
// whole text picks its terms); a picture goes with the clean-up (each
// request would send it again, and the card would show it while the user
// speaks); or the language model is served on this Mac, where its requests
// would share the GPU with the speech model while the user speaks and
// beside the Final after Stop.
//
// Nor where the head start asks the whole text at every pause (headstart.go):
// it is ahead of the sentences this would clean one by one.
func newLivePolish(r *run) *livePolish {
	if r.head.enabled {
		return nil
	}
	r.m.mu.Lock()
	client, llmInfo, llmCtx := r.svc.LLM, r.svc.LLMInfo, r.llmCtx
	r.m.mu.Unlock()
	if r.req.Kind != store.KindTranscribe || r.settings.Features.Transcribe.CleanAfterStop || !r.transcribeCleanup() ||
		client == nil || len(r.glossary()) > compose.GlossaryAllUpTo || r.shot != nil || len(r.req.Images) > 0 {
		return nil
	}
	if r.llmOnThisMac(llmInfo.ID) {
		r.logf("info", "no clean-up while recording: the language model %s runs on this Mac", llmInfo.ID)
		return nil
	}
	thinkOffUnder := 0
	for _, provider := range r.settings.Providers.LLM {
		if provider.ID == llmInfo.ID {
			thinkOffUnder = provider.Generation.ThinkingOffUnder
			break
		}
	}
	ctx, cancel := context.WithCancel(llmCtx)
	lp := &livePolish{r: r, ctx: ctx, cancel: cancel, thinkOffUnder: thinkOffUnder, llm: client,
		reclean: max(0, r.settings.Features.Transcribe.RecleanSentences)}
	lp.cond = sync.NewCond(&lp.mu)
	go func() {
		<-ctx.Done()
		lp.mu.Lock()
		lp.closed = true
		if lp.flight != nil {
			lp.flight.detached = true
		}
		lp.cond.Broadcast()
		lp.mu.Unlock()
	}()
	go lp.work()
	return lp
}

// note (luojiaxuan): llmOnThisMac says whether language model id (the
// session's) is served on this Mac: its address's host is a loopback one (Ollama, LM
// Studio, a custom server here).
func (r *run) llmOnThisMac(id string) bool {
	endpoint := ""
	for _, provider := range r.settings.Providers.LLM {
		if provider.ID == id {
			endpoint = provider.Endpoint
			break
		}
	}
	preset, err := llm.ProviderConfig{ID: id, BaseURL: endpoint}.Resolve()
	if err != nil {
		return false
	}
	u, err := url.Parse(preset.BaseURL)
	return err == nil && core.IsLoopback(u.Hostname())
}

// note (luojiaxuan): stop gives the live clean-up up: Stop cleans the whole text.
func (lp *livePolish) stop() {
	lp.mu.Lock()
	lp.closed = true
	lp.blocks, lp.done = nil, nil
	if lp.flight != nil {
		lp.flight.detached = true
	}
	lp.mu.Unlock()
	lp.cancel()
}

// note (luojiaxuan): add takes the newest live text: the blocks it no longer starts with go,
// and so does a request for a stretch it no longer has.
func (lp *livePolish) add(text string) {
	lp.mu.Lock()
	defer lp.mu.Unlock()
	if lp.closed || text == lp.text {
		return
	}
	lp.text = text
	if k, _ := chainIn(text, lp.blocks); k < len(lp.blocks) {
		lp.blocks, lp.done = lp.blocks[:k], nil
		lp.epoch++
	}
	if f := lp.flight; f != nil && (f.epoch != lp.epoch || !endsSentence(lp.open(), f.src)) {
		f.detached = true
		f.cancel()
	}
	lp.cond.Broadcast()
}

// note (luojiaxuan): open is the live text after the blocks (lp.mu held; the blocks start it).
func (lp *livePolish) open() string {
	_, end := chainIn(lp.text, lp.blocks)
	return strings.TrimLeftFunc(lp.text[end:], unicode.IsSpace)
}

// note (luojiaxuan): next is the stretch to clean next (lp.mu held), false for none: the
// open block's newest whole sentences, unless they are cleaned already. A
// full block whose clean-up is done is committed first.
func (lp *livePolish) next() (liveResult, bool) {
	if _, trigger := snippet.Whole(lp.text, lp.r.snippets); trigger {
		return liveResult{}, false
	}
	for {
		open := lp.open()
		src, full := blockOf(open)
		if src == "" {
			return liveResult{}, false
		}
		i := slices.IndexFunc(lp.done, func(d liveResult) bool { return d.src == src })
		if i < 0 {
			chars := utf8.RuneCountInString(lp.text[:len(lp.text)-len(open)+len(src)])
			return liveResult{src: src, chars: chars, thinkOff: lp.thinkOffUnder > 0 && chars < lp.thinkOffUnder, epoch: lp.epoch}, true
		}
		if !full {
			return liveResult{}, false
		}
		lp.blocks, lp.done = append(lp.blocks, lp.done[i]), nil
		lp.epoch++
		lp.cond.Broadcast()
	}
}

// note (luojiaxuan): blockOf is the stretch of open to clean: the longest one that ends a
// sentence within compose.PieceChars characters; full when open is longer
// (the stretch is a block of its own, the text after it the next block's).
func blockOf(open string) (string, bool) {
	src := ""
	for _, end := range compose.SentenceEnds(open) {
		if utf8.RuneCountInString(open[:end]) > compose.PieceChars {
			break
		}
		src = open[:end]
	}
	return src, src != "" && utf8.RuneCountInString(open) > compose.PieceChars
}

// note (luojiaxuan): work cleans the stretches next gives, one request at a
// time, until finish or the session ends; finish does not wait for it, and
// an answer that comes after finish counts only as the request it awaits. A
// request that fails ends the live clean-up: Stop cleans the text, as
// without it.
func (lp *livePolish) work() {
	for {
		lp.mu.Lock()
		job, ok := liveResult{}, false
		for !lp.closed {
			if job, ok = lp.next(); ok {
				break
			}
			lp.cond.Wait()
		}
		if !ok {
			lp.mu.Unlock()
			return
		}
		ctx, cancel := context.WithCancel(lp.ctx)
		f := &liveFlight{liveResult: job, ctx: ctx, cancel: cancel, done: make(chan struct{}), updates: make(chan struct{}, 1)}
		lp.flight = f
		lp.cond.Broadcast()
		before := liveContext(joinResults(lp.blocks, lp.text))
		lp.mu.Unlock()
		start := time.Now()
		res, err := lp.clean(ctx, job, before, f)
		lp.mu.Lock()
		var beforeCommit func()
		if lp.testHooks != nil {
			beforeCommit = lp.testHooks.beforeFlightCommit
		}
		lp.mu.Unlock()
		if beforeCommit != nil {
			beforeCommit()
		}
		cancelled := ctx.Err() != nil // note (luojiaxuan): before disposal below, a failure is not a cancellation
		lp.mu.Lock()
		cancelled = cancelled || ctx.Err() != nil
		lp.flight = nil
		f.liveResult = res
		f.ok = err == nil && !cancelled && !f.detached
		f.streamClosed = true
		switch {
		case lp.closed:
			// finish owns an adopted flight; the caller consumes its terminal state.
		case cancelled || f.detached:
			lp.dropped++
		case err == nil && res.epoch == lp.epoch:
			lp.done = append(lp.done, res)
			lp.r.logf("info", "cleaned up %d characters while recording in %v (%d prompt tokens, %d cached)",
				utf8.RuneCountInString(res.src), time.Since(start).Round(time.Millisecond), res.usage.PromptTokens, res.usage.CachedTokens)
		case err != nil:
			lp.r.logf("warning", "the live clean-up stops, Stop cleans the whole text: %v", err)
			lp.closed = true
			lp.blocks, lp.done = nil, nil
		}
		close(f.done)
		lp.cond.Broadcast()
		lp.mu.Unlock()
		cancel()
	}
}

// note (luojiaxuan): clean asks the language model to clean job's stretch, after before (the
// blocks' cleaned text, "" for the first block).
func (lp *livePolish) clean(ctx context.Context, job liveResult, before string, flight *liveFlight) (liveResult, error) {
	r := lp.r
	in := r.transcribeInput(job.src)
	in.Before = before
	parts, err := compose.BuildParts(in)
	if err != nil {
		return job, err
	}
	job.parts = parts
	lp.mu.Lock()
	flight.parts = slices.Clone(parts)
	flight.partIndex = 0
	flight.delta.Reset()
	lp.notifyFlightLocked(flight)
	lp.mu.Unlock()
	for partIndex, p := range parts {
		if err := ctx.Err(); err != nil {
			return job, err
		}
		lp.mu.Lock()
		lp.asked++
		lp.mu.Unlock()
		req := r.m.cfg.llmDeadlines(llm.Request{Messages: llmMessages(p.Output), Skill: llm.SkillTranscribeEnhance, InputChars: job.chars},
			utf8.RuneCountInString(p.Text), r.llmFloor(r.m.cfg.LLMTimeout))
		lp.mu.Lock()
		flight.partIndex = partIndex
		flight.delta.Reset()
		lp.mu.Unlock()
		// A retry may discard this private part only before finish adopts the flight.
		req.OnDelta = func(delta string) {
			if delta == "" {
				return
			}
			lp.mu.Lock()
			if !flight.detached && !flight.streamClosed && flight.partIndex == partIndex {
				flight.delta.WriteString(delta)
				lp.notifyFlightLocked(flight)
			}
			lp.mu.Unlock()
		}
		req.RetryPrivateStream = func() bool {
			lp.mu.Lock()
			defer lp.mu.Unlock()
			if flight.detached || flight.adopted || flight.streamClosed || flight.partIndex != partIndex {
				return false
			}
			flight.delta.Reset()
			lp.notifyFlightLocked(flight)
			return true
		}
		reqCtx, cancel := context.WithTimeout(ctx, req.Timeout)
		resp, err := lp.llm.Chat(reqCtx, req) // note (luojiaxuan): no picture and no snapshot change: newLivePolish runs only without a picture
		cancel()
		if err != nil {
			return job, err
		}
		answer := compose.CleanOutput(resp.Text, p.Text)
		job.answers = append(job.answers, answer)
		job.usage = job.usage.Add(resp.Usage)
		lp.mu.Lock()
		if !flight.detached {
			flight.answers = append(flight.answers, answer)
			flight.partIndex = partIndex + 1
			flight.delta.Reset()
			lp.notifyFlightLocked(flight)
		}
		lp.mu.Unlock()
	}
	return job, nil
}

func (lp *livePolish) notifyFlightLocked(flight *liveFlight) {
	select {
	case flight.updates <- struct{}{}:
	default:
	}
}
