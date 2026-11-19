package session

import (
	"context"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// earlyClean runs the clean-up of the stretches the speech model
// finalised while the recording went on (asr.EventPiece), so Stop leaves
// the remaining early work and tail to clean. add is the
// stream's reader goroutine's, work the worker's, drain the session
// goroutine's (which runs after the reader returned); mu guards the
// rest, and cond (on mu) wakes the worker for new stretches, the
// recording's end (closed) or a cancellation.
type earlyClean struct {
	r          *run
	cond       *sync.Cond
	workerDone chan struct{}
	ctx        context.Context
	cancel     context.CancelFunc

	mu      sync.Mutex
	stable  string // the last EventPiece's text
	jobs    []segJob
	covered string // the stretches joined as recognised, aliases replaced
	chars   int    // covered's length in characters
	segs    []*segResult
	failed  error // the first request's failure
	broken  bool  // an event did not extend the text so far: give up
	closed  bool  // the recording ended: no more stretches
}

type segJob struct {
	sep, replaced string // what joined the stretch to the one before, and the stretch with the dictionary's aliases replaced
}

// segResult is one stretch's clean-up: its parts and their answers, in
// order (a failed request keeps the stretch's text, as a failed piece's
// clean-up does). A stretch that would not compose has no parts: it
// stops the covered prefix, and Stop cleans it with the tail.
type segResult struct {
	sep      string
	replaced string // the stretch with the dictionary's aliases replaced
	parts    []compose.Part
	answers  []string
	done     bool
}

func newEarlyClean(r *run) *earlyClean {
	ctx, cancel := context.WithCancel(r.llmCtx)
	e := &earlyClean{r: r, workerDone: make(chan struct{}), ctx: ctx, cancel: cancel}
	e.cond = sync.NewCond(&e.mu)
	go func() {
		<-ctx.Done()
		e.mu.Lock()
		e.cond.Broadcast()
		e.mu.Unlock()
	}()
	return e
}

// stop gives the early clean-up up (the whole text is cleaned at Stop,
// as without the events) and wakes the worker.
func (e *earlyClean) stop() {
	e.mu.Lock()
	e.broken = true
	e.mu.Unlock()
	e.cancel()
}

// hearPiece takes the final text so far of a recording that finalises
// stretches while it continues (asr.EventPiece): the stretch since the
// last event goes to the early clean-up (Transcribe with the clean-up
// on only; the final text repeats these stretches as its prefix).
func (r *run) hearPiece(stable string) {
	if r.req.Kind != store.KindTranscribe || !r.settings.Features.Transcribe.Enhance {
		return
	}
	r.earlyOnce.Do(func() { r.early = newEarlyClean(r) })
	r.early.add(stable)
}

// add queues the stretch the text grew by since the last event. The
// stretches join as the speech model joined them (a space between
// Latin-script sides, nothing otherwise): an event that is not the last
// one plus one stretch breaks that, and the whole text is cleaned at
// Stop as without the events.
func (e *earlyClean) add(stable string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.broken || !strings.HasPrefix(stable, e.stable) {
		e.broken = true
		e.cancel()
		return
	}
	delta := stable[len(e.stable):]
	e.stable = stable
	sep := ""
	if rest, ok := strings.CutPrefix(delta, " "); ok {
		sep, delta = " ", rest
	}
	text := strings.TrimSpace(delta)
	if text == "" {
		return
	}
	replaced := e.r.spoken(text)
	e.jobs = append(e.jobs, segJob{sep: sep, replaced: replaced})
	e.covered += sep + replaced
	e.chars += utf8.RuneCountInString(sep + replaced)
	if len(e.jobs) == 1 {
		go e.work()
	}
	e.cond.Broadcast()
}

// work cleans the queued stretches in order, one request at a time,
// until the recording ends (drain closed it) or the session ends.
func (e *earlyClean) work() {
	defer close(e.workerDone)
	r := e.r
	select {
	case <-r.svcResolved:
	case <-r.ctx.Done():
		return
	}
	if r.svc.LLM == nil {
		return // no clean-up at all: the text goes on as recognised
	}
	// Later text can add terms to a large glossary, so only whole-text
	// clean-up can give every part the same prompt.
	if len(r.glossary()) > compose.GlossaryAllUpTo {
		e.stop()
		return
	}
	thinkOffUnder := 0
	for _, provider := range r.settings.Providers.LLM {
		if provider.ID == r.svc.LLMInfo.ID {
			thinkOffUnder = provider.Generation.ThinkingOffUnder
			break
		}
	}
	next := 0
	for {
		e.mu.Lock()
		for !e.closed && e.ctx.Err() == nil && next >= len(e.jobs) {
			e.cond.Wait()
		}
		if e.ctx.Err() != nil || (e.closed && next >= len(e.jobs)) {
			e.mu.Unlock()
			return
		}
		job := e.jobs[next]
		next++
		e.mu.Unlock()
		e.clean(job, thinkOffUnder)
	}
}

func (e *earlyClean) clean(job segJob, thinkOffUnder int) {
	r := e.r
	e.mu.Lock()
	chars := e.chars
	e.mu.Unlock()
	// A shorter prefix must not turn thinking off when the finished
	// dictation would leave it on.
	if thinkOffUnder > 0 && chars < thinkOffUnder {
		e.stop()
		return
	}
	parts, err := compose.BuildParts(r.transcribeInput(job.replaced))
	if err != nil {
		e.fail(err)
		e.mu.Lock()
		e.segs = append(e.segs, &segResult{sep: job.sep, replaced: job.replaced, done: true})
		e.mu.Unlock()
		return
	}
	res := &segResult{sep: job.sep, replaced: job.replaced, parts: parts}
	for _, p := range parts {
		req := llm.Request{Messages: llmMessages(p.Output), Skill: llm.SkillTranscribeEnhance, InputChars: chars}
		resp, err := r.llmChatContext(e.ctx, r.m.cfg.llmDeadlines(req, utf8.RuneCountInString(p.Text), r.llmFloor(r.m.cfg.LLMTimeout)))
		if e.ctx.Err() != nil {
			return
		}
		if err != nil {
			e.fail(err)
			res.answers = append(res.answers, p.Text)
			continue
		}
		res.answers = append(res.answers, compose.CleanOutput(resp.Text, p.Text))
	}
	res.done = true
	e.mu.Lock()
	e.segs = append(e.segs, res)
	e.mu.Unlock()
}

func (e *earlyClean) fail(err error) {
	e.mu.Lock()
	if e.failed == nil {
		e.failed = err
	}
	e.mu.Unlock()
}

func (e *earlyClean) finalMatches(text string) bool {
	e.mu.Lock()
	covered := e.covered
	e.mu.Unlock()
	return covered != "" && strings.HasPrefix(text, covered)
}

// drain waits for the worker and gives what the early clean-up covered:
// the parts and answers of the stretches whose joined text is a prefix
// of text, and the tail after them. ok is false when nothing usable was
// cleaned (no events, a broken join, no language model): the whole text
// is cleaned as without the events. Only the session goroutine calls
// it, after the stream's reader returned.
func (e *earlyClean) drain(text string) (parts []compose.Part, answers []string, tail string, failed error, ok bool) {
	defer e.cancel()
	e.mu.Lock()
	got := len(e.jobs) > 0
	broken := e.broken
	e.mu.Unlock()
	if !got {
		return nil, nil, "", nil, false
	}
	fallback := broken || !e.finalMatches(text)
	e.mu.Lock()
	e.closed = true
	e.mu.Unlock()
	e.cond.Broadcast()
	if fallback {
		e.cancel()
	}
	<-e.workerDone
	e.mu.Lock()
	defer e.mu.Unlock()
	if fallback || e.broken {
		return nil, nil, "", nil, false
	}
	covered := ""
	for _, s := range e.segs {
		if !s.done || len(s.parts) == 0 {
			break
		}
		covered += s.sep + s.replaced
		parts = append(parts, s.parts...)
		answers = append(answers, s.answers...)
	}
	if covered == "" || !strings.HasPrefix(text, covered) {
		e.r.logf("debug", "early clean-up does not apply: %d stretches, %d covered characters, text %d characters", len(e.segs), len([]rune(covered)), len([]rune(text)))
		return nil, nil, "", nil, false
	}
	return parts, answers, text[len(covered):], e.failed, true
}
