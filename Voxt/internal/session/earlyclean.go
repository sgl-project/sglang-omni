package session

import (
	"context"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
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

	mu        sync.Mutex
	stable    string // the last EventPiece's text
	jobs      []segJob
	covered   string // the stretches joined as recognised, aliases replaced
	chars     int    // covered's length in characters
	segs      []*segResult
	failed    error // the first request's failure
	broken    bool  // an event did not extend the text so far: give up
	closed    bool  // the recording ended: no more stretches
	finalized bool  // the worker may no longer publish
}

type segJob struct {
	sep, replaced string // what joined the stretch to the one before, and the stretch with the dictionary's aliases replaced
}

// segResult is one stretch's clean-up: its parts and their answers, in
// order (a failed request keeps the stretch's text, as a failed piece's
// clean-up does). A stretch that would not compose has no parts: it
// stops the covered prefix, and Stop cleans it with the tail.
type segResult struct {
	sep         string
	replaced    string // the stretch with the dictionary's aliases replaced
	parts       []compose.Part
	answers     []string
	done        bool
	completedAt time.Time
	publishedAt time.Time
}

func newEarlyClean(r *run) *earlyClean {
	r.m.mu.Lock()
	llmCtx := r.llmCtx
	r.m.mu.Unlock()
	ctx, cancel := context.WithCancel(llmCtx)
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

func (r *run) earlyCleanup() *earlyClean {
	r.earlyMu.Lock()
	defer r.earlyMu.Unlock()
	if r.early == nil {
		r.early = newEarlyClean(r)
	}
	return r.early
}

func (r *run) finalizeEarlyCleanup() {
	r.earlyMu.Lock()
	e := r.early
	r.earlyMu.Unlock()
	if e != nil {
		e.finalize()
	}
}

// stop gives the early clean-up up (the whole text is cleaned at Stop,
// as without the events) and wakes the worker.
func (e *earlyClean) stop() {
	e.mu.Lock()
	if !e.finalized {
		e.broken = true
	}
	e.mu.Unlock()
	e.cancel()
}

func (e *earlyClean) finalize() {
	e.mu.Lock()
	e.finalized = true
	e.closed = true
	e.cancel()
	e.cond.Broadcast()
	e.mu.Unlock()
}

// hearPiece takes the final text so far of a recording that finalises
// stretches while it continues (asr.EventPiece): the stretch since the
// last event goes to the early clean-up (Transcribe with the clean-up
// on only; the final text repeats these stretches as its prefix).
//
// note (luojiaxuan): not when the user has the clean-up wait for Stop
// (settings.Transcribe.CleanAfterStop); the live clean-up ends here.
func (r *run) hearPiece(stable string) {
	r.endLive()
	if r.req.Kind != store.KindTranscribe {
		return
	}
	select {
	case <-r.svcResolved:
	default:
		return
	}
	if r.settings.Features.Transcribe.CleanAfterStop || !r.transcribeCleanup() {
		return
	}
	r.head.hearFinal(stable)
	if r.head.enabled.Load() && asr.Duration(r.pcm.bytes(), asr.SampleRate) < earlyCleanAfter {
		return // not yet long: the whole text goes in one request, which knows what is said later
	}
	r.earlyCleanup().add(stable)
}

// earlyCleanAfter is how long a recording is when its finalised stretches
// are cleaned one by one, for a service that the head start serves.
const earlyCleanAfter = 60 * time.Second

// earlyPieces says whether stretches sooner are worth it: the head start
// puts them before the live text, and a Final has less left to recognise.
// Each stretch costs the speech model its dictionary, so only a short
// one.
func (r *run) earlyPieces() bool {
	return r.head.enabled.Load() && r.head.under == 0 && !r.settings.Audio.LiveResult &&
		len(r.glossary()) <= compose.GlossaryAllUpTo && len(r.hints()) <= compose.GlossaryAllUpTo
}

// enableEarlyPieces applies the finalized eligibility decision to an
// already-open stream when it could not be known at Open time.
func (r *run) enableEarlyPieces() {
	if r.stream == nil || r.init.split && !r.init.cleanupReady || !r.earlyPieces() {
		return
	}
	if stream, ok := r.stream.(asr.EarlyPiecesEnabler); ok {
		stream.EnableEarlyPieces()
	}
}

// working says whether stretches are still being cleaned.
func (e *earlyClean) working() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return !e.broken && len(e.segs) < len(e.jobs)
}

// add queues the stretch the text grew by since the last event. The
// stretches join as the speech model joined them (a space between
// Latin-script sides, nothing otherwise): an event that is not the last
// one plus one stretch breaks that, and the whole text is cleaned at
// Stop as without the events.
func (e *earlyClean) add(stable string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.finalized || e.closed || e.ctx.Err() != nil {
		return
	}
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
	r.m.mu.Lock()
	client, llmInfo := r.svc.LLM, r.svc.LLMInfo
	r.m.mu.Unlock()
	if client == nil {
		return // no clean-up at all: the text goes on as recognised
	}
	// Later text can add terms to a large glossary, so only whole-text
	// clean-up can give every part the same prompt.
	if len(r.glossary()) > compose.GlossaryAllUpTo {
		e.stop()
		return
	}
	thinkOffUnder := r.thinkOffUnder(llmInfo.ID)
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
		if !e.finalized {
			e.segs = append(e.segs, &segResult{sep: job.sep, replaced: job.replaced, done: true})
		}
		e.mu.Unlock()
		return
	}
	res := &segResult{sep: job.sep, replaced: job.replaced, parts: parts}
	for index, p := range parts {
		if index > 0 {
			e.mu.Lock()
			if e.finalized || e.closed {
				if !e.finalized {
					e.segs = append(e.segs, res)
				}
				e.mu.Unlock()
				return
			}
			e.mu.Unlock()
		}
		req := llm.Request{Messages: llmMessages(p.Output), Skill: llm.SkillTranscribeEnhance, InputChars: chars}
		if !r.cleanupMayContinue() {
			return
		}
		resp, err := r.llmChatContext(e.ctx, r.m.cfg.llmDeadlines(req, utf8.RuneCountInString(p.Text), r.llmFloor(r.m.cfg.LLMTimeout)))
		if e.ctx.Err() != nil {
			return
		}
		if err != nil {
			e.fail(err)
			res.answers = append(res.answers, p.Text)
			continue
		}
		res.answers = append(res.answers, compose.CleanOutput(resp.Text, p.Text, p.Custom()))
	}
	res.done, res.completedAt = true, r.now()
	e.mu.Lock()
	if !e.finalized {
		res.publishedAt = r.now()
		e.segs = append(e.segs, res)
		e.cond.Broadcast()
	}
	e.mu.Unlock()
}

func (e *earlyClean) fail(err error) {
	e.mu.Lock()
	if !e.finalized && e.failed == nil {
		e.failed = err
	}
	e.mu.Unlock()
}

// rest is what Stop would leave of text to clean after the stretches
// cleaned while the recording went on, and false when text does not start
// with them.
func (e *earlyClean) rest(text string) (string, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.broken {
		return text, true
	}
	return strings.CutPrefix(text, e.covered)
}

// completedCompatible returns only a full result whose segments were
// already complete and published before the Stop-relative deadline. It
// never waits for the worker or returns a cleaned-prefix/raw-tail mix.
func (e *earlyClean) completedCompatible(text string, deadline time.Time) (cleanupOutput, time.Time, bool) {
	e.mu.Lock()
	if e.broken || e.failed != nil || e.covered == "" || e.covered != text || len(e.segs) != len(e.jobs) || len(e.segs) == 0 {
		e.mu.Unlock()
		return cleanupOutput{}, time.Time{}, false
	}
	var parts []compose.Part
	var answers []string
	var completedAt, publishedAt time.Time
	for _, segment := range e.segs {
		if !segment.done || len(segment.parts) == 0 || !segment.completedAt.Before(deadline) || !segment.publishedAt.Before(deadline) || len(segment.answers) != len(segment.parts) {
			e.mu.Unlock()
			return cleanupOutput{}, time.Time{}, false
		}
		parts = append(parts, segment.parts...)
		answers = append(answers, segment.answers...)
		if segment.completedAt.After(completedAt) {
			completedAt = segment.completedAt
		}
		if segment.publishedAt.After(publishedAt) {
			publishedAt = segment.publishedAt
		}
	}
	e.finalized, e.closed = true, true
	e.mu.Unlock()
	e.cancel()
	e.cond.Broadcast()
	if len(parts) == 0 {
		return cleanupOutput{}, time.Time{}, false
	}
	return cleanupOutput{text: compose.JoinParts(parts, answers), profile: parts[0].Profile(), completedAt: completedAt}, publishedAt, true
}

// drain abandons a mismatching prefix without waiting. Matching work keeps
// its existing drain behavior; finalized workers cannot publish late results.
func (e *earlyClean) drain(text string) (parts []compose.Part, answers []string, tail string, failed error, completedAt time.Time, ok bool) {
	defer e.cancel()
	e.mu.Lock()
	if len(e.jobs) == 0 {
		e.mu.Unlock()
		return nil, nil, "", nil, time.Time{}, false
	}
	fallback := e.broken || e.covered == "" || !strings.HasPrefix(text, e.covered)
	e.closed = true
	if fallback {
		e.finalized = true
		e.mu.Unlock()
		e.cancel()
		e.cond.Broadcast()
		return nil, nil, "", nil, time.Time{}, false
	}
	cancelWorker := len(e.segs) < len(e.jobs)-1
	if cancelWorker {
		e.finalized = true
	}
	e.mu.Unlock()
	e.cond.Broadcast()
	if cancelWorker {
		// Finalized workers cannot publish further results. Cancellation
		// must not delay cleanup of the remaining authoritative text.
		e.cancel()
	} else {
		<-e.workerDone
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.finalized = true
	if e.broken {
		return nil, nil, "", nil, time.Time{}, false
	}
	covered := ""
	for _, segment := range e.segs {
		if !segment.done || len(segment.parts) == 0 {
			break
		}
		covered += segment.sep + segment.replaced
		parts = append(parts, segment.parts...)
		answers = append(answers, segment.answers...)
		if segment.completedAt.After(completedAt) {
			completedAt = segment.completedAt
		}
	}
	if !strings.HasPrefix(text, covered) {
		e.r.logf("debug", "early clean-up does not apply: %d stretches, %d covered characters, text %d characters", len(e.segs), len([]rune(covered)), len([]rune(text)))
		return nil, nil, "", nil, time.Time{}, false
	}
	if covered == "" {
		return nil, nil, "", nil, time.Time{}, false
	}
	return parts, answers, text[len(covered):], e.failed, completedAt, true
}
