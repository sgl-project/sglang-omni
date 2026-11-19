package session

import (
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// earlyClean runs the clean-up of the stretches the speech model
// finalised while the recording went on (asr.EventPiece), one request
// per stretch, so Stop leaves only the tail to clean. add is the
// stream's reader goroutine's, work the worker's, drain the session
// goroutine's (which runs after the reader returned); mu guards the
// rest.
type earlyClean struct {
	r          *run
	queue      chan segJob
	workerDone chan struct{}
	workerOnce sync.Once

	mu      sync.Mutex
	stable  string // the last EventPiece's text
	segs    []*segResult
	failed  error // the first request's failure
	broken  bool  // an event did not extend the text so far: give up
	gotWork bool  // a stretch was queued (the worker runs)
}

type segJob struct {
	sep, text string // the stretch and what joined it to the one before
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
	return &earlyClean{r: r, queue: make(chan segJob, 256), workerDone: make(chan struct{})}
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
	if e.broken || !strings.HasPrefix(stable, e.stable) {
		e.broken = true
		e.mu.Unlock()
		return
	}
	delta := stable[len(e.stable):]
	e.stable = stable
	e.mu.Unlock()
	sep := ""
	if rest, ok := strings.CutPrefix(delta, " "); ok {
		sep, delta = " ", rest
	}
	text := strings.TrimSpace(delta)
	if text == "" {
		return
	}
	e.mu.Lock()
	e.gotWork = true
	e.mu.Unlock()
	e.workerOnce.Do(func() { go e.work() })
	select {
	case e.queue <- segJob{sep: sep, text: text}:
	default:
		// The reader must not block on the clean-up; this much audio is
		// cleaned at Stop as without the events.
		e.mu.Lock()
		e.broken = true
		e.mu.Unlock()
	}
}

// work cleans the queued stretches in order, one request at a time,
// until the queue closes or the session ends.
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
	for job := range e.queue {
		e.clean(job)
		if r.ctx.Err() != nil {
			return
		}
	}
}

func (e *earlyClean) clean(job segJob) {
	r := e.r
	replaced := r.spoken(job.text)
	parts, err := compose.BuildParts(r.transcribeInput(replaced))
	if err != nil {
		e.fail(err)
		e.mu.Lock()
		e.segs = append(e.segs, &segResult{sep: job.sep, replaced: replaced, done: true})
		e.mu.Unlock()
		return
	}
	res := &segResult{sep: job.sep, replaced: replaced, parts: parts}
	// InputChars is the dictation's length so far, the nearest to the
	// whole text's count a request after Stop would carry (§5.27c).
	e.mu.Lock()
	chars := utf8.RuneCountInString(e.stable)
	e.mu.Unlock()
	for _, p := range parts {
		req := llm.Request{Messages: llmMessages(p.Output), Skill: llm.SkillTranscribeEnhance, InputChars: chars}
		resp, err := r.llmChat(r.m.cfg.llmDeadlines(req, utf8.RuneCountInString(p.Text), r.llmFloor(r.m.cfg.LLMTimeout)))
		if r.ctx.Err() != nil {
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

// drain waits for the worker and gives what the early clean-up covered:
// the parts and answers of the stretches whose joined text is a prefix
// of text, and the tail after them. ok is false when nothing usable was
// cleaned (no events, a broken join, no language model): the whole text
// is cleaned as without the events. Only the session goroutine calls
// it, after the stream's reader returned.
func (e *earlyClean) drain(text string) (parts []compose.Part, answers []string, tail string, failed error, ok bool) {
	e.mu.Lock()
	got := e.gotWork
	e.mu.Unlock()
	if !got {
		return nil, nil, "", nil, false
	}
	close(e.queue)
	<-e.workerDone
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.broken {
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
