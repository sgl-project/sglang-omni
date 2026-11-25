package session

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type cleanupOutput struct {
	text        string
	profile     string
	err         error
	completedAt time.Time
	generation  uint64
}

type cleanupRequest struct {
	generation uint64
	ctx        context.Context
	thinking   *llm.Thinking
}

type cleanupChoice uint8

const (
	cleanupUndecided cleanupChoice = iota
	cleanupComplete
	cleanupRaw
	cleanupRetry
	cleanupPending
	cleanupCancelled
)

type cleanupOutcome string

const (
	cleanupOutcomeCleaned      cleanupOutcome = "cleaned"
	cleanupOutcomeManualSkip   cleanupOutcome = "manual_skip"
	cleanupOutcomeSoftDeadline cleanupOutcome = "soft_deadline"
	cleanupOutcomeLLMError     cleanupOutcome = "llm_error"
	cleanupOutcomeCancelled    cleanupOutcome = "cancelled"
)

func (r *run) cleanupMayContinueLocked() bool {
	return r.cleanupDecisionOpenLocked() && !r.skipped.Load() && !r.cancelled.Load() && !r.rethink.Load()
}

func (r *run) cleanupMayContinue() bool {
	r.m.mu.Lock()
	defer r.m.mu.Unlock()
	return r.cleanupMayContinueLocked()
}

// spoken is the recognised text with the dictionary's aliases replaced
// by their terms (dict.Replace), when the dictionary is on: before the
// clean-up, so it holds without a language model too. The snapshot's
// Raw, and the history's, stay as recognised.
func (r *run) spoken(raw string) string {
	if !r.settings.Dictionary.Enabled {
		return raw
	}
	text := dict.Replace(raw, r.terms)
	if text != raw {
		r.logf("info", "the dictionary's aliases replaced: %s", r.logText(text))
	}
	return text
}

// transcribe cleans up text with the language model (unless the feature
// has it off or no model is ready) and delivers it. A long dictation is
// cleaned up in pieces (compose.BuildParts); where the model fails, the
// text goes on as recognised, with a warning.
func (r *run) transcribe(text string) {
	result := text
	var problem *Problem
	cleanupRequested := r.transcribeCleanup() && r.svc.LLM != nil
	ordinaryTranscribe := r.req.Kind == store.KindTranscribe && !r.req.Note && !r.req.Try && r.req.Text == nil
	policy := r.m.cfg.CleanupLatency
	softDeadline := policy.AutoFallbackAfter > 0 && ordinaryTranscribe
	if ordinaryTranscribe {
		level := "debug"
		if softDeadline {
			level = "info"
		}
		r.logf(level, "cleanup requested=%t", cleanupRequested)
		if !cleanupRequested {
			r.logf(level, "cleanup outcome=not_requested")
		}
	}
	if cleanupRequested {
		card := r.settings.Features.Transcribe.ShowCard && !r.req.Note && !r.req.Try // a note is saved, not shown
		r.m.update(r, func(s *Snapshot) { s.State, s.CanSkip, s.Card, s.Spoken = Composing, true, card, text })
		llmStart := r.now()
		var deadlineAt time.Time
		if softDeadline {
			stopAt := r.stoppedAt()
			deadlineAt = stopAt.Add(policy.AutoFallbackAfter)
			if stopAt.IsZero() {
				deadlineAt = r.now().Add(policy.AutoFallbackAfter)
			}
			r.m.setCleanupDeadline(r, deadlineAt)
		}
		if softDeadline && r.now().Before(deadlineAt) {
			stopTimer := r.m.cfg.AfterFunc(deadlineAt.Sub(r.now()), func() { r.m.notifyCleanupDeadline(r) })
			defer stopTimer()
		}
		startCleanup := func() {
			r.m.mu.Lock()
			if r.cleanupChoice != cleanupUndecided || r.cleanupWorkerRunning || r.cancelled.Load() {
				r.m.mu.Unlock()
				return
			}
			r.cleanupWorkerRunning = true
			request := cleanupRequest{generation: r.cleanupGeneration, ctx: r.llmCtx, thinking: r.think}
			r.m.mu.Unlock()
			go func() {
				output := r.cleanTranscribe(text, request)
				r.m.publishCleanupCandidate(r, output, output.generation, time.Time{})
			}()
		}
		if softDeadline && !r.now().Before(deadlineAt) {
			r.earlyMu.Lock()
			early := r.early
			r.earlyMu.Unlock()
			if early != nil {
				if candidate, publishedAt, ok := early.completedCompatible(text, deadlineAt); ok {
					r.m.mu.Lock()
					generation := r.cleanupGeneration
					r.m.mu.Unlock()
					r.m.publishCleanupCandidate(r, candidate, generation, publishedAt)
				}
			}
			r.m.notifyCleanupDeadline(r)
		} else {
			startCleanup()
		}
		var output cleanupOutput
		decisionMade := false
		for !decisionMade {
			select {
			case <-r.cleanupWake:
				choice, candidate := r.m.chooseCleanup(r)
				switch choice {
				case cleanupComplete, cleanupRaw:
					output = candidate
					if choice == cleanupRaw {
						output = cleanupOutput{text: text}
					}
					decisionMade = true
				case cleanupRetry:
					startCleanup()
				}
			case <-r.ctx.Done():
				choice, _ := r.m.chooseCleanup(r)
				if choice == cleanupCancelled {
					r.endCancelled(nil)
					return
				}
			}
		}
		llmTime := r.now().Sub(llmStart)
		r.m.mu.Lock()
		outcome := r.cleanupOutcome
		r.m.mu.Unlock()
		r.m.updateIf(r, func(s *Snapshot) bool {
			if r.cancelled.Load() || r.ctx.Err() != nil {
				return false
			}
			s.Profile, s.Timings.LLM = output.profile, llmTime
			if outcome == cleanupOutcomeSoftDeadline || outcome == cleanupOutcomeManualSkip {
				s.Draft = ""
			}
			return true
		})
		if r.cancelled.Load() || r.ctx.Err() != nil {
			r.endCancelled(nil)
			return
		}
		result = output.text
		switch {
		case outcome == cleanupOutcomeManualSkip:
			r.logf("info", "the language model was skipped after %v: the text goes on as recognised", llmTime.Round(time.Millisecond))
			result = text
		case outcome == cleanupOutcomeSoftDeadline:
			result = text
		case outcome == cleanupOutcomeCancelled:
			r.endCancelled(nil)
			return
		case output.err != nil:
			r.logf("warning", "language model %s (%s) failed after %v, the text goes on as recognised: %v", r.svc.LLMInfo.ID, r.svc.LLMInfo.Model, llmTime.Round(time.Millisecond), output.err)
			problem = &Problem{Code: ProblemLLM, Err: output.err, Degraded: true}
		case result == "":
			r.finish(Done, &Problem{Code: ProblemNoSpeech}, "", finishOpts{})
			return
		default:
			r.logf("info", "composed %s in %v", r.logText(result), llmTime.Round(time.Millisecond))
		}
	}
	r.finish(Done, problem, r.expandSnippets(text, result), r.transcribeOpts())
}

// transcribeOpts are how a Transcribe result is delivered.
func (r *run) transcribeOpts() finishOpts {
	var o finishOpts
	if r.settings.Features.Transcribe.ShowCard {
		o.card = deliver.ReasonCard
	}
	return o
}

func (r *run) cleanTranscribe(text string, request cleanupRequest) cleanupOutput {
	output := cleanupOutput{text: text, completedAt: r.now()}
	if r.cleanupMayContinue() && request.ctx.Err() == nil {
		output = r.cleanUpFirstContext(text, request)
	}
	for {
		if nextRequest, ok := r.takeRethink(); ok {
			request = nextRequest
			output = r.cleanUpOutputContext(text, request.thinking, request.ctx)
			output.generation = request.generation
			continue
		}
		if !r.cleanupMayContinue() {
			return cleanupOutput{text: text, completedAt: r.now(), generation: request.generation}
		}
		if !r.useFallback(output.err) {
			output.generation = request.generation
			return output
		}
		// The fallback joins the re-ask loop: it keeps the level chosen,
		// and a Rethink while it streams asks it again.
		if !r.swapFallback(output.err) {
			// A Rethink may have won after useFallback checked eligibility.
			// Leave it pending for the loop's single generation-capturing path.
			continue
		}
		nextRequest, ok := r.currentCleanupRequest()
		if !ok {
			continue
		}
		request = nextRequest
		output = r.cleanUpOutputContext(text, request.thinking, request.ctx)
		output.generation = request.generation
	}
}

// cleanUpFirstContext is a dictation's first clean-up: what the early clean-up
// finished while the recording went on is not asked again, only the
// tail left at Stop is (earlyclean.go). A Rethink reruns cleanUp on the
// whole text instead.
//
// note (luojiaxuan): without the speech model's stretches, the clean-ups of
// the live text (livepolish.go) count the same way; their tail goes with the
// end of the cleaned text before it (liveContext), after the last sentences
// of that text that are asked again (redo).
func (r *run) cleanUpFirstContext(text string, request cleanupRequest) cleanupOutput {
	parts, answers, tail, failed, completedAt, ok := r.earlyCleanup().drain(text)
	before := ""
	livePath := false
	var flight *liveFlight
	if !ok {
		r.liveOnce.Do(func() {})
		var redo string
		if r.live != nil {
			livePath = true
			parts, answers, redo, tail, flight = r.live.finish(text)
			completedAt = r.now()
			if flight != nil {
				if result, adopted := r.live.adopt(text, flight, request); adopted {
					r.live.mu.Lock()
					blocks := slices.Clone(r.live.blocks)
					r.live.mu.Unlock()
					parts, answers, tail = laid(append(blocks, result), text)
					redo = ""
					completedAt = r.now()
				}
			}
		}
		if len(answers) == 0 && redo == "" {
			if !r.cleanupMayContinue() {
				return cleanupOutput{text: text, completedAt: r.now()}
			}
			if livePath && !r.publishCleanupDraft(request, "") {
				return cleanupOutput{text: text, completedAt: r.now()}
			}
			return r.cleanUpOutputContext(text, nil, request.ctx)
		}
		before = liveContext(compose.JoinParts(parts, answers))
		tail = joinRedo(redo, tail)
	} else if len(answers) > 0 {
		r.logf("info", "%d characters were cleaned while the recording went on", len([]rune(text))-len([]rune(tail)))
	}
	if strings.TrimSpace(tail) == "" {
		return cleanupOutput{text: compose.JoinParts(parts, answers), profile: parts[0].Profile(), err: failed, completedAt: completedAt}
	}
	if !r.cleanupMayContinue() {
		return cleanupOutput{text: text, completedAt: r.now()}
	}
	if livePath {
		retained := r.draftSnippets(compose.JoinParts(parts, answers), parts)
		if !r.publishCleanupDraft(request, retained) {
			return cleanupOutput{text: text, completedAt: r.now()}
		}
	}
	in := r.transcribeInput(tail)
	in.Before = before
	tailParts, err := compose.BuildParts(in)
	if err != nil {
		return cleanupOutput{text: text, err: err, completedAt: r.now()}
	}
	result, profile, err := r.cleanPartsContext(text, append(parts, tailParts...), answers, nil, request.ctx)
	if err == nil {
		err = failed
	}
	return cleanupOutput{text: result, profile: profile, err: err, completedAt: r.now()}
}

func (r *run) publishCleanupDraft(request cleanupRequest, draft string) bool {
	return r.m.updateIf(r, func(s *Snapshot) bool {
		if s.State != Composing || request.ctx == nil || request.ctx.Err() != nil || r.ctx.Err() != nil ||
			r.cleanupGeneration != request.generation || !r.cleanupMayContinueLocked() {
			return false
		}
		s.Draft = draft
		return true
	})
}

// note (luojiaxuan): joinRedo is the cleaned sentences asked again (redo,
// without the snippets' marks, which the request marks again) and the text
// left after them, joined as the clean-up joins Latin text (a space) or
// Chinese (nothing).
func joinRedo(redo, tail string) string {
	if redo = snippet.Unmark(redo); redo == "" {
		return tail
	}
	if last, _ := utf8.DecodeLastRuneInString(redo); last < utf8.RuneSelf {
		return redo + " " + tail
	}
	return redo + tail
}

// cleanUp runs the language model over text, a request per part, each
// answer streaming into the snapshot's Draft after those before it. A
// part that fails keeps its text as recognised; err is the first
// failure (the text itself when the request could not be composed).
// think overrides the service's thinking (nil: its setting).
func (r *run) cleanUp(text string, think *llm.Thinking) (string, string, error) {
	output := r.cleanUpOutput(text, think)
	return output.text, output.profile, output.err
}

func (r *run) cleanUpOutput(text string, think *llm.Thinking) cleanupOutput {
	return r.cleanUpOutputContext(text, think, r.currentLLMContext())
}

func (r *run) cleanUpOutputContext(text string, think *llm.Thinking, ctx context.Context) cleanupOutput {
	parts, err := compose.BuildParts(r.transcribeInput(text))
	if err != nil {
		return cleanupOutput{text: text, err: err, completedAt: r.now()}
	}
	if len(parts) > 1 {
		r.logf("info", "cleaning up %d characters in %d pieces", len([]rune(text)), len(parts))
	}
	result, profile, err := r.cleanPartsContext(text, parts, nil, think, ctx)
	return cleanupOutput{text: result, profile: profile, err: err, completedAt: r.now()}
}

// cleanParts runs the requests of the parts after len(answers) (those
// before are answered already, by the early clean-up), each answer
// streaming into the snapshot's Draft after those before it, and joins
// them all. err is the first failure (the text itself when the request
// could not be composed). think overrides the service's thinking (nil:
// its setting).
func (r *run) cleanParts(text string, parts []compose.Part, answers []string, think *llm.Thinking) (string, string, error) {
	return r.cleanPartsContext(text, parts, answers, think, r.currentLLMContext())
}

func (r *run) cleanPartsContext(text string, parts []compose.Part, answers []string, think *llm.Thinking, ctx context.Context) (string, string, error) {
	var failed error
	for i := len(answers); i < len(parts); i++ {
		p := parts[i]
		if !r.cleanupMayContinue() {
			return text, p.Profile(), context.DeadlineExceeded
		}
		draft := newDrafter(r, ctx, parts[:i+1], answers)
		req := llm.Request{Messages: llmMessages(p.Output), Skill: llm.SkillTranscribeEnhance, Thinking: think, InputChars: utf8.RuneCountInString(text), OnDelta: draft.add}
		resp, err := r.llmChatContext(ctx, r.m.cfg.llmDeadlines(req, utf8.RuneCountInString(p.Text), r.llmFloor(r.m.cfg.LLMTimeout)))
		switch {
		case r.ctx.Err() != nil || r.skipped.Load() || r.rethink.Load():
			return text, p.Profile(), err
		case err != nil:
			if len(parts) > 1 {
				err = fmt.Errorf("piece %d of %d: %w", i+1, len(parts), err)
				r.logf("warning", "%v: kept as recognised", err)
			}
			if failed == nil {
				failed = err
			}
			answers = append(answers, p.Text)
		default:
			answers = append(answers, compose.CleanOutput(resp.Text, p.Text))
			if u := resp.Usage; u.PromptTokens > 0 && !u.Estimated {
				r.logf("info", "piece %d of %d: %d prompt tokens, %d cached", i+1, len(parts), u.PromptTokens, u.CachedTokens)
			}
		}
		draft.done(answers)
	}
	return compose.JoinParts(parts, answers), parts[0].Profile(), failed
}

// llmMessages are a composed request's messages for the model.
func llmMessages(out compose.Output) []llm.Message {
	msgs := make([]llm.Message, 0, len(out.History)+2)
	for _, m := range out.Messages() {
		msgs = append(msgs, llm.Message{Role: llm.Role(m.Role), Content: m.Content})
	}
	return msgs
}

// drafter streams the clean-up of a part into the snapshot's Draft,
// after the answers of the parts before it, every draftEvery at most.
type drafter struct {
	r       *run
	ctx     context.Context
	parts   []compose.Part
	answers []string
	buf     strings.Builder
	last    time.Time
}

func newDrafter(r *run, ctx context.Context, parts []compose.Part, answers []string) *drafter {
	return &drafter{r: r, ctx: ctx, parts: parts, answers: answers[:len(answers):len(answers)]}
}

func (d *drafter) add(delta string) {
	if d.ctx.Err() != nil || d.r.ctx.Err() != nil || d.r.skipped.Load() {
		return
	}
	d.buf.WriteString(delta)
	if now := time.Now(); now.Sub(d.last) >= draftEvery {
		d.last = now
		d.publish()
	}
}

// publish puts the draft so far in the snapshot, without the contract's
// <output> tags (PartialOutput, with the part's own text as CleanOutput
// gets it).
func (d *drafter) publish() {
	if d.ctx.Err() != nil || d.r.ctx.Err() != nil || d.r.skipped.Load() {
		return
	}
	part := d.parts[len(d.parts)-1]
	draft := compose.JoinParts(d.parts, append(d.answers, compose.PartialOutput(d.buf.String(), part.Text)))
	draft = d.r.draftSnippets(draft, d.parts)
	d.r.m.update(d.r, func(s *Snapshot) {
		if d.ctx.Err() == nil && d.r.ctx.Err() == nil && !d.r.skipped.Load() && d.r.cleanupDecisionOpenLocked() {
			s.Draft = draft
		}
	})
}

// done puts the answers of the parts so far in the snapshot.
func (d *drafter) done(answers []string) {
	if d.ctx.Err() != nil || d.r.ctx.Err() != nil || d.r.skipped.Load() {
		return
	}
	draft := compose.JoinParts(d.parts, answers)
	draft = d.r.draftSnippets(draft, d.parts)
	d.r.m.update(d.r, func(s *Snapshot) {
		if d.ctx.Err() == nil && d.r.ctx.Err() == nil && !d.r.skipped.Load() && d.r.cleanupDecisionOpenLocked() {
			s.Draft = draft
		}
	})
}

func (r *run) currentCleanupRequest() (cleanupRequest, bool) {
	r.m.mu.Lock()
	defer r.m.mu.Unlock()
	if !r.cleanupMayContinueLocked() {
		return cleanupRequest{}, false
	}
	return cleanupRequest{generation: r.cleanupGeneration, ctx: r.llmCtx, thinking: r.think}, true
}

// takeRethink consumes one pending request-control event and captures all
// request state under the same lock, so a generation cannot borrow a newer
// thinking level or a context that belongs to another Rethink.
func (r *run) takeRethink() (cleanupRequest, bool) {
	if !r.rethink.Load() || r.cancelled.Load() || r.skipped.Load() || r.stopped.Load() {
		return cleanupRequest{}, false
	}
	r.m.mu.Lock()
	if !r.rethink.Load() || r.cancelled.Load() || r.skipped.Load() || r.stopped.Load() || !r.cleanupDecisionOpenLocked() {
		r.m.mu.Unlock()
		return cleanupRequest{}, false
	}
	r.rethink.Store(false)
	r.llmCtx, r.llmStop = context.WithCancel(r.ctx)
	request := cleanupRequest{generation: r.cleanupGeneration, ctx: r.llmCtx, thinking: r.think}
	r.m.mu.Unlock()
	r.logf("info", "asking the language model again at the thinking level chosen: %+v", request.thinking)
	return request, true
}

// askAgain is shared by Translate and Rewrite, which need only the restart
// side effect. Transcribe uses takeRethink to keep generation and request
// state together.
func (r *run) askAgain() bool {
	_, ok := r.takeRethink()
	if ok {
		r.m.update(r, func(s *Snapshot) { s.Draft, s.Title = "", "" })
	}
	return ok
}

// thinking is the level the requests ask with: Rethink's, else the
// session's (StartRequest.Thinking; nil: the service's setting).
func (r *run) thinking() *llm.Thinking {
	r.m.mu.Lock()
	defer r.m.mu.Unlock()
	return r.think
}
