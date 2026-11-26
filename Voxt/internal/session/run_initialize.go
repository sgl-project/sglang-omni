package session

import (
	"errors"
	"fmt"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func (r *run) resolve(speech bool) {
	if speech && r.req.Kind == store.KindTranscribe && !r.req.Note && !r.req.Try && r.req.Text == nil &&
		r.m.cfg.ResolveASR != nil && r.m.cfg.ResolveLLM != nil {
		r.resolveIndependent()
		return
	}
	r.resolveCombined(speech)
}

func (r *run) resolveCombined(speech bool) {
	cfg := r.m.cfg
	done := make(chan svcResult, 1)
	r.svcDone = done
	go func() {
		services, err := r.resolveServices(speech)
		var terms []store.Term
		if err == nil && r.settings.Dictionary.Enabled && cfg.Glossary != nil {
			terms, _ = cfg.Glossary(r.ctx)
		}
		var snippets []snippet.Snippet
		if err == nil && cfg.Snippets != nil {
			snippets, _ = cfg.Snippets(r.ctx)
		}
		if r.ctx.Err() == nil {
			select {
			case done <- svcResult{svc: services, terms: terms, snippets: snippets, err: err}:
			case <-r.ctx.Done():
			}
		}
	}()
}

func (r *run) resolveIndependent() {
	cfg := r.m.cfg
	r.init = runInitialization{
		split: true, asrDone: make(chan ASRReadiness, 1), llmDone: make(chan LLMReadiness, 1),
		termsDone: make(chan []store.Term, 1), snippetsDone: make(chan []snippet.Snippet, 1),
	}
	asrDone := r.init.asrDone
	termsDone, snippetsDone := r.init.termsDone, r.init.snippetsDone
	select {
	case <-r.sc.done:
		r.init.sceneReady = true
	default:
	}
	r.resolveCleanupChoice()
	r.svcDone = nil
	go func() {
		result := cfg.ResolveASR(r.ctx, r.req.Kind, r.settings)
		if r.ctx.Err() == nil {
			select {
			case asrDone <- result:
			case <-r.ctx.Done():
			}
		}
	}()
	go func() {
		var terms []store.Term
		if r.settings.Dictionary.Enabled && cfg.Glossary != nil {
			terms, _ = cfg.Glossary(r.ctx)
		}
		if r.ctx.Err() == nil {
			select {
			case termsDone <- terms:
			case <-r.ctx.Done():
			}
		}
	}()
	go func() {
		var snippets []snippet.Snippet
		if cfg.Snippets != nil {
			snippets, _ = cfg.Snippets(r.ctx)
		}
		if r.ctx.Err() == nil {
			select {
			case snippetsDone <- snippets:
			case <-r.ctx.Done():
			}
		}
	}()
}

func (r *run) resolveCleanupChoice() {
	if !r.init.split || r.init.cleanupDecisionReady || !r.init.sceneReady && r.sc.cleanupPending {
		return
	}
	r.init.cleanupDecisionReady = true
	r.init.cleanupRequested = r.transcribeCleanup()
	if !r.init.cleanupRequested {
		r.init.llmOutcome = LLMNotRequested
		r.init.llmReady = true
		r.maybeCleanupReady()
		return
	}
	if r.init.llmStarted {
		return
	}
	r.init.llmStarted = true
	resultChannel := r.init.llmDone
	go func() {
		effective := r.settings
		effective.Features.Transcribe.Enhance = true
		result := r.m.cfg.ResolveLLM(r.ctx, r.req.Kind, effective)
		if r.ctx.Err() == nil {
			select {
			case resultChannel <- result:
			case <-r.ctx.Done():
			}
		}
	}()
}

func (r *run) needsSpeechHints() bool {
	return r.settings.Dictionary.Enabled && r.settings.Dictionary.ASRHints
}

func (r *run) sceneReadySignal() <-chan struct{} {
	if r.init.sceneReady {
		return nil
	}
	if !r.init.split {
		select {
		case <-r.sc.done:
			return nil
		default:
		}
	}
	return r.sc.done
}

func (r *run) markSceneReady() {
	if r.init.sceneReady {
		return
	}
	r.init.sceneReady = true
	r.resolveCleanupChoice()
	r.maybeCleanupReady()
}

func (r *run) useASRReadiness(result ASRReadiness) bool {
	r.m.mu.Lock()
	if r.ended || r.cancelled.Load() || r.ctx.Err() != nil {
		r.m.mu.Unlock()
		return false
	}
	r.svc.ASR, r.svc.ASRInfo, r.svc.VAD = result.ASR, result.ASRInfo, result.VAD
	r.resolved = true
	r.m.mu.Unlock()
	r.m.update(r, func(snapshot *Snapshot) {
		snapshot.ASR = result.ASRInfo
		snapshot.UploadLen = uploadLen(result.ASR, r.settings.Audio.UploadMB)
	})
	return true
}

func (r *run) useLLMReadiness(result LLMReadiness) bool {
	if result.Outcome == "" {
		if result.LLM == nil {
			result.Outcome = LLMUnavailable
		} else {
			result.Outcome = LLMAvailable
		}
	} else if result.Outcome == LLMAvailable && result.LLM == nil {
		result.Outcome = LLMUnavailable
	}
	r.m.mu.Lock()
	if r.ended || r.cancelled.Load() || r.ctx.Err() != nil {
		r.m.mu.Unlock()
		return false
	}
	if r.cleanupChoice != cleanupUndecided {
		r.m.mu.Unlock()
		return true
	}
	r.svc.LLM, r.svc.LLMInfo = result.LLM, result.LLMInfo
	r.svc.Fallback, r.svc.FallbackInfo = result.Fallback, result.FallbackInfo
	r.m.mu.Unlock()
	r.init.llmOutcome = result.Outcome
	r.init.llmReady = true
	r.m.update(r, func(snapshot *Snapshot) { snapshot.LLM = result.LLMInfo })
	r.maybeCleanupReady()
	return true
}

func (r *run) useTerms(terms []store.Term) {
	r.terms = terms
	if r.needsSpeechHints() {
		r.speechHints = SpeechHints(r.settings, terms)
	}
	r.init.termsReady = true
	r.maybeCleanupReady()
}

func (r *run) useSnippets(snippets []snippet.Snippet) {
	r.snippets = snippets
	r.init.snippetsReady = true
	r.maybeCleanupReady()
}

func (r *run) maybeCleanupReady() {
	if !r.init.split || r.init.cleanupReady || !r.init.cleanupDecisionReady || r.init.cleanupRequested && (!r.init.llmReady || !r.init.sceneReady) || !r.init.termsReady || !r.init.snippetsReady || r.cancelled.Load() || r.ctx.Err() != nil {
		return
	}
	r.init.cleanupReady = true
	r.startHead()
	r.enableEarlyPieces()
	close(r.svcResolved)
}

func (r *run) awaitTranscribeData() bool {
	sceneDone := r.sceneReadySignal()
	for !r.init.termsReady || !r.init.snippetsReady {
		select {
		case terms := <-r.init.termsDone:
			r.init.termsDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				r.endCancelled(nil)
				return false
			}
			r.useTerms(terms)
		case snippets := <-r.init.snippetsDone:
			r.init.snippetsDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				r.endCancelled(nil)
				return false
			}
			r.useSnippets(snippets)
		case <-sceneDone:
			sceneDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				r.endCancelled(nil)
				return false
			}
			r.markSceneReady()
		case <-r.ctx.Done():
			r.endCancelled(nil)
			return false
		}
	}
	r.maybeCleanupReady()
	return true
}

func (r *run) awaitCleanupReadiness() bool {
	if !r.init.split {
		return true
	}
	r.maybeCleanupReady()
	if r.init.cleanupReady {
		return true
	}
	policy := r.m.cfg.CleanupLatency
	softDeadline := policy.AutoFallbackAfter > 0
	var deadlineAt time.Time
	var deadlineActive bool
	var stopTimer func() bool
	activateDeadline := func() {
		if !softDeadline || deadlineActive || !r.init.cleanupDecisionReady || !r.init.cleanupRequested {
			return
		}
		stopAt := r.stoppedAt()
		if stopAt.IsZero() {
			stopAt = r.recordStopAt(r.now())
		}
		deadlineAt = stopAt.Add(policy.AutoFallbackAfter)
		r.m.setCleanupDeadline(r, deadlineAt)
		deadlineActive = true
		if r.now().Before(deadlineAt) {
			stopTimer = r.m.cfg.AfterFunc(deadlineAt.Sub(r.now()), func() { r.m.notifyCleanupDeadline(r) })
		} else {
			r.m.notifyCleanupDeadline(r)
		}
	}
	defer func() {
		if stopTimer != nil {
			stopTimer()
		}
	}()
	sceneDone := r.sceneReadySignal()
	r.resolveCleanupChoice()
	r.maybeCleanupReady()
	activateDeadline()
	for !r.init.cleanupReady {
		select {
		case result := <-r.init.llmDone:
			r.init.llmDone = nil
			if !r.useLLMReadiness(result) {
				r.endCancelled(nil)
				return false
			}
		case <-sceneDone:
			sceneDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				r.endCancelled(nil)
				return false
			}
			r.markSceneReady()
			activateDeadline()
		case <-r.cleanupWake:
			if deadlineActive {
				choice, _ := r.m.chooseCleanup(r)
				if choice == cleanupRaw {
					return false
				}
				if choice == cleanupCancelled {
					r.endCancelled(nil)
					return false
				}
			}
		case <-r.ctx.Done():
			r.endCancelled(nil)
			return false
		}
		if deadlineActive && !r.now().Before(deadlineAt) {
			r.m.notifyCleanupDeadline(r)
			choice, _ := r.m.chooseCleanup(r)
			if choice == cleanupRaw {
				return false
			}
			if choice == cleanupCancelled {
				r.endCancelled(nil)
				return false
			}
		}
	}
	return true
}

// awaitServices waits for the services; false when the session ended
// (cancelled, or the services failed).
func (r *run) awaitServices() bool {
	if r.resolved {
		return true
	}
	select {
	case s := <-r.svcDone:
		if s.err != nil {
			r.fail(ProblemOf(s.err))
			return false
		}
		r.useServices(s)
		return true
	case <-r.ctx.Done():
		r.endCancelled(nil)
		return false
	}
}

// useServices keeps the resolved services.
func (r *run) useServices(s svcResult) {
	r.resolved = true
	r.svc.ASR, r.svc.ASRInfo, r.svc.VAD = s.svc.ASR, s.svc.ASRInfo, s.svc.VAD
	r.svc.LLM, r.svc.LLMInfo = s.svc.LLM, s.svc.LLMInfo
	r.svc.Fallback, r.svc.FallbackInfo = s.svc.Fallback, s.svc.FallbackInfo
	r.terms, r.snippets = s.terms, s.snippets
	if r.needsSpeechHints() {
		r.speechHints = SpeechHints(r.settings, s.terms)
	}
	r.startHead()
	close(r.svcResolved)
	up := uploadLen(s.svc.ASR, r.settings.Audio.UploadMB)
	r.m.update(r, func(sn *Snapshot) { sn.ASR, sn.LLM, sn.UploadLen = s.svc.ASRInfo, s.svc.LLMInfo, up })
}

// uploadLen is how much recording fits in one upload to p, under the
// user's limit of userMB when it is the smaller (0: p splits no uploads).
func uploadLen(p asr.Provider, userMB int) time.Duration {
	if p == nil {
		return 0
	}
	most := p.Caps().MaxBytes
	if u := int64(userMB) << 20; u > 0 && u < most {
		most = u
	}
	if most <= 44 {
		return 0
	}
	return time.Duration(most-44) * time.Second / (2 * asr.SampleRate) // 44: the WAV header
}

// lacks reports what the services lack for the session: a speech
// model, or the language model Translate and Rewrite need.
func (r *run) lacks() *Problem {
	if r.svc.ASR == nil {
		return ProblemOf(fmt.Errorf("%s: %w", r.req.Kind, ErrNoASR))
	}
	if r.svc.LLM == nil && r.req.Kind != store.KindTranscribe {
		return &Problem{Code: ProblemNoLLM, Err: errors.New("no language model is ready")}
	}
	return nil
}
