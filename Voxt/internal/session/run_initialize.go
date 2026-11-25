package session

import (
	"time"

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
	asrDone, llmDone := r.init.asrDone, r.init.llmDone
	termsDone, snippetsDone := r.init.termsDone, r.init.snippetsDone
	select {
	case <-r.sc.done:
		r.init.sceneReady = true
	default:
	}
	r.svcDone = nil
	r.logTiming("asr_resolve_started")
	if !r.needsSpeechHints() {
		r.logTiming("asr_hints_ready")
	}
	go func() {
		result := cfg.ResolveASR(r.ctx, r.req.Kind, r.settings)
		r.logTiming("asr_resolve_completed")
		if r.ctx.Err() == nil {
			select {
			case asrDone <- result:
			case <-r.ctx.Done():
			}
		}
	}()
	r.logTiming("llm_resolve_started")
	go func() {
		effective := r.settings
		effective.Features.Transcribe.Enhance = r.transcribeCleanup()
		result := cfg.ResolveLLM(r.ctx, r.req.Kind, effective)
		r.logTiming("llm_resolve_completed")
		if r.ctx.Err() == nil {
			select {
			case llmDone <- result:
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

func (r *run) needsSpeechHints() bool {
	return r.settings.Dictionary.Enabled && r.settings.Dictionary.ASRHints
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
	r.m.mu.Lock()
	if r.ended || r.cancelled.Load() || r.ctx.Err() != nil {
		r.m.mu.Unlock()
		return false
	}
	r.svc.LLM, r.svc.LLMInfo = result.LLM, result.LLMInfo
	r.svc.Fallback, r.svc.FallbackInfo = result.Fallback, result.FallbackInfo
	r.m.mu.Unlock()
	r.init.llmReady = true
	r.m.update(r, func(snapshot *Snapshot) { snapshot.LLM = result.LLMInfo })
	r.maybeCleanupReady()
	return true
}

func (r *run) useTerms(terms []store.Term) {
	r.terms = terms
	r.speechHints = SpeechHints(r.settings, terms)
	r.init.termsReady = true
	r.logTiming("dictionary_ready")
	if r.needsSpeechHints() {
		r.logTiming("asr_hints_ready")
	}
	r.maybeCleanupReady()
}

func (r *run) useSnippets(snippets []snippet.Snippet) {
	r.snippets = snippets
	r.init.snippetsReady = true
	r.maybeCleanupReady()
}

func (r *run) maybeCleanupReady() {
	if !r.init.split || r.init.cleanupReady || !r.init.llmReady || !r.init.termsReady || !r.init.snippetsReady || !r.init.sceneReady || r.cancelled.Load() || r.ctx.Err() != nil {
		return
	}
	r.init.cleanupReady = true
	close(r.svcResolved)
	r.logTiming("cleanup_context_ready")
}

func (r *run) awaitTranscribeData() bool {
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
	var stopTimer func() bool
	if softDeadline {
		stopAt := r.stoppedAt()
		if stopAt.IsZero() {
			stopAt = r.recordStopAt(r.now())
		}
		deadlineAt = stopAt.Add(policy.AutoFallbackAfter)
		r.m.setCleanupDeadline(r, deadlineAt)
		if r.now().Before(deadlineAt) {
			stopTimer = r.m.cfg.AfterFunc(deadlineAt.Sub(r.now()), func() { r.m.notifyCleanupDeadline(r) })
		} else {
			r.m.notifyCleanupDeadline(r)
		}
	}
	if stopTimer != nil {
		defer stopTimer()
	}
	for !r.init.cleanupReady {
		select {
		case result := <-r.init.llmDone:
			r.init.llmDone = nil
			if !r.useLLMReadiness(result) {
				r.endCancelled(nil)
				return false
			}
		case <-r.sc.done:
			if r.cancelled.Load() || r.ctx.Err() != nil {
				r.endCancelled(nil)
				return false
			}
			r.init.sceneReady = true
			r.maybeCleanupReady()
		case <-r.cleanupWake:
			if softDeadline {
				choice, _ := r.m.chooseCleanup(r)
				if choice == cleanupRaw {
					r.logTiming("cleanup_completed")
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
		if softDeadline && !r.now().Before(deadlineAt) {
			r.m.notifyCleanupDeadline(r)
			choice, _ := r.m.chooseCleanup(r)
			if choice == cleanupRaw {
				r.logTiming("cleanup_completed")
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
