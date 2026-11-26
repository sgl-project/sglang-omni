package session

import "github.com/sgl-project/sglang-omni/Voxt/internal/asr"

// awaitASRReadiness resolves speech and required hints after a valid capture.
// Optional cleanup readiness never blocks recognition here.
func (r *run) awaitASRReadiness(stream asr.Stream, startOpen, onReadiness func(), sceneDone *<-chan struct{}) bool {
	if r.init.split {
		for !r.resolved || r.needsSpeechHints() && !r.init.termsReady {
			select {
			case <-r.ctx.Done():
				r.endCancelled(stream)
				return false
			case result := <-r.init.asrDone:
				r.init.asrDone = nil
				if r.cancelled.Load() || r.ctx.Err() != nil {
					r.endCancelled(stream)
					return false
				}
				if result.Err != nil {
					r.fail(ProblemOf(result.Err))
					return false
				}
				if !r.useASRReadiness(result) {
					r.endCancelled(stream)
					return false
				}
				if p := r.lacks(); p != nil {
					r.fail(p)
					return false
				}
			case result := <-r.init.llmDone:
				r.init.llmDone = nil
				if !r.useLLMReadiness(result) {
					r.endCancelled(stream)
					return false
				}
				onReadiness()
			case terms := <-r.init.termsDone:
				r.init.termsDone = nil
				if r.cancelled.Load() || r.ctx.Err() != nil {
					r.endCancelled(stream)
					return false
				}
				r.useTerms(terms)
				onReadiness()
			case snippets := <-r.init.snippetsDone:
				r.init.snippetsDone = nil
				if r.cancelled.Load() || r.ctx.Err() != nil {
					r.endCancelled(stream)
					return false
				}
				r.useSnippets(snippets)
				onReadiness()
			case <-*sceneDone:
				*sceneDone = nil
				if r.cancelled.Load() || r.ctx.Err() != nil {
					r.endCancelled(stream)
					return false
				}
				r.markSceneReady()
				onReadiness()
			}
		}
	} else if !r.resolved {
		// The compatibility resolver is still shared by non-Transcribe paths.
		if !r.awaitServices() {
			return false
		}
	}
	if p := r.lacks(); p != nil {
		r.fail(p)
		return false
	}
	startOpen()
	return true
}
