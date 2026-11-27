package session

import (
	"context"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

type fileTranscriptionResult struct {
	result asr.Result
	err    error
}

func (r *run) transcribeFile(actx context.Context, provider asr.FileTranscriber, request asr.FileRequest, sceneDone *<-chan struct{}, onReadiness func()) (asr.Result, error, bool) {
	resultCh := make(chan fileTranscriptionResult, 1)
	go func() {
		result, err := provider.Transcribe(actx, request)
		resultCh <- fileTranscriptionResult{result: result, err: err}
	}()
	for {
		select {
		case result := <-r.init.llmDone:
			r.init.llmDone = nil
			if !r.useLLMReadiness(result) {
				r.endCancelled(nil)
				return asr.Result{}, nil, false
			}
			onReadiness()
		case terms := <-r.init.termsDone:
			r.init.termsDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				r.endCancelled(nil)
				return asr.Result{}, nil, false
			}
			r.useTerms(terms)
			onReadiness()
		case snippets := <-r.init.snippetsDone:
			r.init.snippetsDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				r.endCancelled(nil)
				return asr.Result{}, nil, false
			}
			r.useSnippets(snippets)
			onReadiness()
		case <-*sceneDone:
			*sceneDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				r.endCancelled(nil)
				return asr.Result{}, nil, false
			}
			r.markSceneReady()
			onReadiness()
		case result := <-resultCh:
			return result.result, result.err, true
		case <-actx.Done():
			return asr.Result{}, r.ctxErr(actx), true
		}
	}
}
