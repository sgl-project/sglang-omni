package session

import (
	"context"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

type fileTranscriptionResult struct {
	result asr.Result
	err    error
}

func (r *run) awaitStreamFinal(stream asr.Stream, finals <-chan streamEnd, actx context.Context, sceneDone *<-chan struct{}) (streamEnd, bool) {
	for {
		select {
		case result := <-finals:
			return result, true
		case <-*sceneDone:
			*sceneDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				r.endCancelled(stream)
				return streamEnd{}, false
			}
			r.markSceneReady()
		case <-actx.Done():
			return streamEnd{err: r.ctxErr(actx)}, true
		}
	}
}

func (r *run) transcribeFile(actx context.Context, provider asr.FileTranscriber, request asr.FileRequest, sceneDone *<-chan struct{}) (asr.Result, error, bool) {
	resultCh := make(chan fileTranscriptionResult, 1)
	go func() {
		result, err := provider.Transcribe(actx, request)
		resultCh <- fileTranscriptionResult{result: result, err: err}
	}()
	for {
		select {
		case <-*sceneDone:
			*sceneDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				r.endCancelled(nil)
				return asr.Result{}, nil, false
			}
			r.markSceneReady()
		case result := <-resultCh:
			return result.result, result.err, true
		case <-actx.Done():
			return asr.Result{}, r.ctxErr(actx), true
		}
	}
}
