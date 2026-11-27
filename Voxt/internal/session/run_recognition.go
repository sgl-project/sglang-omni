package session

import "context"

// note (chenye): waitFinal consumes whole-text updates while the session owner
// waits for the authoritative ASR result.
func (r *run) waitFinal(ctx context.Context, finals <-chan streamEnd, heard <-chan heardText, sceneDone *<-chan struct{}, onReadiness func()) (streamEnd, error) {
	for {
		if end, err, ready := r.recognitionReady(ctx, finals); ready {
			return end, err
		}
		select {
		case end := <-finals:
			return end, nil
		case <-ctx.Done():
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return end, err
			}
			return streamEnd{}, r.ctxErr(ctx)
		case update := <-heard:
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return end, err
			}
			plan, ok := r.head.prepareLate(update.text)
			if !ok {
				continue
			}
			if end, err, ready := r.dispatchLate(ctx, finals, plan); ready {
				return end, err
			}
		case result := <-r.init.llmDone:
			r.init.llmDone = nil
			if !r.useLLMReadiness(result) {
				return streamEnd{}, r.ctxErr(ctx)
			}
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return end, err
			}
			onReadiness()
		case terms := <-r.init.termsDone:
			r.init.termsDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				return streamEnd{}, r.ctxErr(ctx)
			}
			r.useTerms(terms)
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return end, err
			}
			onReadiness()
		case snippets := <-r.init.snippetsDone:
			r.init.snippetsDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				return streamEnd{}, r.ctxErr(ctx)
			}
			r.useSnippets(snippets)
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return end, err
			}
			onReadiness()
		case <-*sceneDone:
			*sceneDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				return streamEnd{}, r.ctxErr(ctx)
			}
			r.markSceneReady()
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return end, err
			}
			onReadiness()
		}
	}
}

// note (chenye): dispatchLate returns an observable terminal result before it
// starts prepared speculative requests.
func (r *run) dispatchLate(ctx context.Context, finals <-chan streamEnd, plan headPlan) (streamEnd, error, bool) {
	if end, err, ready := r.recognitionReady(ctx, finals); ready {
		return end, err, true
	}
	r.head.submitLate(plan)
	return streamEnd{}, nil, false
}

func (r *run) recognitionReady(ctx context.Context, finals <-chan streamEnd) (streamEnd, error, bool) {
	select {
	case end := <-finals:
		return end, nil, true
	default:
	}
	select {
	case <-ctx.Done():
		return streamEnd{}, r.ctxErr(ctx), true
	default:
		return streamEnd{}, nil, false
	}
}
