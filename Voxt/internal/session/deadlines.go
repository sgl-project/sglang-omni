package session

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

// llmPrefillPerChar is a request's extra time per character of the input
// before its answer starts: a long selection's prompt takes seconds to
// read.
const llmPrefillPerChar = 5 * time.Millisecond

// llmDeadlines is a request over an input of chars characters, its
// deadlines set (behavior.md 5.5): once the answer streams, no gap of
// LLMStall between its pieces; in all LLMBase plus llmPrefillPerChar and
// LLMPerChar per character, at least floor. The first piece has no
// deadline of its own: a model that reasons before it answers, or loads
// on its first request, sends nothing for a while and is not stuck.
func (cfg *Config) llmDeadlines(req llm.Request, chars int, floor time.Duration) llm.Request {
	req.StallTimeout = cfg.LLMStall
	req.Timeout = max(floor, cfg.LLMBase+time.Duration(chars)*(llmPrefillPerChar+cfg.LLMPerChar))
	return req
}

// llmFloor is a request's least deadline: the wait the user chose
// (settings.Models.LLMWait, #3821), else auto.
func (r *run) llmFloor(auto time.Duration) time.Duration {
	return cmp.Or(r.settings.Models.LLMWaitFor(), auto)
}

// TimedOut says err is a language model's deadline passing: llmChat's,
// or the client's own.
func TimedOut(err error) bool {
	return errors.Is(err, context.DeadlineExceeded) || llm.Code(err) == llm.CodeTimeout
}

// llmChat runs req within its deadlines, until SkipLLM; an error says
// which one passed.
func (r *run) llmChat(req llm.Request) (llm.Response, error) {
	ctx, cancel := context.WithTimeout(r.llmCtx, req.Timeout)
	defer cancel()
	resp, err := r.svc.LLM.Chat(ctx, req)
	if err == nil || r.ctx.Err() != nil {
		return resp, err
	}
	var e *llm.Error
	switch {
	case errors.As(err, &e) && e.Params["wait"] == llm.WaitStall:
		err = fmt.Errorf("no new token for %v after %s pieces: %w", req.StallTimeout, e.Params["deltas"], err)
	case ctx.Err() == context.DeadlineExceeded:
		err = fmt.Errorf("no answer within %v: %w", req.Timeout.Round(time.Second/10), err)
	}
	return resp, err
}

// asrLimit is how long recognition may take after Stop for a recording
// of audio: ASRTimeout plus ASRPerSecond per second of it (behavior.md
// 5.5a).
func (cfg *Config) asrLimit(audio time.Duration) time.Duration {
	return cfg.ASRTimeout + time.Duration(audio.Seconds()*float64(cfg.ASRPerSecond))
}
