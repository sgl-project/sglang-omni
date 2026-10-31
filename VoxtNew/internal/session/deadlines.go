package session

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
)

// llmPrefillPerChar is the first token's extra time per character of the
// input: a long selection's prompt takes seconds to read under memory
// pressure (about 100 tokens a second on mac-air, docs/eval.md).
const llmPrefillPerChar = 5 * time.Millisecond

// llmDeadlines is a request over an input of chars characters, its
// deadlines set (Config.LLMFirstToken…; behavior.md 5.5); its whole time
// is at least floor.
func (cfg *Config) llmDeadlines(req llm.Request, chars int, floor time.Duration) llm.Request {
	req.FirstTokenTimeout = cfg.LLMFirstToken + time.Duration(chars)*llmPrefillPerChar
	req.StallTimeout = cfg.LLMStall
	req.Timeout = max(floor, req.FirstTokenTimeout+time.Duration(chars)*cfg.LLMPerChar)
	return req
}

// llmChat runs req within its deadlines; an error says which one passed.
func (r *run) llmChat(req llm.Request) (llm.Response, error) {
	ctx, cancel := context.WithTimeout(r.ctx, req.Timeout)
	defer cancel()
	resp, err := r.svc.LLM.Chat(ctx, req)
	if err == nil || r.ctx.Err() != nil {
		return resp, err
	}
	var e *llm.Error
	switch {
	case errors.As(err, &e) && e.Params["wait"] == llm.WaitFirstToken:
		err = fmt.Errorf("no first token within %v: %w", req.FirstTokenTimeout.Round(time.Second/10), err)
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
