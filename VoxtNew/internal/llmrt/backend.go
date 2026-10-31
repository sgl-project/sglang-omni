package llmrt

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/wire"
)

// Provider is the backend's name in llm.Response.Provider.
const Provider = "voxtnew-llm"

// Backend is one installed model over a Runtime (llm.Backend): each
// request becomes one chat of the model's Profile. Wrap it in llm.Client
// (Open does) for retries, JSON repair and thinking stripped.
type Backend struct {
	rt      *Runtime
	profile Profile
	model   wire.Model
}

// NewBackend is the model of profile at path (an MLX model's directory,
// or a GGUF model's file) over rt.
func NewBackend(rt *Runtime, profile Profile, path string) *Backend {
	return &Backend{rt: rt, profile: profile, model: wire.Model{ID: profile.ID, Format: profile.Format, Path: path}}
}

func (b *Backend) Info() llm.BackendInfo { return llm.BackendInfo{ID: b.profile.ID} }

// Warm loads the model ahead of its first chat.
func (b *Backend) Warm(ctx context.Context) error {
	_, err := b.rt.Load(ctx, b.model)
	return err
}

// Prefill evaluates the shared part of req's prompt (its messages before
// the last: compose's system prompt) into the process's prompt cache,
// without an answer, so that a chat with the same start skips it. The
// last message's text does not matter. A GGUF model has nothing to warm.
func (b *Backend) Prefill(ctx context.Context, req llm.Request) error {
	if b.profile.Format != wire.FormatMLX {
		return nil
	}
	params, err := b.params(req)
	if err != nil {
		return err
	}
	params.MaxTokens = 0
	raw, err := b.rt.modelCall(ctx, b.model, wire.MethodChat, params, nil)
	if err != nil {
		return b.chatErr(ctx, err)
	}
	var out wire.ChatResult
	if err := json.Unmarshal(raw, &out); err != nil {
		return &llm.Error{Code: llm.CodeBadResponse, Params: map[string]string{"provider": b.profile.ID, "detail": err.Error()}, Err: err}
	}
	b.rt.noteModel(&b.model, out.Memory)
	b.rt.log(core.SeverityInfo, chatLine(b.profile.ID, req.Skill+" prefill", out))
	return nil
}

// Chat runs req as one chat. Its OnDelta is called on this goroutine.
func (b *Backend) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	params, err := b.params(req)
	if err != nil {
		return llm.Response{}, err
	}
	start := time.Now()
	system, _ := readSystemMemory()
	pump := &deltaPump{signal: make(chan struct{}, 1), start: start}
	type result struct {
		raw json.RawMessage
		err error
	}
	done := make(chan result, 1)
	go func() {
		raw, err := b.rt.modelCall(ctx, b.model, wire.MethodChat, params, pump.push)
		done <- result{raw, err}
	}()
	var res result
	for waiting := true; waiting; {
		select {
		case <-pump.signal:
			pump.deliver(req.OnDelta)
		case res = <-done:
			waiting = false
		}
	}
	pump.deliver(req.OnDelta)
	if res.err != nil {
		if stopped(ctx) {
			stats := pump.stats()
			now, nowErr := readSystemMemory()
			b.rt.log(core.SeverityWarning, b.stoppedLine(req.Skill, stats, system, now, nowErr))
			b.rt.notePace(stoppedPace(b.profile.ID, req.Skill, stats), now)
		}
		return llm.Response{}, b.chatErr(ctx, res.err)
	}
	var out wire.ChatResult
	if err := json.Unmarshal(res.raw, &out); err != nil {
		return llm.Response{}, &llm.Error{Code: llm.CodeBadResponse, Params: map[string]string{"provider": b.profile.ID, "detail": err.Error()}, Err: err}
	}
	b.rt.noteModel(&b.model, out.Memory)
	line := chatLine(b.profile.ID, req.Skill, out)
	pace := resultPace(b.profile.ID, req.Skill, out)
	now, nowErr := readSystemMemory()
	if pace.Slow() {
		line += "; slow: " + systemLine(system, now, nowErr, pump.stats().longestGap)
	}
	b.rt.log(core.SeverityInfo, line)
	b.rt.notePace(pace, now)
	finish := llm.FinishStop
	if out.Finish == wire.FinishLength {
		finish = llm.FinishLength
	}
	return llm.Response{
		Text: out.Text, FinishReason: finish, Model: b.profile.Repo, Provider: Provider,
		Usage:   llm.Usage{PromptTokens: out.Usage.PromptTokens, CompletionTokens: out.Usage.CompletionTokens, TotalTokens: out.Usage.PromptTokens + out.Usage.CompletionTokens},
		Latency: time.Since(start),
	}, nil
}

// params builds the chat of req from the profile.
func (b *Backend) params(req llm.Request) (wire.ChatParams, error) {
	p := wire.ChatParams{Model: b.model, Sampling: b.profile.SamplingFor(req.Skill), Template: b.profile.Template}
	var input string
	switch b.profile.Prompt {
	case PromptHyMT:
		if req.Translate == nil || req.Skill != llm.SkillTranslateText {
			return p, &llm.Error{Code: llm.CodeUnsupported, Params: map[string]string{"provider": b.profile.ID, "feature": string(req.Skill)}}
		}
		input = req.Translate.Text
		p.Messages = []wire.ChatMessage{{Role: "user", Content: HyMTPrompt(req.Translate.Text, req.Translate.Target)}}
	default:
		for _, m := range req.Messages {
			p.Messages = append(p.Messages, wire.ChatMessage{Role: string(m.Role), Content: m.Content})
			if m.Role == llm.User {
				input = m.Content
			}
		}
	}
	if len(p.Messages) == 0 {
		return p, &llm.Error{Code: llm.CodeBadRequest, Params: map[string]string{"provider": b.profile.ID, "detail": "no messages"}}
	}
	chars := 0
	for _, m := range p.Messages {
		chars += utf8.RuneCountInString(m.Content)
	}
	p.MaxTokens = b.profile.MaxTokens(req.Skill, utf8.RuneCountInString(input))
	if req.MaxTokens > 0 {
		p.MaxTokens = req.MaxTokens
	}
	p.PrefillStepSize = b.profile.PrefillStepSize(chars)
	if req.Seed != nil {
		seed := uint64(*req.Seed)
		p.Sampling.Seed = &seed
	}
	return p, nil
}

// chatErr is err as llm's: a crash is retryable (llm.Client starts the
// process again), a model that does not load is not.
func (b *Backend) chatErr(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err() // llm.Client tells a timeout from a cancel
	}
	params := func(kv ...string) map[string]string {
		m := map[string]string{"provider": b.profile.ID}
		for i := 0; i+1 < len(kv); i += 2 {
			m[kv[i]] = kv[i+1]
		}
		return m
	}
	var we *wire.Error
	switch {
	case errors.As(err, &we):
		switch we.Code {
		case wire.CodeLoadFailed:
			return &llm.Error{Code: llm.CodeModelNotFound, Params: params("model", b.profile.ID), Err: err}
		case wire.CodeContextLength:
			return &llm.Error{Code: llm.CodeContextLength, Params: params("model", b.profile.ID), Err: err}
		case wire.CodeBadRequest:
			return &llm.Error{Code: llm.CodeBadRequest, Params: params("status", "0", "detail", we.Message), Err: err}
		case wire.CodeCancelled:
			return &llm.Error{Code: llm.CodeCancelled, Err: err}
		}
		return &llm.Error{Code: llm.CodeServer, Params: params("status", "0"), Retryable: true, Err: err}
	case errors.Is(err, ErrExited):
		return &llm.Error{Code: llm.CodeServer, Params: params("status", "0"), Retryable: true, Err: err}
	}
	// ErrUnavailable, ErrStopped: starting again within a session will not help.
	return &llm.Error{Code: llm.CodeServer, Params: params("status", "0"), Err: err}
}

// chatLine is the log's line of a chat: how fast, how much memory.
func chatLine(id string, skill llm.Skill, r wire.ChatResult) string {
	load := ""
	if r.Timings.LoadMs > 0 {
		load = fmt.Sprintf(", load %.0f ms", r.Timings.LoadMs)
	}
	cache := ""
	if c := r.Timings.Cache; c != nil {
		switch {
		case c.Hit:
			cache = fmt.Sprintf(", prompt cache hit (%d tokens)", c.PrefixTokens)
		case c.PrefixTokens > 0:
			cache = fmt.Sprintf(", prompt cache miss (%d tokens kept)", c.PrefixTokens)
		default:
			cache = ", prompt cache unused"
		}
	}
	return fmt.Sprintf("%s %s: %d+%d tokens, first token %.0f ms%s%s, %.1f tokens/s, %.0f ms in all, %s; footprint %d MB (peak %d MB), resident %d MB",
		id, skill, r.Usage.PromptTokens, r.Usage.CompletionTokens, r.Timings.FirstTokenMs, load, cache, r.Timings.TokensPerSecond,
		r.Timings.TotalMs, r.Finish, r.Memory.FootprintBytes>>20, r.Memory.PeakFootprintBytes>>20, r.Memory.ResidentBytes>>20)
}
