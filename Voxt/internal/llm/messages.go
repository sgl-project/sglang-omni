package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"strings"
	"time"
)

// APIMessages is Preset.API for Anthropic's Messages API
// (POST {base}/messages).
const APIMessages = "messages"

// anthropicVersion is the Messages API version the requests are written
// for (the anthropic-version header).
const anthropicVersion = "2023-06-01"

// messagesMaxTokens is max_tokens when the request sets none: the Messages
// API requires the field.
const messagesMaxTokens = 16000

func (o *OpenAI) messagesAPI() bool { return o.preset.API == APIMessages }

// messagesBody rewrites b, a finished chat completions body, for the
// Messages API, as responsesBody does for the Responses API. A key of
// extra, the user's Extra Body JSON, wins over what is written here.
// promptCache marks the system prompt for the service's prompt cache.
func messagesBody(b, extra map[string]any, promptCache bool) map[string]any {
	put := func(key string, v any) {
		if _, own := extra[key]; !own {
			b[key] = v
		}
	}
	messages, _ := b["messages"].([]oaMessage)
	var system []string
	turns := make([]oaMessage, 0, len(messages))
	for _, m := range messages {
		if m.Role == string(System) {
			system = append(system, m.Content.(string))
		} else {
			turns = append(turns, m)
		}
	}
	put("messages", turns)
	if len(system) > 0 {
		var prompt any = strings.Join(system, "\n\n")
		if promptCache {
			// The system prompt is the part a feature's requests share: what
			// is dictated or selected comes after it and is not cached.
			prompt = []map[string]any{{"type": "text", "text": prompt, "cache_control": map[string]any{"type": "ephemeral"}}}
		}
		put("system", prompt)
	}
	if _, set := b["max_tokens"]; !set {
		maxTokens := messagesMaxTokens
		if thinking, _ := b["thinking"].(map[string]any); thinking != nil {
			if budget, ok := thinking["budget_tokens"].(int); ok {
				maxTokens += budget
			}
		}
		b["max_tokens"] = maxTokens
	}
	if v, ok := b["stop"]; ok {
		delete(b, "stop")
		put("stop_sequences", v)
	}
	// Not fields of the Messages API (usage comes in message_start and
	// message_delta).
	for _, key := range []string{"stream_options", "seed", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty", "logprobs", "top_logprobs", "keep_alive", "options", "reasoning_effort", "verbosity", "response_format"} {
		if _, own := extra[key]; !own {
			delete(b, key)
		}
	}
	return b
}

// messagesImage is a JPEG as a Messages API content block.
func messagesImage(jpeg []byte) map[string]any {
	return map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/jpeg", "data": base64.StdEncoding.EncodeToString(jpeg)}}
}

// msUsage is a Messages API usage: input_tokens counts only the prompt
// tokens neither read from nor written to the cache.
type msUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

func (u msUsage) usage() Usage {
	prompt := u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
	return Usage{PromptTokens: prompt, CompletionTokens: u.OutputTokens, TotalTokens: prompt + u.OutputTokens, CachedTokens: u.CacheReadInputTokens}
}

type msError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// msMessage is a Messages API message: the non-streamed body and the
// message of message_start.
type msMessage struct {
	Model   string `json:"model"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	StopReason *string  `json:"stop_reason"`
	Usage      *msUsage `json:"usage"`
}

// msEvent is a stream event.
type msEvent struct {
	Type    string    `json:"type"`
	Message msMessage `json:"message"`
	Delta   struct {
		Type       string `json:"type"`
		Text       string `json:"text"`
		StopReason string `json:"stop_reason"`
	} `json:"delta"`
	Usage *msUsage `json:"usage"`
	Error *msError `json:"error"`
}

// messagesFailure is an error the server reported in a body or a stream
// event, as the status its type has.
func (o *OpenAI) messagesFailure(model string, e msError) *Error {
	status := 500
	switch e.Type {
	case "invalid_request_error":
		status = 400
	case "authentication_error":
		status = 401
	case "billing_error":
		status = 402
	case "permission_error":
		status = 403
	case "not_found_error":
		status = 404
	case "request_too_large":
		status = 413
	case "rate_limit_error":
		status = 429
	case "overloaded_error":
		status = 529
	}
	detail := e.Message
	if e.Type != "" {
		detail = e.Type + ": " + e.Message
	}
	return HTTPError(o.preset.ID, model, status, string(o.hide([]byte(detail))), 0)
}

func (o *OpenAI) readMessage(ctx context.Context, body io.Reader, timeout time.Duration) (Response, error) {
	raw, err := io.ReadAll(io.LimitReader(body, maxBody))
	if err != nil {
		return Response{}, ctxErr(ctx, o.preset.ID, err, timeout)
	}
	var r struct {
		Type  string   `json:"type"`
		Error *msError `json:"error"`
		msMessage
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return Response{}, o.badResponse(raw, err)
	}
	if r.Error != nil {
		return Response{}, o.messagesFailure(r.Model, *r.Error)
	}
	if r.StopReason == nil {
		return Response{}, newErr(CodeBadResponse, "provider", o.preset.ID, "detail", "no stop_reason")
	}
	var text strings.Builder
	for _, block := range r.Content {
		if block.Type == "text" { // not thinking blocks
			text.WriteString(block.Text)
		}
	}
	out := Response{Model: r.Model, Text: text.String(), FinishReason: normFinish(*r.StopReason)}
	if r.Usage != nil {
		out.Usage = r.Usage.usage()
	}
	return out, nil
}

// readMessagesStream reads the events up to message_stop; without it the
// stream was cut. Only text deltas are the answer's.
func (o *OpenAI) readMessagesStream(ctx context.Context, body io.Reader, onDelta func(string), timeout time.Duration) (Response, error) {
	var out Response
	var usage msUsage
	var text strings.Builder
	var perr error
	done := false
	err := readSSE(body, func(data string) bool {
		var ev msEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			perr = o.badResponse([]byte(data), err)
			return false
		}
		switch ev.Type {
		case "message_start":
			out.Model = ev.Message.Model
			if ev.Message.Usage != nil {
				usage = *ev.Message.Usage
			}
		case "content_block_delta":
			if ev.Delta.Type == "text_delta" && ev.Delta.Text != "" {
				text.WriteString(ev.Delta.Text)
				onDelta(ev.Delta.Text)
			}
		case "message_delta":
			if ev.Delta.StopReason != "" {
				out.FinishReason = normFinish(ev.Delta.StopReason)
			}
			if u := ev.Usage; u != nil { // the counts so far
				usage.InputTokens = max(usage.InputTokens, u.InputTokens)
				usage.OutputTokens = max(usage.OutputTokens, u.OutputTokens)
				usage.CacheCreationInputTokens = max(usage.CacheCreationInputTokens, u.CacheCreationInputTokens)
				usage.CacheReadInputTokens = max(usage.CacheReadInputTokens, u.CacheReadInputTokens)
			}
		case "message_stop":
			done = true
			return false
		case "error":
			var e msError
			if ev.Error != nil {
				e = *ev.Error
			}
			perr = o.messagesFailure(out.Model, e)
			return false
		}
		return true
	})
	if perr != nil {
		return Response{}, perr
	}
	if err != nil {
		return Response{}, ctxErr(ctx, o.preset.ID, err, timeout)
	}
	if ctx.Err() != nil {
		return Response{}, ctxErr(ctx, o.preset.ID, ctx.Err(), timeout)
	}
	if !done {
		e := newErr(CodeNetwork, "provider", o.preset.ID, "detail", "stream ended early")
		e.Retryable = true
		return Response{}, e
	}
	out.Text, out.Usage = text.String(), usage.usage()
	return out, nil
}
