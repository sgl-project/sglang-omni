package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenAI is a remote language model over the OpenAI-compatible chat
// completions API (POST {base}/chat/completions), Auralink's transport:
// the services of presets.json and any server the user names.
type OpenAI struct {
	preset Preset
	key    string
	client *http.Client
}

// NewOpenAI returns a backend for p (a resolved preset) with its API key
// ("" for none: a keyless server). client nil means http.DefaultClient.
func NewOpenAI(p Preset, key string, client *http.Client) *OpenAI {
	if client == nil {
		client = http.DefaultClient
	}
	return &OpenAI{preset: p, key: strings.TrimSpace(key), client: client}
}

func (o *OpenAI) Info() BackendInfo { return BackendInfo{ID: o.preset.ID} }

type oaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type oaUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (u *oaUsage) usage() Usage {
	if u == nil {
		return Usage{}
	}
	t := u.TotalTokens
	if t == 0 {
		t = u.PromptTokens + u.CompletionTokens
	}
	return Usage{PromptTokens: u.PromptTokens, CompletionTokens: u.CompletionTokens, TotalTokens: t}
}

func (o *OpenAI) body(req Request, model string, stream, streamUsage, extra bool) map[string]any {
	msgs := req.Messages
	if req.Format == FormatJSON {
		msgs = withJSONInstruction(msgs, req.Schema, o.preset.JSONMode != JSONSchemaMode || req.Schema == nil)
	}
	oms := make([]oaMessage, len(msgs))
	for i, m := range msgs {
		oms[i] = oaMessage{Role: string(m.Role), Content: m.Content}
	}
	b := map[string]any{}
	// A thinking mode the user chose replaces the preset's (ExtraBody).
	if extra && o.preset.Generation.Thinking.Mode == ThinkDefault {
		for k, v := range o.preset.ExtraBody {
			b[k] = v
		}
	}
	b["model"], b["messages"] = model, oms
	if req.Temperature != nil {
		b["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		b["top_p"] = *req.TopP
	}
	if req.MaxTokens > 0 {
		f := o.preset.MaxTokensField
		if f == "" {
			f = "max_tokens"
		}
		b[f] = req.MaxTokens
	}
	if len(req.Stop) > 0 {
		b["stop"] = req.Stop
	}
	if req.Seed != nil {
		b["seed"] = *req.Seed
	}
	if req.Format == FormatJSON {
		switch {
		case o.preset.JSONMode == JSONSchemaMode && req.Schema != nil:
			name := req.Schema.Name
			if name == "" {
				name = "result"
			}
			b["response_format"] = map[string]any{
				"type":        "json_schema",
				"json_schema": map[string]any{"name": name, "schema": req.Schema, "strict": false},
			}
		case o.preset.JSONMode == JSONSchemaMode, o.preset.JSONMode == JSONObjectMode:
			b["response_format"] = map[string]any{"type": "json_object"}
		}
	}
	if stream {
		b["stream"] = true
		if streamUsage {
			b["stream_options"] = map[string]any{"include_usage": true}
		}
	}
	applyGeneration(b, o.preset, model, stream)
	return b
}

// Chat makes one attempt. It streams when the request does (OnDelta) or
// bounds its progress (FirstTokenTimeout, StallTimeout).
func (o *OpenAI) Chat(ctx context.Context, req Request) (Response, error) {
	out, err := o.chat(ctx, req)
	return out, o.scrub(err)
}

// scrub blanks the key out of err's details: a server may quote it in its
// error message, and session errors are kept in the history.
func (o *OpenAI) scrub(err error) error {
	var e *Error
	if len(o.key) < 4 || !errors.As(err, &e) {
		return err
	}
	for k, v := range e.Params {
		e.Params[k] = strings.ReplaceAll(v, o.key, "[key]")
	}
	return err
}

func (o *OpenAI) chat(ctx context.Context, req Request) (Response, error) {
	model := req.Model
	if model == "" {
		model = o.preset.DefaultModel
	}
	if model == "" {
		return Response{}, newErr(CodeConfig, "provider", o.preset.ID, "field", "model")
	}
	if o.key == "" && !o.preset.KeyOptional {
		return Response{}, newErr(CodeNoKey, "provider", o.preset.ID)
	}
	if ctx.Err() != nil {
		return Response{}, ctxErr(ctx, o.preset.ID, ctx.Err(), req.Timeout)
	}
	start := time.Now()
	stream := req.OnDelta != nil
	streamUsage, extra := true, len(o.preset.ExtraBody) > 0
	url := o.preset.BaseURL + "/chat/completions"
	resp, err := o.send(ctx, url, o.body(req, model, stream, streamUsage, extra), model, req.Timeout)
	// Relax parameters a server rejects, once each: stream_options (older
	// compatible servers; usage then gets estimated), temperature / top_p
	// (reasoning models that only allow the default) and the preset's
	// ExtraBody (a model that cannot turn thinking off).
relax:
	for range 3 {
		var e *Error
		if !errors.As(err, &e) || e.Code != CodeBadRequest {
			break
		}
		detail := strings.ToLower(e.Params["detail"])
		switch {
		case stream && streamUsage && strings.Contains(detail, "stream_options"):
			streamUsage = false
		case (req.Temperature != nil || req.TopP != nil) && (strings.Contains(detail, "temperature") || strings.Contains(detail, "top_p")):
			req.Temperature, req.TopP = nil, nil
		case extra && o.mentionsExtra(detail):
			extra = false
		default:
			break relax
		}
		resp, err = o.send(ctx, url, o.body(req, model, stream, streamUsage, extra), model, req.Timeout)
	}
	if err != nil {
		return Response{}, err
	}
	defer resp.Body.Close()
	var out Response
	if stream {
		out, err = o.readStream(ctx, resp.Body, req.OnDelta, req.Timeout)
	} else {
		out, err = o.readOne(ctx, resp.Body, req.Timeout)
	}
	if err != nil {
		return Response{}, err
	}
	if out.Model == "" {
		out.Model = model
	}
	out.Provider = o.preset.ID
	out.Latency = time.Since(start)
	if out.FinishReason == FinishFilter && out.Text == "" {
		return out, newErr(CodeFiltered, "provider", o.preset.ID)
	}
	return out, nil
}

// mentionsExtra reports whether a rejection names one of ExtraBody's keys
// (or thinking in general).
func (o *OpenAI) mentionsExtra(detail string) bool {
	if strings.Contains(detail, "thinking") || strings.Contains(detail, "reasoning") {
		return true
	}
	for k := range o.preset.ExtraBody {
		if strings.Contains(detail, strings.ToLower(k)) {
			return true
		}
	}
	return false
}

type oaChoice struct {
	Message *struct {
		Content *string `json:"content"`
		Refusal *string `json:"refusal"`
	} `json:"message"`
	Delta *struct {
		Content *string `json:"content"`
	} `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

type oaResponse struct {
	Model   string          `json:"model"`
	Choices []oaChoice      `json:"choices"`
	Usage   *oaUsage        `json:"usage"`
	Error   json.RawMessage `json:"error"`
}

func (o *OpenAI) readOne(ctx context.Context, body io.Reader, timeout time.Duration) (Response, error) {
	raw, err := io.ReadAll(io.LimitReader(body, maxBody))
	if err != nil {
		return Response{}, ctxErr(ctx, o.preset.ID, err, timeout)
	}
	var r oaResponse
	if err := json.Unmarshal(raw, &r); err != nil {
		e := newErr(CodeBadResponse, "provider", o.preset.ID, "detail", clip(string(o.hide(raw)), 200))
		e.Err = err
		return Response{}, e
	}
	if len(r.Error) > 0 && string(r.Error) != "null" {
		// Some servers answer 200 with an error body.
		return Response{}, HTTPError(o.preset.ID, r.Model, 500, errorDetail(o.hide(raw)), 0)
	}
	if len(r.Choices) == 0 || r.Choices[0].Message == nil {
		return Response{}, newErr(CodeBadResponse, "provider", o.preset.ID, "detail", "no choices")
	}
	c := r.Choices[0]
	out := Response{Model: r.Model, Usage: r.Usage.usage()}
	if c.Message.Content != nil {
		out.Text = *c.Message.Content
	}
	if c.FinishReason != nil {
		out.FinishReason = normFinish(*c.FinishReason)
	}
	if out.Text == "" && c.Message.Refusal != nil && *c.Message.Refusal != "" {
		out.FinishReason = FinishFilter
	}
	return out, nil
}

func (o *OpenAI) readStream(ctx context.Context, body io.Reader, onDelta func(string), timeout time.Duration) (Response, error) {
	var out Response
	var text strings.Builder
	var perr error
	done := false
	err := readSSE(body, func(data string) bool {
		if data == "[DONE]" {
			done = true
			return false
		}
		var r oaResponse
		if err := json.Unmarshal([]byte(data), &r); err != nil {
			e := newErr(CodeBadResponse, "provider", o.preset.ID, "detail", clip(string(o.hide([]byte(data))), 200))
			e.Err = err
			perr = e
			return false
		}
		if len(r.Error) > 0 && string(r.Error) != "null" {
			perr = HTTPError(o.preset.ID, r.Model, 500, errorDetail(o.hide([]byte(data))), 0)
			return false
		}
		if r.Model != "" {
			out.Model = r.Model
		}
		if r.Usage != nil {
			out.Usage = r.Usage.usage()
		}
		for _, c := range r.Choices {
			if c.Delta != nil && c.Delta.Content != nil && *c.Delta.Content != "" {
				text.WriteString(*c.Delta.Content)
				onDelta(*c.Delta.Content)
			}
			if c.FinishReason != nil && *c.FinishReason != "" {
				out.FinishReason = normFinish(*c.FinishReason)
			}
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
	if !done && out.FinishReason == "" {
		// The stream ended without [DONE] or a finish reason: truncated.
		e := newErr(CodeNetwork, "provider", o.preset.ID, "detail", "stream ended early")
		e.Retryable = true
		return Response{}, e
	}
	out.Text = text.String()
	return out, nil
}
