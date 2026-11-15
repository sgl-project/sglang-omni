package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// OpenAI is a remote language model over the OpenAI-compatible chat
// completions API (POST {base}/chat/completions), Auralink's transport:
// the services of presets.json and any server the user names. Preset.API
// switches it to OpenAI's Responses API (responses.go).
type OpenAI struct {
	preset Preset
	key    string
	client *http.Client
	tokens TokenSource

	mu   sync.Mutex
	seen []string // the tokens used so far, to blank them out of errors
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
	Content any    `json:"content"` // a string, or parts with an image
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
	last := -1
	for i, m := range msgs {
		oms[i] = oaMessage{Role: string(m.Role), Content: m.Content}
		if m.Role == User {
			last = i
		}
	}
	if req.Image != nil && last >= 0 {
		oms[last].Content = o.withImage(msgs[last].Content, req.Image)
	}
	p := o.preset
	switch t := req.Thinking; {
	case t != nil:
		p.Generation.Thinking.Mode, p.Generation.Thinking.Effort, p.Generation.Thinking.Budget = t.Mode, t.Effort, t.Budget
	case req.InputChars > 0 && req.InputChars < p.Generation.ThinkOffUnder:
		p.Generation.Thinking.Mode = ThinkDefault
	}
	b := map[string]any{}
	// A thinking mode the user chose replaces the preset's (ExtraBody).
	if extra && p.Generation.Thinking.Mode == ThinkDefault {
		for k, v := range p.ExtraBody {
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
	applyGeneration(b, p, model, stream)
	if o.responsesAPI() {
		b = responsesBody(b, o.preset.Generation.ExtraBody)
	}
	if o.preset.SignIn {
		return bareBody(b)
	}
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
	if !errors.As(err, &e) {
		return err
	}
	for k, v := range e.Params {
		for _, secret := range o.secrets() {
			v = strings.ReplaceAll(v, secret, "[key]")
		}
		e.Params[k] = v
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
	switch {
	case o.preset.SignIn && o.tokens == nil:
		return Response{}, newErr(CodeSignIn, "provider", o.preset.ID)
	case !o.preset.SignIn && o.key == "" && !o.preset.KeyOptional:
		return Response{}, newErr(CodeNoKey, "provider", o.preset.ID)
	}
	if ctx.Err() != nil {
		return Response{}, ctxErr(ctx, o.preset.ID, ctx.Err(), req.Timeout)
	}
	start := time.Now()
	stream := req.OnDelta != nil
	if o.preset.SignIn && !stream { // that service only streams
		stream, req.OnDelta = true, func(string) {}
	}
	streamUsage, extra, sampling := true, len(o.preset.ExtraBody) > 0, true
	url := o.preset.BaseURL + o.path()
	build := func() map[string]any {
		b := o.body(req, model, stream, streamUsage, extra)
		if !sampling { // after applyGeneration: its values and Extra Body win over the request's
			delete(b, "temperature")
			delete(b, "top_p")
		}
		return b
	}
	sent := build()
	resp, err := o.send(ctx, url, sent, model, req.Timeout)
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
		case (sent["temperature"] != nil || sent["top_p"] != nil) && (strings.Contains(detail, "temperature") || strings.Contains(detail, "top_p")):
			sampling = false
		case extra && o.mentionsExtra(detail):
			extra = false
		case req.Image != nil: // a model that takes no images
			req.Image = nil
		default:
			break relax
		}
		sent = build()
		resp, err = o.send(ctx, url, sent, model, req.Timeout)
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

// withImage is text and a JPEG as one message's content parts, in the
// API's format.
func (o *OpenAI) withImage(text string, jpeg []byte) []map[string]any {
	url := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(jpeg)
	if o.responsesAPI() {
		return []map[string]any{{"type": "input_text", "text": text}, {"type": "input_image", "image_url": url}}
	}
	return []map[string]any{{"type": "text", "text": text}, {"type": "image_url", "image_url": map[string]any{"url": url}}}
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
	if o.responsesAPI() {
		return o.readResponse(ctx, body, timeout)
	}
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
	if o.responsesAPI() {
		return o.readResponseStream(ctx, body, onDelta, timeout)
	}
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
