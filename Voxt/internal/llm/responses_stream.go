package llm

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"time"
)

type rsUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type rsError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// rsResponse is a Responses API response: the non-streamed body and the
// response in the events that end a stream.
type rsResponse struct {
	Model  string `json:"model"`
	Output []struct {
		Content []struct {
			Type    string `json:"type"`
			Text    string `json:"text"`
			Refusal string `json:"refusal"`
		} `json:"content"`
	} `json:"output"`
	Usage             *rsUsage `json:"usage"`
	Error             *rsError `json:"error"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
}

// rsEvent is a stream event; the error event has its code and message at
// the top level.
type rsEvent struct {
	Type     string     `json:"type"`
	Delta    string     `json:"delta"`
	Response rsResponse `json:"response"`
	rsError
}

// result is r as a Response. An incomplete response is an answer cut
// short, as finish_reason "length" is in chat completions, not a failure.
func (r *rsResponse) result() Response {
	out := Response{Model: r.Model, FinishReason: FinishStop}
	if u := r.Usage; u != nil {
		out.Usage = (&oaUsage{PromptTokens: u.InputTokens, CompletionTokens: u.OutputTokens, TotalTokens: u.TotalTokens}).usage()
	}
	var text strings.Builder
	refused := false
	for _, item := range r.Output {
		for _, c := range item.Content {
			switch c.Type {
			case "output_text":
				text.WriteString(c.Text)
			case "refusal":
				refused = refused || c.Refusal != ""
			}
		}
	}
	out.Text = text.String()
	if d := r.IncompleteDetails; d != nil {
		switch d.Reason {
		case "max_output_tokens":
			out.FinishReason = FinishLength
		case "content_filter":
			out.FinishReason = FinishFilter
		default:
			out.FinishReason = FinishOther
		}
	}
	if out.Text == "" && refused {
		out.FinishReason = FinishFilter
	}
	return out
}

// failure is a failure the server reported after answering 200, as the
// status it would have had.
func (o *OpenAI) failure(model string, e rsError) *Error {
	status := 400
	switch e.Code {
	case "", "server_error":
		status = 500
	case "rate_limit_exceeded", "insufficient_quota":
		status = 429
	}
	detail := e.Message
	if e.Code != "" {
		detail = e.Code + ": " + e.Message
	}
	return HTTPError(o.preset.ID, model, status, string(o.hide([]byte(detail))), 0)
}

func (o *OpenAI) badResponse(raw []byte, err error) *Error {
	e := newErr(CodeBadResponse, "provider", o.preset.ID, "detail", clip(string(o.hide(raw)), 200))
	e.Err = err
	return e
}

func (o *OpenAI) readResponse(ctx context.Context, body io.Reader, timeout time.Duration) (Response, error) {
	raw, err := io.ReadAll(io.LimitReader(body, maxBody))
	if err != nil {
		return Response{}, ctxErr(ctx, o.preset.ID, err, timeout)
	}
	var r rsResponse
	if err := json.Unmarshal(raw, &r); err != nil {
		return Response{}, o.badResponse(raw, err)
	}
	if r.Error != nil {
		return Response{}, o.failure(r.Model, *r.Error)
	}
	if len(r.Output) == 0 && r.IncompleteDetails == nil {
		return Response{}, newErr(CodeBadResponse, "provider", o.preset.ID, "detail", "no output")
	}
	return r.result(), nil
}

// readResponseStream reads the events up to the one that ends the
// response. The stream has no [DONE]: without such an event it was cut.
func (o *OpenAI) readResponseStream(ctx context.Context, body io.Reader, onDelta func(string), timeout time.Duration) (Response, error) {
	var out Response
	var text strings.Builder
	var perr error
	done := false
	err := readSSE(body, func(data string) bool {
		var ev rsEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			perr = o.badResponse([]byte(data), err)
			return false
		}
		switch ev.Type {
		case "response.output_text.delta":
			if ev.Delta != "" {
				text.WriteString(ev.Delta)
				onDelta(ev.Delta)
			}
		case "response.completed", "response.incomplete":
			out, done = ev.Response.result(), true
			return false
		case "response.failed":
			var e rsError
			if ev.Response.Error != nil {
				e = *ev.Response.Error
			}
			perr = o.failure(ev.Response.Model, e)
			return false
		case "error":
			perr = o.failure("", ev.rsError)
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
	if text.Len() > 0 {
		out.Text = text.String()
	}
	return out, nil
}
