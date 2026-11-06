package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// maxBody caps a non-streamed response body.
const maxBody = 32 << 20

// UserAgent is sent with every request to a remote model.
var UserAgent = "Voxt"

// send POSTs body and returns the response when the status is 2xx, else
// an *Error from the body. The key goes in the Authorization header only.
func (o *OpenAI) send(ctx context.Context, url string, body any, model string, timeout time.Duration) (*http.Response, error) {
	b, err := json.Marshal(body)
	if err != nil {
		e := newErr(CodeBadRequest, "provider", o.preset.ID, "detail", "marshal")
		e.Err = err
		return nil, e
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		e := newErr(CodeConfig, "provider", o.preset.ID, "field", "baseURL")
		e.Err = err
		return nil, e
	}
	req.Header.Set("User-Agent", UserAgent)
	for k, v := range o.preset.Headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Content-Type", "application/json")
	if o.key != "" {
		req.Header.Set("Authorization", "Bearer "+o.key)
	}
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, ctxErr(ctx, o.preset.ID, err, timeout)
	}
	if resp.StatusCode/100 == 2 {
		return resp, nil
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	return nil, HTTPError(o.preset.ID, model, resp.StatusCode, errorDetail(raw), parseRetryAfter(resp.Header))
}

// errorDetail pulls the message out of the usual error bodies:
// {"error":{"message":…,"code":…}}, {"error":"…"}, {"message":…}, {"detail":…}.
func errorDetail(raw []byte) string {
	var v struct {
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
		Detail  any             `json:"detail"`
	}
	if json.Unmarshal(raw, &v) != nil {
		return strings.TrimSpace(string(raw))
	}
	if len(v.Error) > 0 {
		var s string
		if json.Unmarshal(v.Error, &s) == nil {
			return s
		}
		var o struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		}
		if json.Unmarshal(v.Error, &o) == nil {
			parts := []string{}
			for _, p := range []any{o.Type, o.Code, o.Message} {
				if s, ok := p.(string); ok && s != "" {
					parts = append(parts, s)
				}
			}
			if len(parts) > 0 {
				return strings.Join(parts, ": ")
			}
		}
	}
	if v.Message != "" {
		return v.Message
	}
	if v.Detail != nil {
		b, _ := json.Marshal(v.Detail)
		return string(b)
	}
	return strings.TrimSpace(string(raw))
}

// parseRetryAfter reads Retry-After (seconds or HTTP date) and the
// OpenAI-style retry-after-ms.
func parseRetryAfter(h http.Header) time.Duration {
	if ms := h.Get("retry-after-ms"); ms != "" {
		if f, err := strconv.ParseFloat(ms, 64); err == nil && f > 0 {
			return time.Duration(f * float64(time.Millisecond))
		}
	}
	v := h.Get("Retry-After")
	if v == "" {
		return 0
	}
	if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
		return time.Duration(f * float64(time.Second))
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

// withJSONInstruction returns msgs with an instruction to answer in JSON
// (and the schema, when withSchema) appended to the system message (one
// is added when there is none), for servers without a native JSON mode.
func withJSONInstruction(msgs []Message, s *Schema, withSchema bool) []Message {
	var b strings.Builder
	b.WriteString("\n\nReply with a single JSON value only: no prose, no Markdown fences.")
	if withSchema && s != nil {
		js, _ := json.Marshal(s)
		b.WriteString(" It must match this JSON Schema:\n")
		b.Write(js)
	}
	instr := b.String()
	out := make([]Message, 0, len(msgs)+1)
	added := false
	for _, m := range msgs {
		if m.Role == System && !added {
			m.Content += instr
			added = true
		}
		out = append(out, m)
	}
	if !added {
		out = append([]Message{{Role: System, Content: strings.TrimSpace(instr)}}, out...)
	}
	return out
}

func normFinish(r string) string {
	switch r {
	case "stop", "end_turn", "stop_sequence", "eos":
		return FinishStop
	case "length", "max_tokens":
		return FinishLength
	case "content_filter", "refusal", "safety":
		return FinishFilter
	case "":
		return ""
	}
	return FinishOther
}

// readSSE calls fn with the data of every Server-Sent Event in r until
// EOF, an error, or fn returning false. Lines are capped at 4 MiB.
func readSSE(r io.Reader, fn func(data string) bool) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 4<<20)
	var data strings.Builder
	hasData := false
	flush := func() bool {
		if !hasData {
			return true
		}
		ok := fn(data.String())
		hasData = false
		data.Reset()
		return ok
	}
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			if !flush() {
				return nil
			}
			continue
		}
		field, value, _ := strings.Cut(line, ":")
		if field != "data" {
			continue // comments (keep-alives such as ": OPENROUTER PROCESSING"), event names, ids
		}
		if hasData {
			data.WriteByte('\n')
		}
		data.WriteString(strings.TrimPrefix(value, " "))
		hasData = true
	}
	if err := sc.Err(); err != nil {
		return err
	}
	flush()
	return nil
}
