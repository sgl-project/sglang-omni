package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"strings"
	"time"
)

// Defaults for Client.
const (
	DefaultTimeout     = 120 * time.Second // one attempt
	DefaultMaxAttempts = 3
	DefaultRepairs     = 2
)

// RetryPolicy is exponential backoff with full jitter, honouring the
// server's Retry-After.
type RetryPolicy struct {
	MaxAttempts   int           // total attempts, ≥1 (0: DefaultMaxAttempts)
	Base          time.Duration // first backoff (0: 500 ms)
	Max           time.Duration // backoff cap (0: 20 s)
	MaxRetryAfter time.Duration // longest Retry-After honoured; longer fails at once (0: 60 s)
}

func (p RetryPolicy) attempts() int {
	if p.MaxAttempts <= 0 {
		return DefaultMaxAttempts
	}
	return p.MaxAttempts
}

// delay is the wait before retry n (1-based) after err.
func (p RetryPolicy) delay(n int, e *Error, jitter func() float64) (time.Duration, bool) {
	base, maxD, maxRA := p.Base, p.Max, p.MaxRetryAfter
	if base <= 0 {
		base = 500 * time.Millisecond
	}
	if maxD <= 0 {
		maxD = 20 * time.Second
	}
	if maxRA <= 0 {
		maxRA = 60 * time.Second
	}
	if e.RetryAfter > 0 {
		if e.RetryAfter > maxRA {
			return 0, false
		}
		return e.RetryAfter, true
	}
	d := base << (n - 1)
	if d > maxD || d <= 0 {
		d = maxD
	}
	// Full jitter, floored at half so retries don't fire instantly.
	return d/2 + time.Duration(jitter()*float64(d/2)), true
}

// Client adds timeouts, retries, usage accounting and JSON output to a
// Backend. The zero value of every field is usable; Backend is required.
type Client struct {
	Backend Backend
	Timeout time.Duration // per attempt (0: DefaultTimeout); Request.Timeout overrides
	Retry   RetryPolicy
	Meter   *Meter // optional

	// Sleep waits between retries; tests replace it. nil: a timer honouring ctx.
	Sleep func(ctx context.Context, d time.Duration) error
}

func (c *Client) sleep(ctx context.Context, d time.Duration) error {
	if c.Sleep != nil {
		return c.Sleep(ctx, d)
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// Prefill passes req to the backend when it is a Prefiller; nil
// otherwise.
func (c *Client) Prefill(ctx context.Context, req Request) error {
	if p, ok := c.Backend.(Prefiller); ok {
		return p.Prefill(ctx, req)
	}
	return nil
}

// Chat runs req with retries. A streaming request is retried only while
// nothing has been streamed yet. The returned Usage sums every attempt.
func (c *Client) Chat(ctx context.Context, req Request) (Response, error) {
	if c.Backend == nil {
		return Response{}, newErr(CodeNoRoute, "skill", string(req.Skill))
	}
	timeout := req.Timeout
	if timeout <= 0 {
		timeout = c.Timeout
	}
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	req.Timeout = timeout
	streamed := false
	var filter *thinkFilter
	if user := req.OnDelta; user != nil {
		filter = &thinkFilter{emit: func(s string) { streamed = true; user(s) }}
		req.OnDelta = filter.write
	}
	var watch *progress
	if req.FirstTokenTimeout > 0 || req.StallTimeout > 0 {
		watch = &progress{firstToken: req.FirstTokenTimeout, stall: req.StallTimeout}
		inner := req.OnDelta
		req.OnDelta = func(s string) {
			watch.tick()
			if inner != nil {
				inner(s)
			}
		}
	}
	var total Usage
	start := time.Now()
	for attempt := 1; ; attempt++ {
		if err := ctx.Err(); err != nil {
			return Response{}, ctxErr(ctx, c.Backend.Info().ID, err, timeout)
		}
		if filter != nil {
			filter.reset()
		}
		actx, cancel := context.WithTimeout(ctx, timeout)
		if watch != nil {
			actx = watch.begin(actx)
		}
		resp, err := c.Backend.Chat(actx, req)
		var stalled *Error
		if watch != nil {
			wait, waited, deltas, elapsed := watch.end()
			if wait != "" && err != nil && ctx.Err() == nil {
				stalled = progressErr(c.Backend.Info().ID, wait, waited, deltas, elapsed, err)
			}
		}
		cancel()
		if err == nil {
			if filter != nil {
				filter.flush()
			}
			resp.Text = StripThinking(resp.Text)
			if resp.Usage.TotalTokens == 0 {
				resp.Usage = estimateUsage(req.Messages, resp.Text)
			}
			total = total.Add(resp.Usage)
			resp.Usage, resp.Attempts = total, attempt
			if resp.Text == "" && resp.FinishReason != FinishLength {
				err = newErr(CodeEmpty, "provider", c.Backend.Info().ID)
				c.Meter.record(c.Backend.Info().ID, req, resp, err, time.Since(start))
				return resp, err
			}
			c.Meter.record(c.Backend.Info().ID, req, resp, nil, time.Since(start))
			return resp, nil
		}
		var e *Error
		if !errors.As(err, &e) {
			e = ctxErr(ctx, c.Backend.Info().ID, err, timeout)
		}
		if stalled != nil {
			e = stalled
		}
		// A timeout of the attempt (not the caller's deadline) is retryable.
		if ctx.Err() != nil {
			e = ctxErr(ctx, c.Backend.Info().ID, ctx.Err(), timeout)
		}
		if !e.Retryable || streamed || attempt >= c.Retry.attempts() || ctx.Err() != nil {
			c.Meter.record(c.Backend.Info().ID, req, Response{Usage: total, Attempts: attempt}, e, time.Since(start))
			return Response{Usage: total, Attempts: attempt}, e
		}
		d, ok := c.Retry.delay(attempt, e, rand.Float64)
		if !ok {
			c.Meter.record(c.Backend.Info().ID, req, Response{Usage: total, Attempts: attempt}, e, time.Since(start))
			return Response{Usage: total, Attempts: attempt}, e
		}
		if dl, ok := ctx.Deadline(); ok && time.Until(dl) < d {
			return Response{Usage: total, Attempts: attempt}, e // no time left to wait
		}
		if err := c.sleep(ctx, d); err != nil {
			return Response{Usage: total, Attempts: attempt}, ctxErr(ctx, c.Backend.Info().ID, err, timeout)
		}
	}
}

// JSON runs req as a JSON request and decodes the reply into out (a
// pointer, or nil). When the reply holds no JSON, fails to decode, or
// breaks req.Schema, it asks the model to fix it, up to repairs times
// (<0: none; 0: DefaultRepairs). Response.JSON holds the accepted value.
func (c *Client) JSON(ctx context.Context, req Request, out any, repairs int) (Response, error) {
	if repairs == 0 {
		repairs = DefaultRepairs
	}
	if repairs < 0 {
		repairs = 0
	}
	req.Format = FormatJSON
	base := req.Messages
	var total Usage
	attempts := 0
	for turn := 0; ; turn++ {
		resp, err := c.Chat(ctx, req)
		total = total.Add(resp.Usage)
		attempts += resp.Attempts
		if err != nil {
			return Response{Usage: total, Attempts: attempts, Repairs: turn}, err
		}
		resp.Usage, resp.Attempts, resp.Repairs = total, attempts, turn
		raw := ExtractJSON(resp.Text)
		var problems []string
		code := CodeInvalidJSON
		switch {
		case raw == nil:
			problems = []string{"the reply contained no JSON value"}
			if resp.FinishReason == FinishLength {
				problems = []string{"the reply was cut off; send a shorter, complete JSON value"}
			}
		case req.Schema != nil:
			for _, se := range req.Schema.Validate(raw) {
				problems = append(problems, se.String())
			}
			if len(problems) > 0 {
				code = CodeSchema
			}
		}
		if len(problems) == 0 && out != nil {
			if err := json.Unmarshal(raw, out); err != nil {
				problems = []string{"decode: " + err.Error()}
				code = CodeSchema
			}
		}
		if len(problems) == 0 {
			resp.JSON = raw
			return resp, nil
		}
		if turn >= repairs {
			e := newErr(code, "detail", clip(strings.Join(problems, "; "), 300))
			if code == CodeSchema {
				e = e.with("path", firstPath(problems))
			}
			return resp, e
		}
		req.Messages = append(append([]Message{}, base...),
			Message{Role: Assistant, Content: clip(resp.Text, 8000)},
			Message{Role: User, Content: repairPrompt(problems)})
	}
}

func firstPath(problems []string) string {
	if len(problems) == 0 {
		return ""
	}
	p, _, _ := strings.Cut(problems[0], ": ")
	return p
}

func repairPrompt(problems []string) string {
	var b strings.Builder
	b.WriteString("Your previous reply could not be used:\n")
	for i, p := range problems {
		if i == 10 {
			fmt.Fprintf(&b, "- … and %d more\n", len(problems)-10)
			break
		}
		b.WriteString("- " + p + "\n")
	}
	b.WriteString("Send the corrected answer again as one JSON value only, with nothing before or after it.")
	return b.String()
}

// thinkTags are the reasoning blocks a model may open its answer with:
// Qwen's <think>…</think>, and Gemma 4's thought channel
// (<|channel>thought…<channel|>; it comes with thinking on, which
// VoxtNew never asks for, and its card says a model may still send an
// empty one).
var thinkTags = [][2]string{{"<think>", "</think>"}, {"<|channel>", "<channel|>"}}

// thinkOpen is the tag pair whose opening t starts with (nil for none);
// undecided when t could still become one.
func thinkOpen(t string) (tag *[2]string, undecided bool) {
	for i := range thinkTags {
		open := thinkTags[i][0]
		if strings.HasPrefix(t, open) {
			return &thinkTags[i], false
		}
		if len(t) < len(open) && strings.HasPrefix(open, t) {
			undecided = true
		}
	}
	return nil, undecided
}

// StripThinking removes a leading reasoning block (thinkTags: reasoning
// models, such as the Qwen and Gemma models voxtnew-llm runs).
func StripThinking(s string) string {
	t := strings.TrimLeft(s, " \t\r\n")
	tag, _ := thinkOpen(t)
	if tag == nil {
		return s
	}
	if i := strings.Index(t, tag[1]); i >= 0 {
		return strings.TrimLeft(t[i+len(tag[1]):], " \t\r\n")
	}
	return "" // still thinking when cut off
}

// thinkFilter hides a leading reasoning block (thinkTags) from stream
// deltas.
type thinkFilter struct {
	emit  func(string)
	buf   strings.Builder
	tag   *[2]string
	state int // 0 deciding, 1 inside think, 2 passing through, 3 after think (trim leading space)
}

func (f *thinkFilter) reset() { f.buf.Reset(); f.state, f.tag = 0, nil }

func (f *thinkFilter) write(s string) {
	switch f.state {
	case 2:
		f.emit(s)
		return
	case 3:
		s = strings.TrimLeft(s, " \t\r\n")
		if s != "" {
			f.state = 2
			f.emit(s)
		}
		return
	}
	f.buf.WriteString(s)
	b := f.buf.String()
	if f.state == 0 {
		tag, undecided := thinkOpen(strings.TrimLeft(b, " \t\r\n"))
		switch {
		case tag != nil:
			f.state, f.tag = 1, tag
		case undecided:
			return
		default:
			f.state = 2
			f.buf.Reset()
			f.emit(b)
			return
		}
	}
	if i := strings.Index(b, f.tag[1]); i >= 0 {
		rest := b[i+len(f.tag[1]):]
		f.buf.Reset()
		f.state = 3
		f.write(rest)
	}
}

func (f *thinkFilter) flush() {
	if f.state == 0 && f.buf.Len() > 0 {
		f.emit(f.buf.String())
	}
	f.buf.Reset()
}
