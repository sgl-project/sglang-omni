package llm

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net"
	"sort"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
)

// Error codes. The UI maps them to messages; Params carry the details
// named in each comment.
const (
	CodeAuth          = "llm.auth"             // provider, status: refused (401/403)
	CodeRateLimited   = "llm.rate_limited"     // provider, retry_after (seconds, may be empty)
	CodeServer        = "llm.server"           // provider, status: 5xx
	CodeOverloaded    = "llm.overloaded"       // provider: 529 / "overloaded"
	CodeBadRequest    = "llm.bad_request"      // provider, status, detail
	CodeModelNotFound = "llm.model_not_found"  // provider, model
	CodeContextLength = "llm.context_length"   // provider, model: prompt too long
	CodeFiltered      = "llm.content_filter"   // provider: refused by a safety filter
	CodeTimeout       = "llm.timeout"          // provider, seconds; without progress also wait (WaitFirstToken, WaitStall), deltas, elapsed
	CodeCancelled     = "llm.cancelled"        //
	CodeNetwork       = "llm.network"          // provider, detail (connection refused, reset)
	CodeBadResponse   = "llm.bad_response"     // provider, detail: unparseable body / stream
	CodeEmpty         = "llm.empty"            // provider: no text
	CodeInvalidJSON   = "llm.invalid_json"     // detail: no JSON value in the reply after repairs
	CodeSchema        = "llm.schema"           // path, detail: JSON does not match the schema after repairs
	CodeNoRoute       = "llm.no_route"         // skill: the Client has no Backend
	CodeUnknown       = "llm.unknown_provider" // provider
	CodeUnsupported   = "llm.unsupported"      // provider, feature
	CodeConfig        = "llm.config"           // provider, field: bad settings
)

// Error is every error this package returns.
type Error struct {
	Code       string
	Params     map[string]string
	Status     int           // HTTP status, 0 when none
	Retryable  bool          // a retry (same backend) may succeed
	RetryAfter time.Duration // the server's Retry-After, 0 when none
	Err        error         // the cause, if any
}

func (e *Error) Error() string {
	var b strings.Builder
	b.WriteString(e.Code)
	keys := make([]string, 0, len(e.Params))
	for k := range e.Params {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Fprintf(&b, " %s=%q", k, e.Params[k])
	}
	if e.Err != nil {
		b.WriteString(": ")
		b.WriteString(e.Err.Error())
	}
	return b.String()
}

// Unwrap exposes the cause and the matching core sentinel, so callers can
// test errors.Is(err, core.ErrCancelled) or context.Canceled.
func (e *Error) Unwrap() []error {
	var out []error
	if e.Err != nil {
		out = append(out, e.Err)
	}
	switch e.Code {
	case CodeCancelled:
		out = append(out, core.ErrCancelled)
	case CodeModelNotFound, CodeUnknown:
		out = append(out, core.ErrNotFound)
	case CodeBadRequest, CodeContextLength, CodeConfig, CodeInvalidJSON, CodeSchema:
		out = append(out, core.ErrInvalid)
	case CodeUnsupported, CodeNoRoute:
		out = append(out, core.ErrUnsupported)
	case CodeRateLimited, CodeOverloaded:
		out = append(out, core.ErrBusy)
	}
	return out
}

// Is matches another *Error by Code, so errors.Is(err, &llm.Error{Code: llm.CodeAuth}) works.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	return ok && t.Code == e.Code && t.Params == nil
}

// Code returns err's code if it is (or wraps) an *Error, else "".
func Code(err error) string {
	var e *Error
	if errors.As(err, &e) {
		return e.Code
	}
	return ""
}

func newErr(code string, params ...string) *Error {
	e := &Error{Code: code}
	if len(params) > 0 {
		e.Params = map[string]string{}
		for i := 0; i+1 < len(params); i += 2 {
			e.Params[params[i]] = params[i+1]
		}
	}
	return e
}

func (e *Error) with(k, v string) *Error {
	c := *e
	c.Params = maps.Clone(e.Params)
	if c.Params == nil {
		c.Params = map[string]string{}
	}
	c.Params[k] = v
	return &c
}

// ctxErr converts a transport error into an *Error, distinguishing
// cancellation, deadlines and network failures.
func ctxErr(ctx context.Context, provider string, err error, timeout time.Duration) *Error {
	var e *Error
	if errors.As(err, &e) {
		return e
	}
	if ctx.Err() != nil {
		if errors.Is(context.Cause(ctx), context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			x := newErr(CodeTimeout, "provider", provider, "seconds", fmt.Sprintf("%.0f", timeout.Seconds()))
			x.Retryable, x.Err = true, err
			return x
		}
		x := newErr(CodeCancelled)
		x.Err = ctx.Err()
		return x
	}
	if errors.Is(err, context.DeadlineExceeded) {
		x := newErr(CodeTimeout, "provider", provider, "seconds", fmt.Sprintf("%.0f", timeout.Seconds()))
		x.Retryable, x.Err = true, err
		return x
	}
	var ne net.Error
	detail := err.Error()
	x := newErr(CodeNetwork, "provider", provider, "detail", detail)
	x.Err = err
	// Connection refused (a local server that isn't running) won't fix
	// itself within a backoff window; resets and timeouts might.
	x.Retryable = !strings.Contains(detail, "connection refused")
	if errors.As(err, &ne) && ne.Timeout() {
		x.Retryable = true
	}
	return x
}

// HTTPError maps an HTTP error status (and the server's error body) to an
// *Error, for a backend over (loopback) HTTP.
func HTTPError(provider, model string, status int, detail string, retryAfter time.Duration) *Error {
	low := strings.ToLower(detail)
	var e *Error
	switch {
	case status == 401 || status == 403:
		e = newErr(CodeAuth, "provider", provider, "status", fmt.Sprint(status))
	case status == 429:
		e = newErr(CodeRateLimited, "provider", provider, "retry_after", retryAfterParam(retryAfter))
		e.Retryable = true
		// A 429 that names a spent quota will not pass on a retry.
		if strings.Contains(low, "insufficient_quota") || strings.Contains(low, "quota exceeded") && !strings.Contains(low, "per minute") {
			e.Retryable = false
		}
	case status == 529 || strings.Contains(low, "overloaded"):
		e = newErr(CodeOverloaded, "provider", provider)
		e.Retryable = true
	case status == 404 || (status == 400 && (strings.Contains(low, "model_not_found") || strings.Contains(low, "model not found") || strings.Contains(low, "does not exist") || strings.Contains(low, "model not exist"))):
		e = newErr(CodeModelNotFound, "provider", provider, "model", model)
	case status == 400 || status == 413 || status == 422:
		if strings.Contains(low, "context_length") || strings.Contains(low, "context length") || strings.Contains(low, "too long") || strings.Contains(low, "maximum context") || status == 413 {
			e = newErr(CodeContextLength, "provider", provider, "model", model)
		} else {
			e = newErr(CodeBadRequest, "provider", provider, "status", fmt.Sprint(status), "detail", clip(detail, 300))
		}
	case status == 408:
		e = newErr(CodeTimeout, "provider", provider, "seconds", "")
		e.Retryable = true
	case status >= 500:
		e = newErr(CodeServer, "provider", provider, "status", fmt.Sprint(status))
		e.Retryable = true
	default:
		e = newErr(CodeBadRequest, "provider", provider, "status", fmt.Sprint(status), "detail", clip(detail, 300))
	}
	e.Status, e.RetryAfter = status, retryAfter
	return e
}

func retryAfterParam(d time.Duration) string {
	if d <= 0 {
		return ""
	}
	return fmt.Sprintf("%.0f", d.Seconds())
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && cut < len(s) && s[cut]&0xC0 == 0x80 {
		cut--
	}
	return s[:cut] + "…"
}
