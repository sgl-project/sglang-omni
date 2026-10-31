package asr

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
)

// Kind classifies an error by what the user can do about it.
type Kind string

const (
	// KindConfig: the setup is incomplete or wrong before anything was
	// sent: a model that is not installed or not a speech model, a
	// missing folder.
	KindConfig Kind = "config"
	// KindAuth: the server refused the request (HTTP 401/403); the
	// local runtimes do not ask for credentials.
	KindAuth Kind = "auth"
	// KindQuota: the server is busy or refused for its own limits
	// (HTTP 429, 402).
	KindQuota Kind = "quota"
	// KindNetwork: the server could not be reached, or the connection
	// broke or timed out.
	KindNetwork Kind = "network"
	// KindProtocol: the server answered something this package does not
	// understand, or reported a bad request.
	KindProtocol Kind = "protocol"
	// KindServer: the server failed on its side (5xx, internal errors, a
	// runtime that did not start or exited).
	KindServer Kind = "server"
	// KindAudio: the audio was refused: too long, too large, a format it
	// does not take.
	KindAudio Kind = "audio"
	// KindCancelled: the caller cancelled.
	KindCancelled Kind = "cancelled"
	// KindTimeout: the model gave no result within the caller's limit,
	// though it had not stopped answering (Detail says the limit).
	KindTimeout Kind = "timeout"
)

// Error is every error this package and the speech models return.
type Error struct {
	Kind     Kind
	Provider string // the speech runtime's id ("omni"), "" when none
	// Status is the HTTP status (or the WebSocket handshake's), 0 for
	// none; Code is the server's own error code, "" for none.
	Status int
	Code   string
	// Detail is the server's message or what went wrong, for the log
	// and the ⓘ of an error.
	Detail string
	// Retryable: the same request may succeed later (network, server,
	// rate limits).
	Retryable bool
	Err       error // the cause
}

func (e *Error) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "asr %s: %s", e.Provider, e.Kind)
	if e.Status != 0 {
		fmt.Fprintf(&b, " (status %d)", e.Status)
	}
	if e.Code != "" {
		fmt.Fprintf(&b, " [%s]", e.Code)
	}
	if e.Detail != "" {
		b.WriteString(": " + e.Detail)
	}
	if e.Err != nil && (e.Detail == "" || !strings.Contains(e.Detail, e.Err.Error())) {
		b.WriteString(": " + e.Err.Error())
	}
	return b.String()
}

// Unwrap exposes the cause and the matching core sentinel.
func (e *Error) Unwrap() []error {
	var out []error
	if e.Err != nil {
		out = append(out, e.Err)
	}
	switch e.Kind {
	case KindCancelled:
		out = append(out, core.ErrCancelled)
	case KindConfig, KindAudio:
		out = append(out, core.ErrInvalid)
	case KindQuota:
		out = append(out, core.ErrBusy)
	}
	return out
}

// Is matches an *Error by Kind alone: errors.Is(err, &asr.Error{Kind: asr.KindAuth}).
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	return ok && t.Kind == e.Kind && t.Provider == "" && t.Status == 0 && t.Code == ""
}

// KindOf returns err's Kind if it is (or wraps) an *Error, else "".
func KindOf(err error) Kind {
	var e *Error
	if errors.As(err, &e) {
		return e.Kind
	}
	return ""
}

// Errorf is an *Error of kind for provider (a runtime's id) with a
// detail.
func Errorf(provider string, kind Kind, format string, args ...any) *Error {
	return &Error{Kind: kind, Provider: provider, Detail: fmt.Sprintf(format, args...)}
}

// HTTPError classifies an HTTP error status and the server's message
// (an error body, or a handshake's refusal).
func HTTPError(provider string, status int, code, detail string) *Error {
	e := &Error{Provider: provider, Status: status, Code: code, Detail: Clip(detail, 300)}
	low := strings.ToLower(code + " " + detail)
	switch {
	case status == 401 || status == 403:
		e.Kind = KindAuth
		if quotaWords(low) {
			e.Kind = KindQuota // a 403 that names a limit, not credentials
		}
	case status == 429:
		e.Kind, e.Retryable = KindQuota, true
		if strings.Contains(low, "insufficient_quota") || strings.Contains(low, "balance") || strings.Contains(low, "arrearage") {
			e.Retryable = false
		}
	case status == 402:
		e.Kind = KindQuota
	case status == 413:
		e.Kind = KindAudio
	case status == 408:
		e.Kind, e.Retryable = KindNetwork, true
	case status >= 500:
		e.Kind, e.Retryable = KindServer, true
	case status == 400 || status == 415 || status == 422:
		e.Kind = KindProtocol
		if strings.Contains(low, "audio") && (strings.Contains(low, "too long") || strings.Contains(low, "too large") || strings.Contains(low, "format") || strings.Contains(low, "duration")) {
			e.Kind = KindAudio
		}
		if strings.Contains(low, "model") && (strings.Contains(low, "not found") || strings.Contains(low, "does not exist") || strings.Contains(low, "invalid model")) {
			e.Kind = KindConfig
		}
	case status == 404:
		e.Kind = KindConfig // a wrong path or model
	default:
		e.Kind = KindProtocol
	}
	return e
}

// NoSpeech reports an error answer that only means nothing was said
// ("no speech found", "NO_VALID_AUDIO" and the like). A speech model
// turns it into an empty result, not an error, so silence reads as
// "nothing heard".
func NoSpeech(code, detail string) bool {
	low := strings.ToLower(code + " " + detail)
	low = strings.NewReplacer("_", " ", "-", " ").Replace(low)
	for _, w := range []string{"no speech", "no valid audio", "no valid fragment", "speech not detected", "no speech detected", "no voice detected", "speech not found"} {
		if strings.Contains(low, w) {
			return true
		}
	}
	return false
}

func quotaWords(low string) bool {
	for _, w := range []string{"quota", "arrearage", "balance", "insufficient", "not activated", "not open", "access denied for model", "free tier"} {
		if strings.Contains(low, w) {
			return true
		}
	}
	return false
}

// NetError classifies a transport error: the caller's cancellation, a
// timeout, or a network failure.
func NetError(ctx context.Context, provider string, err error) *Error {
	var e *Error
	if errors.As(err, &e) {
		return e
	}
	if ctx != nil && ctx.Err() != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return &Error{Kind: KindNetwork, Provider: provider, Detail: "timed out", Retryable: true, Err: err}
		}
		return &Error{Kind: KindCancelled, Provider: provider, Err: err}
	}
	if errors.Is(err, context.Canceled) {
		return &Error{Kind: KindCancelled, Provider: provider, Err: err}
	}
	detail := err.Error()
	x := &Error{Kind: KindNetwork, Provider: provider, Detail: Clip(detail, 300), Err: err, Retryable: true}
	if errors.Is(err, context.DeadlineExceeded) {
		x.Detail = "timed out"
	}
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		x.Detail = "timed out"
	}
	return x
}

// Clip shortens s to at most n bytes, on a rune boundary, with "…".
func Clip(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && s[cut]&0xC0 == 0x80 {
		cut--
	}
	return s[:cut] + "…"
}

// Timeouts shared by the speech models.
const (
	// ConnectTimeout bounds connecting and the handshake of a stream.
	ConnectTimeout = 15 * time.Second
	// FinalTimeout bounds the wait for the final text after Finish.
	FinalTimeout = 20 * time.Second
)
