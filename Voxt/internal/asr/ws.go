package asr

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"time"

	"github.com/coder/websocket"
)

// WSClient is the HTTP client WebSocket handshakes go through: base's
// proxy (the app's network settings; base nil or not an *http.Transport:
// none), HTTP/1.1 only (a WebSocket upgrade needs it), dial and TLS
// timeouts, and no Client.Timeout (the websocket package refuses one).
func WSClient(base *http.Client) *http.Client {
	var proxy func(*http.Request) (*url.URL, error)
	if base != nil {
		if t, ok := base.Transport.(*http.Transport); ok {
			proxy = t.Proxy
		}
	}
	return &http.Client{Transport: &http.Transport{
		Proxy:               proxy,
		DialContext:         (&net.Dialer{Timeout: ConnectTimeout, KeepAlive: 30 * time.Second}).DialContext,
		TLSHandshakeTimeout: ConnectTimeout,
		TLSNextProto:        map[string]func(string, *tls.Conn) http.RoundTripper{}, // no HTTP/2
	}}
}

// WSOptions is how Dial connects.
type WSOptions struct {
	// Client makes the handshake (WSClient); nil: WSClient(nil).
	Client *http.Client
	Header http.Header
	// ReadLimit bounds one message from the server (0: 4 MiB).
	ReadLimit int64
}

// Dial opens a WebSocket to u (checked with CheckEndpoint first) for
// provider, classifying a refused handshake: 401/403 is KindAuth, 429
// KindQuota, and so on, with the body of the refusal as the detail. The
// connection's lifetime is not bound to ctx (only the handshake is).
func Dial(ctx context.Context, provider, u string, o WSOptions) (*websocket.Conn, *http.Response, error) {
	client := o.Client
	if client == nil {
		client = WSClient(nil)
	}
	hctx, cancel := context.WithTimeout(ctx, ConnectTimeout)
	defer cancel()
	conn, resp, err := websocket.Dial(hctx, u, &websocket.DialOptions{HTTPClient: client, HTTPHeader: o.Header})
	if err != nil {
		if resp != nil && resp.StatusCode != http.StatusSwitchingProtocols && resp.StatusCode != 0 {
			var body []byte
			if resp.Body != nil {
				body, _ = io.ReadAll(io.LimitReader(resp.Body, 16<<10))
				resp.Body.Close()
			}
			code, msg := ErrorBody(body)
			if msg == "" {
				msg = resp.Status
			}
			return nil, resp, HTTPError(provider, resp.StatusCode, code, msg)
		}
		if ctx.Err() == nil && hctx.Err() != nil {
			return nil, resp, &Error{Kind: KindNetwork, Provider: provider, Detail: "the handshake timed out", Retryable: true, Err: err}
		}
		return nil, resp, NetError(ctx, provider, err)
	}
	limit := o.ReadLimit
	if limit == 0 {
		limit = 4 << 20
	}
	conn.SetReadLimit(limit)
	return conn, resp, nil
}

// WSError classifies an error of a WebSocket read or write: a close frame
// with a status (the provider's reason as the detail), the caller's
// cancellation, or a network failure.
func WSError(ctx context.Context, provider string, err error) *Error {
	var e *Error
	if errors.As(err, &e) {
		return e
	}
	if ctx.Err() != nil {
		return NetError(ctx, provider, err)
	}
	var ce websocket.CloseError
	if errors.As(err, &ce) {
		x := &Error{Kind: KindProtocol, Provider: provider, Code: ce.Code.String(), Detail: Clip(ce.Reason, 300), Err: err}
		switch ce.Code {
		case websocket.StatusPolicyViolation:
			x.Kind = KindAuth
			if quotaWords(lower(ce.Reason)) {
				x.Kind = KindQuota
			}
		case websocket.StatusInternalError, websocket.StatusTryAgainLater, websocket.StatusBadGateway, websocket.StatusServiceRestart:
			x.Kind, x.Retryable = KindServer, true
		case websocket.StatusMessageTooBig:
			x.Kind = KindAudio
		case websocket.StatusNormalClosure, websocket.StatusGoingAway:
			x.Code = "" // not an error code of the service's
			x.Detail = "the service closed the connection early" + reasonSuffix(ce.Reason)
		}
		return x
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, net.ErrClosed) {
		return &Error{Kind: KindNetwork, Provider: provider, Detail: "the connection closed", Retryable: true, Err: err}
	}
	return NetError(ctx, provider, err)
}

func reasonSuffix(r string) string {
	if r == "" {
		return ""
	}
	return ": " + Clip(r, 200)
}
