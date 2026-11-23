package codexauth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

// callbackPorts are the ports the Codex CLI's client allows in its redirect
// address, the usual one first.
var callbackPorts = []int{1455, 1457}

const (
	callbackPath = "/auth/callback"
	scope        = "openid profile email offline_access"
	originator   = "codex_cli_rs"
	loginTimeout = 5 * time.Minute
	// shutdownGrace lets the browser's page finish before the listener goes.
	shutdownGrace = 2 * time.Second
)

// tokens is the token endpoint's answer.
type tokens struct {
	ID        string `json:"id_token"`
	Access    string `json:"access_token"`
	Refresh   string `json:"refresh_token"`
	ExpiresIn int64  `json:"expires_in"`
}

// expires is when the access token ends: its JWT's exp, else expires_in,
// else zero (unknown).
func (t tokens) expires(now time.Time) time.Time {
	var c struct {
		Exp int64 `json:"exp"`
	}
	if jwtClaims(t.Access, &c) && c.Exp > 0 {
		return time.Unix(c.Exp, 0)
	}
	if t.ExpiresIn > 0 {
		return now.Add(time.Duration(t.ExpiresIn) * time.Second)
	}
	return time.Time{}
}

// identity reads the account id and email from the ID token, the account
// id also from the access token (as Voxt 1.x did). The token came straight
// from the token endpoint over TLS, so its signature adds nothing here.
func (t tokens) identity() (accountID, email string) {
	var c struct {
		Email string `json:"email"`
		Auth  struct {
			AccountID string `json:"chatgpt_account_id"`
		} `json:"https://api.openai.com/auth"`
	}
	if jwtClaims(t.ID, &c) {
		accountID, email = c.Auth.AccountID, c.Email
	}
	if accountID == "" && jwtClaims(t.Access, &c) {
		accountID = c.Auth.AccountID
	}
	return accountID, email
}

func jwtClaims(token string, v any) bool {
	parts := strings.Split(token, ".")
	if len(parts) < 2 {
		return false
	}
	raw, err := base64.RawURLEncoding.DecodeString(parts[1])
	return err == nil && json.Unmarshal(raw, v) == nil
}

func randomString(n int) (string, error) {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

// listen binds the first free callback port, on the loopback only. A port
// taken by something else is not an error until every port is.
func (s *Session) listen() (net.Listener, int, error) {
	var last int
	for _, port := range s.cfg.Ports {
		l, err := net.Listen("tcp4", fmt.Sprintf("127.0.0.1:%d", port))
		if err == nil {
			return l, port, nil
		}
		if !errors.Is(err, syscall.EADDRINUSE) {
			return nil, 0, &Error{Kind: KindNetwork, Err: err}
		}
		last = port
	}
	return nil, 0, &Error{Kind: KindPortInUse, Port: last}
}

// authorizeURL is the page the browser opens.
func (s *Session) authorizeURL(redirect, state, challenge string) string {
	q := url.Values{
		"response_type":              {"code"},
		"client_id":                  {ClientID},
		"redirect_uri":               {redirect},
		"scope":                      {scope},
		"state":                      {state},
		"code_challenge":             {challenge},
		"code_challenge_method":      {"S256"},
		"id_token_add_organizations": {"true"},
		"codex_cli_simplified_flow":  {"true"},
		"originator":                 {originator},
	}
	return s.cfg.Endpoints.Authorize + "?" + q.Encode()
}

// Login signs in through the browser: it listens on the loopback, has open
// show the authorization page and waits for the callback, ctx's end or the
// login timeout, then closes the listener. It returns the account's email
// ("" when the service gave none).
func (s *Session) Login(ctx context.Context, open func(url string) error) (string, error) {
	verifier, err := randomString(64)
	if err != nil {
		return "", &Error{Kind: KindExchange, Err: err}
	}
	state, err := randomString(32)
	if err != nil {
		return "", &Error{Kind: KindExchange, Err: err}
	}
	listener, port, err := s.listen()
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(ctx, s.cfg.LoginTimeout)
	defer cancel()

	redirect := fmt.Sprintf("http://127.0.0.1:%d%s", port, callbackPath)
	sum := sha256.Sum256([]byte(verifier))
	cb := &callback{s: s, ctx: ctx, state: state, verifier: verifier, redirect: redirect, result: make(chan loginResult, 1)}
	srv := &http.Server{Handler: cb, ReadHeaderTimeout: 10 * time.Second}
	served := make(chan struct{})
	go func() {
		defer close(served)
		_ = srv.Serve(listener)
	}()
	defer func() {
		grace, stop := context.WithTimeout(context.Background(), shutdownGrace)
		defer stop()
		if srv.Shutdown(grace) != nil {
			srv.Close()
		}
		<-served
	}()

	if err := open(s.authorizeURL(redirect, state, base64.RawURLEncoding.EncodeToString(sum[:]))); err != nil {
		return "", &Error{Kind: KindExchange, Err: err}
	}
	select {
	case r := <-cb.result:
		if r.err != nil {
			return "", r.err
		}
		return s.finishLogin(r.tokens)
	case <-ctx.Done():
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return "", &Error{Kind: KindTimeout}
		}
		return "", ctx.Err()
	}
}

func (s *Session) finishLogin(t tokens) (string, error) {
	if t.Refresh == "" || t.Access == "" {
		return "", &Error{Kind: KindExchange, Code: "missing_token"}
	}
	st := s.merge(stored{}, t)
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.save(st); err != nil {
		return "", err
	}
	s.gen++
	s.loaded, s.st, s.access, s.expires = true, st, t.Access, t.expires(s.cfg.Now())
	return st.Email, nil
}

type loginResult struct {
	tokens tokens
	err    error
}

// callback is the loopback server's handler.
type callback struct {
	s                         *Session
	ctx                       context.Context
	state, verifier, redirect string
	claimed                   atomic.Bool // one answer only: the state is good once
	result                    chan loginResult
}

func (cb *callback) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != callbackPath {
		http.NotFound(w, r)
		return
	}
	q := r.URL.Query()
	// A request without the state is not the browser coming back: it is
	// refused, and the login goes on waiting for the real one.
	if subtle.ConstantTimeCompare([]byte(q.Get("state")), []byte(cb.state)) != 1 || !cb.claimed.CompareAndSwap(false, true) {
		cb.page(w, http.StatusBadRequest, cb.s.cfg.Failure)
		return
	}
	var res loginResult
	switch code := q.Get("code"); {
	case q.Get("error") != "":
		res.err = &Error{Kind: KindDenied, Code: oauthCode(q.Get("error"))}
	case code == "":
		res.err = &Error{Kind: KindExchange, Code: "missing_code"}
	default:
		res.tokens, res.err = cb.s.postExchange(cb.ctx, code, cb.redirect, cb.verifier)
	}
	if res.err != nil {
		cb.page(w, http.StatusBadRequest, cb.s.cfg.Failure)
	} else {
		cb.page(w, http.StatusOK, cb.s.cfg.Success)
	}
	cb.result <- res
}

func (cb *callback) page(w http.ResponseWriter, status int, text string) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(status)
	fmt.Fprintf(w, "<!doctype html><meta charset=utf-8><title>Voxt</title><p style=\"font:16px system-ui;margin:4em auto;max-width:30em\">%s</p>", html.EscapeString(text))
}

var oauthCodePattern = regexp.MustCompile(`^[a-zA-Z0-9_.-]{1,64}$`)

// oauthCode is a code from the service, or "other" for anything that does
// not look like one: an error's fields must not carry free text.
func oauthCode(c string) string {
	if oauthCodePattern.MatchString(c) {
		return c
	}
	return "other"
}

// postExchange trades the authorization code for tokens.
func (s *Session) postExchange(ctx context.Context, code, redirect, verifier string) (tokens, error) {
	form := url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"redirect_uri":  {redirect},
		"client_id":     {ClientID},
		"code_verifier": {verifier},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.Endpoints.Token, strings.NewReader(form.Encode()))
	if err != nil {
		return tokens{}, &Error{Kind: KindExchange, Err: err}
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	t, status, oc, err := s.postToken(req)
	switch {
	case err != nil:
		return tokens{}, &Error{Kind: KindNetwork, Err: err}
	case status/100 != 2:
		return tokens{}, &Error{Kind: KindExchange, Status: status, Code: oc}
	}
	return t, nil
}

// postRefresh gets a new access token. A refusal of the refresh token
// itself is ErrSignedOut's kind; anything else leaves the sign-in alone.
func (s *Session) postRefresh(ctx context.Context, refresh string) (tokens, error) {
	body, _ := json.Marshal(map[string]string{"client_id": ClientID, "grant_type": "refresh_token", "refresh_token": refresh})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.Endpoints.Token, strings.NewReader(string(body)))
	if err != nil {
		return tokens{}, &Error{Kind: KindNetwork, Err: err}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	t, status, oc, err := s.postToken(req)
	switch {
	case err != nil:
		return tokens{}, &Error{Kind: KindNetwork, Err: err}
	case status == 400 || status == 401:
		return tokens{}, &Error{Kind: KindSignedOut, Status: status, Code: oc}
	case status/100 != 2 || t.Access == "":
		return tokens{}, &Error{Kind: KindNetwork, Status: status, Code: oc}
	}
	return t, nil
}

// postToken sends a token request and returns its tokens, status and OAuth
// error code. The cause, when err is set, is the transport's, which holds
// the address and never the body.
func (s *Session) postToken(req *http.Request) (t tokens, status int, code string, err error) {
	resp, err := s.client().Do(req)
	if err != nil {
		return tokens{}, 0, "", err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return tokens{}, 0, "", err
	}
	if resp.StatusCode/100 != 2 {
		var e struct {
			Error any `json:"error"`
		}
		_ = json.Unmarshal(raw, &e)
		switch v := e.Error.(type) {
		case string:
			code = oauthCode(v)
		case map[string]any:
			if c, ok := v["code"].(string); ok {
				code = oauthCode(c)
			}
		}
		return tokens{}, resp.StatusCode, code, nil
	}
	if json.Unmarshal(raw, &t) != nil {
		return tokens{}, resp.StatusCode, "bad_response", nil
	}
	return t, resp.StatusCode, "", nil
}
