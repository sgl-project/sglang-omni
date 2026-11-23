// Package codexauth keeps Voxt's "ChatGPT (Codex)" sign-in: the browser
// login with PKCE (login.go), and a Session that holds its tokens and
// refreshes them. It never reads or writes the Codex CLI's auth.json.
//
// The refresh token, the account id and the email are one Keychain item
// (Store); the access token stays in memory, so that a sign-in fits on the
// one line /usr/bin/security reads (secrets.Set) and each refresh writes
// one item. No token goes into an error, a log or a page: errors carry a
// kind and the service's short OAuth code, never a response body.
//
// Imports: nothing internal.
package codexauth

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ClientID is the Codex CLI's OAuth client, which this sign-in borrows:
// Voxt has no OAuth client of its own at OpenAI.
const ClientID = "app_EMoamEEZ73f0CkXaXp7hrann"

// Endpoints are the addresses of the OAuth service.
type Endpoints struct{ Authorize, Token string }

// OpenAI is OpenAI's OAuth service.
var OpenAI = Endpoints{Authorize: "https://auth.openai.com/oauth/authorize", Token: "https://auth.openai.com/oauth/token"}

// Store keeps the one Keychain item. Set("") deletes it; Get finds "" when
// there is none. Calls may block.
type Store interface {
	Get() (string, error)
	Set(value string) error
}

// Kind is what went wrong.
type Kind string

const (
	KindSignedOut Kind = "signed_out"  // not signed in, or the session was revoked or expired
	KindPortInUse Kind = "port_in_use" // Port is taken, and every other allowed port too
	KindDenied    Kind = "denied"      // the user or the service refused the login
	KindTimeout   Kind = "timeout"     // no callback in time
	KindExchange  Kind = "exchange"    // the code was not exchanged for tokens
	KindNetwork   Kind = "network"     // the service did not answer; the sign-in is kept
	KindStore     Kind = "store"       // the Keychain refused the item
)

// Error is every error of this package. It holds no token and no response
// body.
type Error struct {
	Kind   Kind
	Port   int    // KindPortInUse
	Status int    // the HTTP status, when there was one
	Code   string // the service's OAuth error code, when it gave one
	Err    error  // the cause; printed only for KindNetwork
}

func (e *Error) Error() string {
	var b strings.Builder
	b.WriteString("chatgpt sign-in: ")
	b.WriteString(string(e.Kind))
	if e.Port != 0 {
		fmt.Fprintf(&b, " port=%d", e.Port)
	}
	if e.Status != 0 {
		fmt.Fprintf(&b, " status=%d", e.Status)
	}
	if e.Code != "" {
		fmt.Fprintf(&b, " code=%s", e.Code)
	}
	if e.Kind == KindNetwork && e.Err != nil {
		b.WriteString(": ")
		b.WriteString(e.Err.Error())
	}
	return b.String()
}

func (e *Error) Unwrap() error { return e.Err }

// Is matches another *Error by Kind.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	return ok && t.Kind == e.Kind
}

// ErrSignedOut is a Token with no sign-in to use.
var ErrSignedOut = &Error{Kind: KindSignedOut}

// Config is a Session's surroundings.
type Config struct {
	Store     Store
	Endpoints Endpoints
	// HTTP is the client for the OAuth service (nil result: the default).
	HTTP func() *http.Client
	// Ports are the login callback's ports; nil: the ones the client allows.
	Ports []int
	// LoginTimeout bounds a login; 0: five minutes.
	LoginTimeout time.Duration
	// Success and Failure are the words the browser shows after the login.
	Success, Failure string
	// SignedOut is called, off the main thread, when a refresh found the
	// sign-in dead and removed it.
	SignedOut func()
	Now       func() time.Time // nil: time.Now
}

// Credential is what one request sends.
type Credential struct{ Access, AccountID string }

// stored is the Keychain item's content.
type stored struct {
	Refresh   string `json:"r"`
	AccountID string `json:"a,omitempty"`
	Email     string `json:"e,omitempty"`
}

// maxStored keeps the item on the one line of at most 4096 bytes that
// /usr/bin/security reads in interactive mode (secrets.Item.Set), with
// room for the rest of its command.
const maxStored = 3900

func (s stored) encode() (string, error) {
	raw, err := json.Marshal(s)
	if err != nil {
		return "", err
	}
	v := base64.RawURLEncoding.EncodeToString(raw)
	if len(v) > maxStored {
		return "", errors.New("the sign-in is too long for the Keychain")
	}
	return v, nil
}

func decode(v string) (stored, bool) {
	raw, err := base64.RawURLEncoding.DecodeString(v)
	if err != nil {
		return stored{}, false
	}
	var s stored
	if json.Unmarshal(raw, &s) != nil || s.Refresh == "" {
		return stored{}, false
	}
	return s, true
}

// Email is the account's email in a saved item, "" when the item has none
// or is not one.
func Email(item string) string {
	s, _ := decode(item)
	return s.Email
}

// Session is one sign-in, safe for concurrent use.
type Session struct {
	cfg Config

	mu      sync.Mutex
	loaded  bool
	st      stored
	access  string
	expires time.Time // zero: unknown, used until the service refuses it
	call    *refreshCall
	gen     int // a new login or a sign-out: a refresh still running is void
}

type refreshCall struct {
	done chan struct{}
	cred Credential
	err  error
}

// NewSession is a session over cfg; it reads the Store when first needed.
func NewSession(cfg Config) *Session {
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if len(cfg.Ports) == 0 {
		cfg.Ports = callbackPorts
	}
	if cfg.LoginTimeout == 0 {
		cfg.LoginTimeout = loginTimeout
	}
	return &Session{cfg: cfg}
}

func (s *Session) client() *http.Client {
	if s.cfg.HTTP != nil {
		if c := s.cfg.HTTP(); c != nil {
			return c
		}
	}
	return http.DefaultClient
}

// refreshBefore treats an access token this close to its expiry as expired
// (as Voxt 1.x did).
const refreshBefore = 60 * time.Second

// refreshTimeout bounds a refresh (as Voxt 1.x did).
const refreshTimeout = 20 * time.Second

// load reads the Keychain once. mu is held.
func (s *Session) load() error {
	if s.loaded {
		return nil
	}
	v, err := s.cfg.Store.Get()
	if err != nil {
		return &Error{Kind: KindStore, Err: err}
	}
	s.loaded = true
	if v != "" {
		s.st, _ = decode(v)
	}
	return nil
}

// Token is a credential for one request. stale, when not "", is the access
// token the service just refused. Requests that find the token expired
// together share one refresh, which goes on when they are cancelled: a
// rotated refresh token must reach the Keychain.
func (s *Session) Token(ctx context.Context, stale string) (Credential, error) {
	s.mu.Lock()
	if err := s.load(); err != nil {
		s.mu.Unlock()
		return Credential{}, err
	}
	if s.st.Refresh == "" {
		s.mu.Unlock()
		return Credential{}, ErrSignedOut
	}
	if s.access != "" && s.access != stale && (s.expires.IsZero() || s.cfg.Now().Add(refreshBefore).Before(s.expires)) {
		c := Credential{Access: s.access, AccountID: s.st.AccountID}
		s.mu.Unlock()
		return c, nil
	}
	call := s.call
	if call == nil {
		call = &refreshCall{done: make(chan struct{})}
		s.call = call
		go s.refresh(call, s.st, s.gen)
	}
	s.mu.Unlock()
	select {
	case <-call.done:
		return call.cred, call.err
	case <-ctx.Done():
		return Credential{}, ctx.Err()
	}
}

func (s *Session) refresh(call *refreshCall, st stored, gen int) {
	ctx, cancel := context.WithTimeout(context.Background(), refreshTimeout)
	defer cancel()
	got, err := s.postRefresh(ctx, st.Refresh)
	signedOut := false

	s.mu.Lock()
	switch {
	case s.gen != gen:
		call.err = ErrSignedOut // signed out or in again meanwhile
	case err != nil:
		call.err = err
		if errors.Is(err, ErrSignedOut) {
			s.st, s.access, s.expires = stored{}, "", time.Time{}
			signedOut = true
		}
	default:
		next := s.merge(st, got)
		s.st, s.access, s.expires = next, got.Access, got.expires(s.cfg.Now())
		call.cred = Credential{Access: s.access, AccountID: next.AccountID}
		call.err = s.save(next)
	}
	s.call = nil
	s.mu.Unlock()

	if signedOut {
		// Deleted so that the next use asks to sign in instead of trying
		// this refresh again; a failed delete only repeats that one try
		// after a restart.
		_ = s.cfg.Store.Set("")
		if s.cfg.SignedOut != nil {
			s.cfg.SignedOut()
		}
	}
	close(call.done)
}

// merge keeps what a token response left out.
func (s *Session) merge(st stored, got tokens) stored {
	if got.Refresh != "" {
		st.Refresh = got.Refresh
	}
	acct, email := got.identity()
	if acct != "" {
		st.AccountID = acct
	}
	if email != "" {
		st.Email = email
	}
	return st
}

// save writes the item. mu may be held: Store calls are short, and a
// sign-in written out of order would lose the newest refresh token.
func (s *Session) save(st stored) error {
	v, err := st.encode()
	if err != nil {
		return &Error{Kind: KindStore, Err: err}
	}
	if err := s.cfg.Store.Set(v); err != nil {
		return &Error{Kind: KindStore, Err: err}
	}
	return nil
}

// SignOut forgets the sign-in here and deletes the Keychain item. It tells
// the service nothing.
func (s *Session) SignOut() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.gen++
	s.loaded, s.st, s.access, s.expires = true, stored{}, "", time.Time{}
	if err := s.cfg.Store.Set(""); err != nil {
		return &Error{Kind: KindStore, Err: err}
	}
	return nil
}
