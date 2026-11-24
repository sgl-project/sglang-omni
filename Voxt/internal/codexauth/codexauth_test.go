package codexauth

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"
)

// Made-up tokens: no test may ever see a real one.
const (
	fakeRefresh  = "rt-FAKE-refresh-0001"
	fakeRefresh2 = "rt-FAKE-refresh-0002"
)

func jwt(claims map[string]any) string {
	raw, _ := json.Marshal(claims)
	return "eyJhbGciOiJub25lIn0." + base64.RawURLEncoding.EncodeToString(raw) + ".sig"
}

type memStore struct {
	mu     sync.Mutex
	v      string
	sets   int
	setErr error
}

func (m *memStore) Get() (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.v, nil
}

func (m *memStore) Set(v string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sets++
	if m.setErr != nil {
		return m.setErr
	}
	m.v = v
	return nil
}

func (m *memStore) value() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.v
}

// authServer is a fake of the OAuth service's token endpoint.
type authServer struct {
	*httptest.Server
	mu    sync.Mutex
	forms []url.Values
	json  []map[string]string
	reply func(w http.ResponseWriter, r *http.Request)
}

func newAuthServer(t *testing.T) *authServer {
	a := &authServer{}
	a.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/oauth/token" {
			t.Errorf("request to %s", r.URL.Path)
		}
		a.mu.Lock()
		if strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
			var m map[string]string
			_ = json.NewDecoder(r.Body).Decode(&m)
			a.json = append(a.json, m)
		} else {
			_ = r.ParseForm()
			a.forms = append(a.forms, r.PostForm)
		}
		reply := a.reply
		a.mu.Unlock()
		reply(w, r)
	}))
	t.Cleanup(a.Close)
	return a
}

func (a *authServer) calls() (forms, refreshes int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.forms), len(a.json)
}

func grant(w http.ResponseWriter, access, refresh, email string) {
	id := jwt(map[string]any{"email": email, "https://api.openai.com/auth": map[string]any{"chatgpt_account_id": "acct-1"}})
	json.NewEncoder(w).Encode(map[string]any{"id_token": id, "access_token": access, "refresh_token": refresh})
}

func freePort(t *testing.T) int {
	l, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func session(t *testing.T, a *authServer, store *memStore, mut ...func(*Config)) *Session {
	cfg := Config{
		Store:     store,
		Endpoints: Endpoints{Authorize: a.URL + "/oauth/authorize", Token: a.URL + "/oauth/token"},
		Ports:     []int{freePort(t)},
		Success:   "ok-page",
		Failure:   "bad-page",
	}
	for _, m := range mut {
		m(&cfg)
	}
	return NewSession(cfg)
}

func accessJWT(until time.Time) string { return jwt(map[string]any{"exp": until.Unix()}) }

func parseAuthorize(t *testing.T, raw string) url.Values {
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	return u.Query()
}

func callbackGet(t *testing.T, q url.Values, query string) (int, string) {
	t.Helper()
	resp, err := http.Get(q.Get("redirect_uri") + "?" + query)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body := new(strings.Builder)
	fmt.Fprint(body, resp.Status)
	buf := make([]byte, 4096)
	n, _ := resp.Body.Read(buf)
	return resp.StatusCode, string(buf[:n])
}

// loginVia runs Login and answers its authorization page with browser.
func loginVia(t *testing.T, s *Session, browser func(q url.Values)) (string, error) {
	t.Helper()
	return s.Login(context.Background(), func(u string) error {
		go browser(parseAuthorize(t, u))
		return nil
	})
}

func assertClosed(t *testing.T, redirect string) {
	t.Helper()
	u, _ := url.Parse(redirect)
	if c, err := net.DialTimeout("tcp4", u.Host, 200*time.Millisecond); err == nil {
		c.Close()
		t.Errorf("the callback listener is still open on %s", u.Host)
	}
}

func TestLogin(t *testing.T) {
	a := newAuthServer(t)
	access := accessJWT(time.Now().Add(time.Hour))
	a.reply = func(w http.ResponseWriter, r *http.Request) { grant(w, access, fakeRefresh, "me@example.com") }
	store := &memStore{}
	s := session(t, a, store)

	var seen url.Values
	email, err := loginVia(t, s, func(q url.Values) {
		seen = q
		if code, body := callbackGet(t, q, "code=the-code&state="+q.Get("state")); code != 200 || !strings.Contains(body, "ok-page") {
			t.Errorf("callback %d %q", code, body)
		}
	})
	if err != nil || email != "me@example.com" {
		t.Fatalf("Login %q %v", email, err)
	}
	for k, want := range map[string]string{"client_id": ClientID, "response_type": "code", "scope": "openid profile email offline_access", "code_challenge_method": "S256", "originator": "codex_cli_rs", "id_token_add_organizations": "true", "codex_cli_simplified_flow": "true"} {
		if seen.Get(k) != want {
			t.Errorf("authorize %s = %q, want %q", k, seen.Get(k), want)
		}
	}
	if !strings.HasPrefix(seen.Get("redirect_uri"), "http://127.0.0.1:") || !strings.HasSuffix(seen.Get("redirect_uri"), "/auth/callback") {
		t.Errorf("redirect_uri %q", seen.Get("redirect_uri"))
	}
	if len(a.forms) != 1 {
		t.Fatalf("token requests %d", len(a.forms))
	}
	f := a.forms[0]
	sum := sha256.Sum256([]byte(f.Get("code_verifier")))
	if f.Get("grant_type") != "authorization_code" || f.Get("code") != "the-code" || f.Get("client_id") != ClientID || f.Get("redirect_uri") != seen.Get("redirect_uri") ||
		base64.RawURLEncoding.EncodeToString(sum[:]) != seen.Get("code_challenge") {
		t.Errorf("token form %v, challenge %q", f, seen.Get("code_challenge"))
	}
	item := store.value()
	if Email(item) != "me@example.com" || strings.Contains(item, access) {
		t.Errorf("stored %q", item)
	}
	if st, ok := decode(item); !ok || st.Refresh != fakeRefresh || st.AccountID != "acct-1" {
		t.Errorf("decoded %+v %v", st, ok)
	}
	if c, err := s.Token(context.Background(), ""); err != nil || c.Access != access || c.AccountID != "acct-1" {
		t.Errorf("Token %+v %v", c, err)
	}
	assertClosed(t, seen.Get("redirect_uri"))
	if _, refreshes := a.calls(); refreshes != 0 {
		t.Errorf("%d refreshes after a login", refreshes)
	}
}

// A forged request (no or wrong state) gets no exchange, and the real
// callback still works.
func TestLoginChecksState(t *testing.T) {
	a := newAuthServer(t)
	a.reply = func(w http.ResponseWriter, r *http.Request) {
		grant(w, accessJWT(time.Now().Add(time.Hour)), fakeRefresh, "me@example.com")
	}
	s := session(t, a, &memStore{})
	_, err := loginVia(t, s, func(q url.Values) {
		for _, bad := range []string{"code=x", "code=x&state=forged", "error=access_denied&state=forged"} {
			if code, body := callbackGet(t, q, bad); code != 400 || !strings.Contains(body, "bad-page") {
				t.Errorf("%s: %d %q", bad, code, body)
			}
		}
		if forms, _ := a.calls(); forms != 0 {
			t.Errorf("a code with a bad state was exchanged")
		}
		callbackGet(t, q, "code=real&state="+q.Get("state"))
	})
	if err != nil {
		t.Fatal(err)
	}
	if a.forms[0].Get("code") != "real" {
		t.Errorf("exchanged %v", a.forms)
	}
}

func TestLoginStateOnlyOnce(t *testing.T) {
	a := newAuthServer(t)
	a.reply = func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(50 * time.Millisecond)
		grant(w, accessJWT(time.Now().Add(time.Hour)), fakeRefresh, "me@example.com")
	}
	s := session(t, a, &memStore{})
	if _, err := loginVia(t, s, func(q url.Values) {
		var wg sync.WaitGroup
		for range 2 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				callbackGet(t, q, "code=c&state="+q.Get("state"))
			}()
		}
		wg.Wait()
	}); err != nil {
		t.Fatal(err)
	}
	if forms, _ := a.calls(); forms != 1 {
		t.Errorf("%d exchanges", forms)
	}
}

func TestLoginDenied(t *testing.T) {
	a := newAuthServer(t)
	a.reply = func(http.ResponseWriter, *http.Request) { t.Error("exchanged") }
	var red string
	s := session(t, a, &memStore{})
	_, err := loginVia(t, s, func(q url.Values) {
		red = q.Get("redirect_uri")
		callbackGet(t, q, "error=access_denied&state="+q.Get("state"))
	})
	var e *Error
	if !errors.As(err, &e) || e.Kind != KindDenied || e.Code != "access_denied" {
		t.Fatalf("err %v", err)
	}
	assertClosed(t, red)
}

func TestLoginPortInUse(t *testing.T) {
	a := newAuthServer(t)
	hold := func() (*net.TCPListener, int) {
		l, err := net.ListenTCP("tcp4", &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)})
		if err != nil {
			t.Fatal(err)
		}
		return l, l.Addr().(*net.TCPAddr).Port
	}
	l1, p1 := hold()
	defer l1.Close()
	l2, p2 := hold()
	defer l2.Close()

	s := session(t, a, &memStore{}, func(c *Config) { c.Ports = []int{p1, p2} })
	_, err := s.Login(context.Background(), func(string) error { t.Error("opened the browser"); return nil })
	var e *Error
	if !errors.As(err, &e) || e.Kind != KindPortInUse || e.Port != p2 {
		t.Fatalf("all taken: %v", err)
	}

	free := freePort(t)
	s = session(t, a, &memStore{}, func(c *Config) { c.Ports = []int{p1, free}; c.LoginTimeout = 100 * time.Millisecond })
	_, err = s.Login(context.Background(), func(u string) error {
		if !strings.Contains(u, fmt.Sprintf("127.0.0.1%%3A%d", free)) {
			t.Errorf("redirect not on the free port: %s", u)
		}
		return nil
	})
	if !errors.As(err, &e) || e.Kind != KindTimeout {
		t.Fatalf("second port: %v", err)
	}
}

func TestLoginBindsLoopbackOnly(t *testing.T) {
	addrs, _ := net.InterfaceAddrs()
	var other string
	for _, ad := range addrs {
		if n, ok := ad.(*net.IPNet); ok && n.IP.To4() != nil && !n.IP.IsLoopback() {
			other = n.IP.String()
		}
	}
	if other == "" {
		t.Skip("no address but the loopback")
	}
	a := newAuthServer(t)
	s := session(t, a, &memStore{}, func(c *Config) { c.LoginTimeout = 150 * time.Millisecond })
	_, _ = s.Login(context.Background(), func(u string) error {
		q := parseAuthorize(t, u)
		ru, _ := url.Parse(q.Get("redirect_uri"))
		if c, err := net.DialTimeout("tcp4", net.JoinHostPort(other, ru.Port()), 100*time.Millisecond); err == nil {
			c.Close()
			t.Errorf("the listener answers on %s", other)
		}
		return nil
	})
}

func TestLoginTimeoutAndCancel(t *testing.T) {
	a := newAuthServer(t)
	s := session(t, a, &memStore{}, func(c *Config) { c.LoginTimeout = 80 * time.Millisecond })
	var red string
	_, err := s.Login(context.Background(), func(u string) error { red = parseAuthorize(t, u).Get("redirect_uri"); return nil })
	var e *Error
	if !errors.As(err, &e) || e.Kind != KindTimeout {
		t.Fatalf("timeout: %v", err)
	}
	assertClosed(t, red)

	s = session(t, a, &memStore{})
	ctx, cancel := context.WithCancel(context.Background())
	_, err = s.Login(ctx, func(u string) error { red = parseAuthorize(t, u).Get("redirect_uri"); cancel(); return nil })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
	assertClosed(t, red)
}

// Every failure of an exchange or a refresh that echoes secrets stays
// out of the error, whatever the service answers.
func TestNoSecretInErrors(t *testing.T) {
	secrets := []string{"the-code", "the-verifier-echo", fakeRefresh, "AT-secret", "ID-secret"}
	echo := strings.Join(secrets, " ")
	for _, tc := range []struct {
		name  string
		reply func(http.ResponseWriter, *http.Request)
	}{
		{"oauth error", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(400)
			fmt.Fprintf(w, `{"error":"invalid_grant","error_description":"bad %s"}`, echo)
		}},
		{"error is free text", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(400)
			fmt.Fprintf(w, `{"error":"bad %s"}`, echo)
		}},
		{"html 500", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(500)
			fmt.Fprintf(w, `<html>%s</html>`, echo)
		}},
		{"no tokens", func(w http.ResponseWriter, r *http.Request) { fmt.Fprintf(w, `{"note":%q}`, echo) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := newAuthServer(t)
			a.reply = tc.reply
			s := session(t, a, &memStore{})
			_, loginErr := loginVia(t, s, func(q url.Values) { callbackGet(t, q, "code=the-code&state="+q.Get("state")) })
			if loginErr == nil {
				t.Fatal("no error")
			}
			st, _ := stored{Refresh: fakeRefresh}.encode()
			s = session(t, a, &memStore{v: st})
			_, refreshErr := s.Token(context.Background(), "")
			if refreshErr == nil {
				t.Fatal("no refresh error")
			}
			for _, err := range []error{loginErr, refreshErr} {
				for _, sec := range secrets {
					if strings.Contains(err.Error(), sec) || strings.Contains(fmt.Sprintf("%+v %v", err, err), sec) {
						t.Errorf("%v holds %q", err, sec)
					}
				}
			}
		})
	}
}

func signedIn(t *testing.T, a *authServer, store *memStore, mut ...func(*Config)) *Session {
	v, err := stored{Refresh: fakeRefresh, AccountID: "acct-1", Email: "me@example.com"}.encode()
	if err != nil {
		t.Fatal(err)
	}
	store.v = v
	return session(t, a, store, mut...)
}

// Twenty requests that find the token expired make one refresh and all
// get its result; the rotated refresh token reaches the Keychain.
func TestConcurrentRefreshIsOne(t *testing.T) {
	a := newAuthServer(t)
	access := accessJWT(time.Now().Add(time.Hour))
	a.reply = func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(30 * time.Millisecond)
		grant(w, access, fakeRefresh2, "me@example.com")
	}
	store := &memStore{}
	s := signedIn(t, a, store)
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if c, err := s.Token(context.Background(), ""); err != nil || c.Access != access {
				t.Errorf("Token %+v %v", c, err)
			}
		}()
	}
	wg.Wait()
	if _, n := a.calls(); n != 1 {
		t.Errorf("%d refreshes", n)
	}
	if m := a.json[0]; m["grant_type"] != "refresh_token" || m["refresh_token"] != fakeRefresh || m["client_id"] != ClientID {
		t.Errorf("refresh body %v", m)
	}
	if st, _ := decode(store.value()); st.Refresh != fakeRefresh2 {
		t.Errorf("Keychain has %+v", st)
	}
}

func TestRefreshOutlivesItsCaller(t *testing.T) {
	a := newAuthServer(t)
	release := make(chan struct{})
	a.reply = func(w http.ResponseWriter, r *http.Request) {
		<-release
		grant(w, accessJWT(time.Now().Add(time.Hour)), fakeRefresh2, "me@example.com")
	}
	store := &memStore{}
	s := signedIn(t, a, store)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, err := s.Token(ctx, ""); done <- err }()
	for {
		if _, n := a.calls(); n == 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("caller: %v", err)
	}
	close(release)
	if _, err := s.Token(context.Background(), ""); err != nil {
		t.Fatal(err)
	}
	if st, _ := decode(store.value()); st.Refresh != fakeRefresh2 {
		t.Errorf("Keychain has %+v", st)
	}
	if _, n := a.calls(); n != 1 {
		t.Errorf("%d refreshes", n)
	}
}

// A refresh token the service refuses signs out: once, asking to sign in
// again, and the next use does not try again.
func TestRevokedRefreshSignsOut(t *testing.T) {
	for _, tc := range []struct {
		status int
		body   string
	}{
		{400, `{"error":"invalid_grant"}`},
		{401, `{"error":{"code":"refresh_token_reused"}}`},
	} {
		a := newAuthServer(t)
		a.reply = func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(tc.status); fmt.Fprint(w, tc.body) }
		store := &memStore{}
		fired := 0
		s := signedIn(t, a, store, func(c *Config) { c.SignedOut = func() { fired++ } })
		_, err := s.Token(context.Background(), "")
		if !errors.Is(err, ErrSignedOut) {
			t.Fatalf("%d: %v", tc.status, err)
		}
		if store.value() != "" || fired != 1 {
			t.Errorf("Keychain %q, SignedOut called %d times", store.value(), fired)
		}
		if _, err := s.Token(context.Background(), ""); !errors.Is(err, ErrSignedOut) {
			t.Errorf("next: %v", err)
		}
		if _, n := a.calls(); n != 1 {
			t.Errorf("%d refreshes after a dead sign-in", n)
		}
	}
}

// A service or network failure keeps the sign-in: the next use tries.
func TestTransientRefreshKeepsSignIn(t *testing.T) {
	for _, status := range []int{500, 503, 403} {
		a := newAuthServer(t)
		a.reply = func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(status) }
		store := &memStore{}
		s := signedIn(t, a, store)
		before := store.value()
		_, err := s.Token(context.Background(), "")
		var e *Error
		if !errors.As(err, &e) || e.Kind != KindNetwork || e.Status != status {
			t.Fatalf("%d: %v", status, err)
		}
		if store.value() != before {
			t.Errorf("%d: the sign-in was changed", status)
		}
		s.Token(context.Background(), "")
		if _, n := a.calls(); n != 2 {
			t.Errorf("%d: %d refreshes", status, n)
		}
	}
}

func TestExpiryAndStale(t *testing.T) {
	a := newAuthServer(t)
	now := time.Unix(1_800_000_000, 0)
	short, long := accessJWT(now.Add(30*time.Second)), accessJWT(now.Add(time.Hour))
	var next string
	a.reply = func(w http.ResponseWriter, r *http.Request) { grant(w, next, fakeRefresh2, "me@example.com") }
	s := signedIn(t, a, &memStore{}, func(c *Config) { c.Now = func() time.Time { return now } })

	next = long
	if c, _ := s.Token(context.Background(), ""); c.Access != long {
		t.Fatalf("first %q", c.Access)
	}
	// Held, not expired: no refresh, and a stale token that is not the
	// current one is not a reason for one.
	if _, n := a.calls(); n != 1 {
		t.Fatal(n)
	}
	s.Token(context.Background(), "some-older-token")
	if _, n := a.calls(); n != 1 {
		t.Errorf("refreshed for an older token: %d", n)
	}
	// The current token refused: a refresh.
	next = short
	c, _ := s.Token(context.Background(), long)
	if c.Access != short {
		t.Errorf("after stale: %q", c.Access)
	}
	// 30 s left is under the margin.
	next = long
	if c, _ := s.Token(context.Background(), ""); c.Access != long {
		t.Errorf("near expiry: %q", c.Access)
	}
}

func TestExpiresFromResponse(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	if got := (tokens{Access: "not-a-jwt", ExpiresIn: 100}).expires(now); !got.Equal(now.Add(100 * time.Second)) {
		t.Errorf("expires_in: %v", got)
	}
	if got := (tokens{Access: "not-a-jwt"}).expires(now); !got.IsZero() {
		t.Errorf("unknown: %v", got)
	}
}

func TestSignOut(t *testing.T) {
	a := newAuthServer(t)
	a.reply = func(http.ResponseWriter, *http.Request) { t.Error("network") }
	store := &memStore{}
	s := signedIn(t, a, store)
	if err := s.SignOut(); err != nil {
		t.Fatal(err)
	}
	if store.value() != "" {
		t.Errorf("Keychain %q", store.value())
	}
	if _, err := s.Token(context.Background(), ""); !errors.Is(err, ErrSignedOut) {
		t.Errorf("after: %v", err)
	}
	if f, r := a.calls(); f+r != 0 {
		t.Error("sign-out touched the network")
	}
}

// A sign-out during a refresh is not undone when the refresh returns.
func TestSignOutDuringRefresh(t *testing.T) {
	a := newAuthServer(t)
	release := make(chan struct{})
	a.reply = func(w http.ResponseWriter, r *http.Request) {
		<-release
		grant(w, accessJWT(time.Now().Add(time.Hour)), fakeRefresh2, "me@example.com")
	}
	store := &memStore{}
	s := signedIn(t, a, store)
	done := make(chan error, 1)
	go func() { _, err := s.Token(context.Background(), ""); done <- err }()
	for {
		if _, n := a.calls(); n == 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if err := s.SignOut(); err != nil {
		t.Fatal(err)
	}
	close(release)
	if err := <-done; !errors.Is(err, ErrSignedOut) {
		t.Errorf("Token: %v", err)
	}
	if store.value() != "" {
		t.Errorf("the refresh put the sign-in back: %q", store.value())
	}
}

func TestStoreFailureIsReported(t *testing.T) {
	a := newAuthServer(t)
	a.reply = func(w http.ResponseWriter, r *http.Request) {
		grant(w, accessJWT(time.Now().Add(time.Hour)), fakeRefresh2, "me@example.com")
	}
	store := &memStore{setErr: errors.New("keychain says no")}
	s := signedIn(t, a, store)
	_, err := s.Token(context.Background(), "")
	var e *Error
	if !errors.As(err, &e) || e.Kind != KindStore || strings.Contains(err.Error(), "keychain says") {
		t.Fatalf("err %v", err)
	}
	// The new tokens are kept for this run.
	if c, err := s.Token(context.Background(), ""); err != nil || c.Access == "" {
		t.Errorf("after: %+v %v", c, err)
	}
}

func TestStoredFitsTheKeychainLine(t *testing.T) {
	if _, err := (stored{Refresh: strings.Repeat("r", maxStored)}).encode(); err == nil {
		t.Error("too long a refresh token was accepted")
	}
	v, err := stored{Refresh: fakeRefresh, Email: "a@b"}.encode()
	if err != nil || strings.ContainsAny(v, "\"\\") {
		t.Errorf("%q %v", v, err)
	}
}

func TestLoadDamagedItem(t *testing.T) {
	a := newAuthServer(t)
	a.reply = func(http.ResponseWriter, *http.Request) { t.Error("network") }
	s := session(t, a, &memStore{v: "not base64!"})
	if _, err := s.Token(context.Background(), ""); !errors.Is(err, ErrSignedOut) {
		t.Errorf("err %v", err)
	}
	if Email("not base64!") != "" {
		t.Error("email from a damaged item")
	}
}

// The real service's addresses and the borrowed client, which the fakes
// above replace.
func TestOpenAIAddresses(t *testing.T) {
	if OpenAI.Authorize != "https://auth.openai.com/oauth/authorize" || OpenAI.Token != "https://auth.openai.com/oauth/token" {
		t.Errorf("endpoints %+v", OpenAI)
	}
	if ClientID != "app_EMoamEEZ73f0CkXaXp7hrann" {
		t.Errorf("client %q", ClientID)
	}
	if len(callbackPorts) != 2 || callbackPorts[0] != 1455 || callbackPorts[1] != 1457 {
		t.Errorf("ports %v", callbackPorts)
	}
}
