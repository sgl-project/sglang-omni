package models

import (
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/hfhub"
)

// testProxy is a systemProxy with no environment's proxy, the given
// system's, a clock the test moves, and a log.
func testProxy(system *SystemProxies, reads *int, now *time.Time, logged *[]string) *systemProxy {
	return &systemProxy{
		env:  func(*http.Request) (*url.URL, error) { return nil, nil },
		read: func() SystemProxies { *reads++; return *system },
		logf: func(format string, args ...any) { *logged = append(*logged, fmt.Sprintf(format, args...)) },
		now:  func() time.Time { return *now },
	}
}

func proxyFor(t *testing.T, s *systemProxy, rawURL string) string {
	t.Helper()
	request, _ := http.NewRequest("GET", rawURL, nil)
	proxyURL, err := s.proxy(request)
	if err != nil {
		t.Fatal(err)
	}
	if proxyURL == nil {
		return ""
	}
	return proxyURL.String()
}

// TestSystemProxy: the scheme's proxy, the exceptions, nothing set, the
// cache of the settings and the log of a change.
func TestSystemProxy(t *testing.T) {
	system := SystemProxies{HTTP: "127.0.0.1:8080", HTTPS: "127.0.0.1:7897",
		Exceptions: []string{"*.local", "169.254/16", "intranet.example", " 10.0.0.0/8 "}, ExcludeSimple: true}
	var reads int
	var logged []string
	now := time.Unix(1000, 0)
	s := testProxy(&system, &reads, &now, &logged)
	for rawURL, want := range map[string]string{
		"https://huggingface.co/org/m/resolve/main/a.json": "http://127.0.0.1:7897",
		"http://huggingface.co/a":                          "http://127.0.0.1:8080",
		"https://printer.local/":                           "",
		"https://169.254.3.4/":                             "",
		"https://10.1.2.3:8443/":                           "",
		"https://11.1.2.3/":                                "http://127.0.0.1:7897",
		"https://Intranet.Example/":                        "",
		"https://intranet/":                                "",
		"https://localhost:9/":                             "",
		"https://127.0.0.1/":                               "",
		"https://[::1]/":                                   "",
	} {
		if got := proxyFor(t, s, rawURL); got != want {
			t.Errorf("%s: proxy %q, want %q", rawURL, got, want)
		}
	}
	if reads != 1 {
		t.Errorf("read %d times in a burst", reads)
	}
	if want := []string{"models: system proxy http 127.0.0.1:8080", "models: system proxy https 127.0.0.1:7897"}; strings.Join(slices.Sorted(slices.Values(logged)), "|") != strings.Join(want, "|") {
		t.Errorf("log %q", logged)
	}

	// The user turns the proxies off: seen after the cache's time.
	system = SystemProxies{}
	now = now.Add(systemProxyTTL - time.Millisecond)
	if got := proxyFor(t, s, "https://huggingface.co/"); got != "http://127.0.0.1:7897" {
		t.Errorf("within the cache: %q", got)
	}
	now = now.Add(time.Millisecond)
	if got := proxyFor(t, s, "https://huggingface.co/"); got != "" || reads != 2 {
		t.Errorf("after the cache: %q, %d reads", got, reads)
	}
	if last := logged[len(logged)-1]; last != "models: system proxy https none" {
		t.Errorf("log %q", logged)
	}

	// Never a proxy: nothing logged.
	var quiet []string
	none := SystemProxies{}
	if got := proxyFor(t, testProxy(&none, &reads, &now, &quiet), "https://huggingface.co/"); got != "" || len(quiet) != 0 {
		t.Errorf("no proxy: %q, log %q", got, quiet)
	}

	// mac-air's list: "<local>" stands for plain names.
	local := SystemProxies{HTTPS: "127.0.0.1:7897", Exceptions: []string{"192.168.0.0/16", "<local>"}}
	for host, want := range map[string]bool{"intranet": true, "192.168.1.20": true, "huggingface.co": false} {
		if got := bypassProxy(host, local); got != want {
			t.Errorf("%s: bypass %v", host, got)
		}
	}
}

// TestSystemProxyEnvironmentWins: a developer's https_proxy is used
// before the system's, which is not even read.
func TestSystemProxyEnvironmentWins(t *testing.T) {
	system := SystemProxies{HTTPS: "127.0.0.1:7897"}
	var reads int
	var logged []string
	now := time.Unix(1000, 0)
	s := testProxy(&system, &reads, &now, &logged)
	envProxy, _ := url.Parse("http://proxy.dev:3128")
	s.env = func(*http.Request) (*url.URL, error) { return envProxy, nil }
	if got := proxyFor(t, s, "https://huggingface.co/"); got != "http://proxy.dev:3128" || reads != 0 || len(logged) != 0 {
		t.Errorf("proxy %q, %d reads, log %q", got, reads, logged)
	}
}

// hubProxy is an HTTP proxy in front of the fake Hub at hubURL, for a Hub
// of a name of its own (hub.example). With user ("name:password") it is a
// proxy that wants them: a request without them gets 407. It returns the
// requests it got, each with its Proxy-Authorization.
func hubProxy(t *testing.T, hubURL, user string) (*httptest.Server, func() []string) {
	var mu sync.Mutex
	var proxied []string
	target, _ := url.Parse(hubURL)
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Proxy-Authorization")
		mu.Lock()
		proxied = append(proxied, r.RequestURI+" "+auth)
		mu.Unlock()
		switch {
		case r.URL.Host != "hub.example":
			http.Error(w, "not proxied: "+r.RequestURI, http.StatusBadGateway)
			return
		case user != "" && auth != "Basic "+base64.StdEncoding.EncodeToString([]byte(user)):
			w.Header().Set("Proxy-Authenticate", `Basic realm="proxy"`)
			http.Error(w, "proxy authentication required", http.StatusProxyAuthRequired)
			return
		}
		out, _ := http.NewRequestWithContext(r.Context(), r.Method, target.String()+r.URL.RequestURI(), nil)
		out.Header = r.Header.Clone()
		out.Header.Del("Proxy-Authorization")
		res, err := http.DefaultTransport.RoundTrip(out)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer res.Body.Close()
		for k, v := range res.Header {
			w.Header()[k] = v
		}
		w.WriteHeader(res.StatusCode)
		io.Copy(w, res.Body)
	}))
	t.Cleanup(proxy.Close)
	return proxy, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(proxied)
	}
}

// TestDownloadThroughProxy: with the automatic proxy, a Manager fetches
// every file through the system's proxy (a proxy in front of the fake
// Hub); "direct" never uses it.
func TestDownloadThroughProxy(t *testing.T) {
	e := newEnv(t, testCatalog(), Options{})
	proxy, requests := hubProxy(t, e.hub.URL, "")
	var logged []string
	read := func() SystemProxies { return SystemProxies{HTTP: strings.TrimPrefix(proxy.URL, "http://")} }
	s := &systemProxy{env: func(*http.Request) (*url.URL, error) { return nil, nil }, read: read, now: time.Now,
		logf: func(format string, args ...any) { logged = append(logged, fmt.Sprintf(format, args...)) }}
	if ProxyFor(hfhub.ProxyDirect, nil, s.proxy) == nil {
		t.Fatal("no rule for direct")
	}
	e.m.SetNetwork(Network{Endpoints: []string{"http://hub.example"}, Client: NewHTTPClient(ProxyFor("", nil, s.proxy))})
	if err := e.m.Download("llm"); err != nil {
		t.Fatal(err)
	}
	if st := wait(t, e.m, "llm"); !st.Installed || st.Err != nil {
		t.Fatalf("%+v", st)
	}
	if proxied := requests(); len(proxied) < 4 || !strings.HasPrefix(proxied[0], "http://hub.example/") {
		t.Errorf("proxied %q", proxied)
	}
	if len(logged) != 1 || !strings.Contains(logged[0], "system proxy http 127.0.0.1:") {
		t.Errorf("log %q", logged)
	}
}

// TestDownloadThroughAuthProxy: a custom proxy that wants a user name and
// password, as Voxt 1.x's could: without them it answers 407 and the
// download fails, with an error that names no password; with them every
// request of the download carries them and it installs.
func TestDownloadThroughAuthProxy(t *testing.T) {
	const password = "Pr0xy-pass-5521"
	e := newEnv(t, testCatalog(), Options{})
	proxy, requests := hubProxy(t, e.hub.URL, "alice:"+password)
	network := func(user *url.Userinfo) Network {
		return Network{Endpoints: []string{"http://hub.example"}, Client: NewHTTPClient(ProxyFor(proxy.URL, user, nil))}
	}
	e.m.SetNetwork(network(nil))
	if err := e.m.Download("llm"); err != nil {
		t.Fatal(err)
	}
	if st := wait(t, e.m, "llm"); st.Installed || st.Err == nil || strings.Contains(st.Err.Error(), password) {
		t.Fatalf("without the user: %+v", st)
	}
	refused := len(requests())
	e.m.SetNetwork(network(url.UserPassword("alice", password)))
	if err := e.m.Download("llm"); err != nil {
		t.Fatal(err)
	}
	if st := wait(t, e.m, "llm"); !st.Installed || st.Err != nil {
		t.Fatalf("with the user: %+v", st)
	}
	want := " Basic " + base64.StdEncoding.EncodeToString([]byte("alice:"+password))
	proxied := requests()[refused:]
	if len(proxied) < 4 || slices.ContainsFunc(proxied, func(r string) bool { return !strings.HasSuffix(r, want) }) {
		t.Errorf("proxied %q", proxied)
	}
}
