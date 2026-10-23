package models

// The downloads' proxy: a GUI app gets no https_proxy, so on a network
// that only reaches the Hub through a proxy, the system's (macOS's
// settings) is what the user set. It is no setting of VoxtNew's.

import (
	"cmp"
	"net"
	"net/http"
	"net/url"
	"path"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/hehehai/actnowkit/hfhub"
)

// SystemProxies are the system's web proxies (macx.Proxies, the same
// fields).
type SystemProxies struct {
	// HTTP and HTTPS are the proxies' "host:port", "" when off.
	HTTP, HTTPS string
	// Exceptions are the hosts, globs ("*.local") and address ranges
	// ("169.254/16") that bypass them.
	Exceptions []string
	// ExcludeSimple: plain host names (no dot) bypass them.
	ExcludeSimple bool
}

// systemProxyTTL is how long the system's proxies are kept before they
// are read again: they may change while the app runs.
const systemProxyTTL = 5 * time.Second

// SystemProxy is a Transport's Proxy: the environment's (https_proxy,
// http_proxy: a developer's shell) when it names one, else the system's
// for the request's scheme, unless the host bypasses it. read reads the
// system's proxies, at most every 5 s; logf tells which one is used, when
// that changes.
func SystemProxy(read func() SystemProxies, logf func(format string, args ...any)) func(*http.Request) (*url.URL, error) {
	return (&systemProxy{env: http.ProxyFromEnvironment, read: read, logf: logf, now: time.Now}).proxy
}

type systemProxy struct {
	env  func(*http.Request) (*url.URL, error)
	read func() SystemProxies
	logf func(format string, args ...any)
	now  func() time.Time

	mu     sync.Mutex
	readAt time.Time
	known  SystemProxies
	logged map[string]string // the proxy last logged, by scheme
}

func (s *systemProxy) proxy(request *http.Request) (*url.URL, error) {
	if proxyURL, err := s.env(request); proxyURL != nil || err != nil {
		return proxyURL, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if now := s.now(); s.readAt.IsZero() || now.Sub(s.readAt) >= systemProxyTTL {
		s.known, s.readAt = s.read(), now
	}
	scheme := request.URL.Scheme
	hostPort := s.known.HTTP
	if scheme == "https" {
		hostPort = s.known.HTTPS
	}
	if s.logged == nil {
		s.logged = map[string]string{}
	}
	if last, seen := s.logged[scheme]; hostPort != last && (seen || hostPort != "") {
		s.logged[scheme] = hostPort
		s.logf("models: system proxy %s %s", scheme, cmp.Or(hostPort, "none"))
	}
	if hostPort == "" || bypassProxy(request.URL.Hostname(), s.known) {
		return nil, nil
	}
	return &url.URL{Scheme: "http", Host: hostPort}, nil
}

// bypassProxy reports whether host goes direct: the loopback, plain names
// when ExcludeSimple or "<local>" is an exception, and the exceptions
// (hosts, globs, address ranges whose missing octets are 0, as macOS
// writes "169.254/16").
func bypassProxy(host string, proxies SystemProxies) bool {
	host = strings.ToLower(strings.TrimSuffix(host, "."))
	ip := net.ParseIP(host)
	if host == "localhost" || ip != nil && ip.IsLoopback() {
		return true
	}
	simple := ip == nil && !strings.Contains(host, ".")
	if simple && (proxies.ExcludeSimple || slices.Contains(proxies.Exceptions, "<local>")) {
		return true
	}
	for _, exception := range proxies.Exceptions {
		exception = strings.ToLower(strings.TrimSpace(exception))
		if prefix, bits, ok := strings.Cut(exception, "/"); ok {
			for strings.Count(prefix, ".") < 3 && !strings.Contains(prefix, ":") {
				prefix += ".0"
			}
			if _, network, err := net.ParseCIDR(prefix + "/" + bits); err == nil && ip != nil && network.Contains(ip) {
				return true
			}
		} else if matched, _ := path.Match(exception, host); matched {
			return true
		}
	}
	return false
}

// ProxyFor is the downloads' proxy rule for the family's network.proxy
// (family spec §9.1): "" automatic, system (the environment's, then the
// Mac's: SystemProxy); "direct", or a proxy URL, as hfhub reads them. The
// loopback is never proxied.
func ProxyFor(mode string, system func(*http.Request) (*url.URL, error)) func(*http.Request) (*url.URL, error) {
	if strings.TrimSpace(mode) == "" && system != nil {
		return system
	}
	return hfhub.ProxyFunc(mode)
}

// NewHTTPClient is the downloads' HTTP client through proxy, with hfhub's
// timeouts (30 s for the response headers).
func NewHTTPClient(proxy func(*http.Request) (*url.URL, error)) *http.Client {
	c := hfhub.NewHTTPClient(hfhub.ProxyDirect, 30*time.Second)
	if t, ok := c.Transport.(*http.Transport); ok {
		t.Proxy = proxy
	}
	return c
}
