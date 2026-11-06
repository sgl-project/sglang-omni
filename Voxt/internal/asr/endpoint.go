package asr

import (
	"net"
	"net/url"
	"slices"
	"strings"
)

// CheckEndpoint parses a provider endpoint and checks it is safe to send a
// key to: an absolute URL whose scheme is one of secure (https, wss), or
// its plain twin (http, ws) when the host is loopback (a local relay or a
// test server); no user or password in it. The returned URL keeps its
// query (providers may need one), but SanitizeEndpoint drops it for logs.
//
//	CheckEndpoint("wss://openspeech.bytedance.com/api/v3/sauc/bigmodel", "wss")
func CheckEndpoint(provider, raw string, secure ...string) (*url.URL, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, Errorf(provider, KindConfig, "no endpoint")
	}
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" || u.Scheme == "" {
		return nil, Errorf(provider, KindConfig, "endpoint %q is not an absolute URL", SanitizeEndpoint(raw))
	}
	u.Scheme = strings.ToLower(u.Scheme)
	if u.User != nil {
		return nil, Errorf(provider, KindConfig, "endpoint must not contain a user or password")
	}
	plain := map[string]string{"https": "http", "wss": "ws"}
	ok := slices.Contains(secure, u.Scheme)
	if !ok {
		for _, s := range secure {
			if plain[s] == u.Scheme && IsLoopback(u.Hostname()) {
				ok = true
			}
		}
	}
	if !ok {
		return nil, Errorf(provider, KindConfig, "endpoint must use %s (plain %s only on this Mac): %s",
			strings.Join(secure, " or "), u.Scheme, SanitizeEndpoint(raw))
	}
	return u, nil
}

// IsLoopback reports whether host is this machine: localhost or a
// loopback address.
func IsLoopback(host string) bool {
	host = strings.Trim(strings.ToLower(host), "[]")
	if host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// SanitizeEndpoint is an endpoint for the log: no user, query or fragment
// (where keys sometimes hide).
func SanitizeEndpoint(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Host == "" {
		return "<invalid endpoint>"
	}
	u.User, u.RawQuery, u.Fragment = nil, "", ""
	return u.String()
}
