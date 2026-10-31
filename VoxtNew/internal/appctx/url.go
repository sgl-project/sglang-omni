package appctx

import (
	"errors"
	"net"
	"net/url"
	"regexp"
	"strings"
)

// Address is a tab's address as rules see it: host and path, lower case,
// without the scheme, "www.", the port, the query and the fragment
// ("github.com/hehehai/voxtnew").
type Address struct {
	Host string // "github.com"
	Path string // "/hehehai/voxtnew" ("/" at least)
}

// String is host + path.
func (a Address) String() string { return a.Host + a.Path }

// ParseAddress reads a tab's URL. Only addresses with a host are
// addresses (http, https, and other schemes with a host such as
// chrome://settings); "about:blank", file:// paths and junk are not.
func ParseAddress(raw string) (Address, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" || strings.ContainsAny(raw, " \t\n") {
		return Address{}, false
	}
	if !strings.Contains(raw, "://") {
		if m := schemeOnly.FindStringSubmatch(raw); m != nil && !isDigits(m[1]) {
			return Address{}, false // "about:blank", "mailto:x" (but "localhost:3000")
		}
		raw = "https://" + raw
	}
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return Address{}, false
	}
	host := strings.ToLower(u.Hostname())
	host = strings.TrimSuffix(host, ".")
	host = strings.TrimPrefix(host, "www.")
	if host == "" {
		return Address{}, false
	}
	path := strings.ToLower(u.Path)
	if path == "" {
		path = "/"
	}
	return Address{Host: host, Path: path}, true
}

// schemeOnly finds a scheme without "//": what follows its colon, up to
// a slash.
var schemeOnly = regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9+.-]*:([^/]*)`)

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// CanonicalPattern is a URL rule as it is matched: trimmed, lower case,
// without a scheme, "www." or a port; a rule without a path is the whole
// site ("github.com" is "github.com/*"), and one ending in "/" takes what
// follows ("github.com/hehehai/" is "github.com/hehehai/*").
func CanonicalPattern(p string) string {
	p = strings.ToLower(strings.TrimSpace(p))
	if i := strings.Index(p, "://"); i >= 0 {
		p = p[i+3:]
	}
	host, path, hasPath := strings.Cut(p, "/")
	if h, _, err := net.SplitHostPort(host); err == nil {
		host = h
	}
	host = strings.TrimSuffix(host, ".")
	host = strings.TrimPrefix(host, "www.")
	switch {
	case !hasPath:
		path = "/*"
	case path == "" || strings.HasSuffix(path, "/"):
		path = "/" + path + "*"
	default:
		path = "/" + path
	}
	// The query and the fragment are not matched.
	if i := strings.IndexAny(path, "?#"); i >= 0 {
		path = path[:i]
		if !strings.HasSuffix(path, "*") {
			path += "*"
		}
	}
	return host + path
}

// Errors of CheckPattern.
var (
	ErrPatternEmpty  = errors.New("appctx: the rule is empty")
	ErrPatternSpace  = errors.New("appctx: the rule has a space")
	ErrPatternHost   = errors.New("appctx: the rule needs a site, as in example.com")
	ErrPatternScheme = errors.New("appctx: only web addresses (http, https)")
)

// CheckPattern says whether p is a usable rule: a host (with "*" for any
// part, "*.example.com"), then a path that may use "*".
func CheckPattern(p string) error {
	raw := strings.TrimSpace(p)
	if raw == "" {
		return ErrPatternEmpty
	}
	if strings.ContainsAny(raw, " \t\n") {
		return ErrPatternSpace
	}
	if i := strings.Index(raw, "://"); i >= 0 {
		switch strings.ToLower(raw[:i]) {
		case "http", "https", "*":
		default:
			return ErrPatternScheme
		}
	}
	c := CanonicalPattern(raw)
	host, _, _ := strings.Cut(c, "/")
	if host == "" || host == "*" {
		return ErrPatternHost
	}
	for _, r := range host {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '.', r == '-', r == '*', r == '_':
		case r > 127: // an internationalised name
		default:
			return ErrPatternHost
		}
	}
	if !strings.Contains(host, ".") && host != "localhost" {
		return ErrPatternHost
	}
	if strings.HasPrefix(host, ".") || strings.HasSuffix(host, ".") || strings.Contains(host, "..") {
		return ErrPatternHost
	}
	return nil
}

// MatchPattern reports whether the rule p matches the address a. "*"
// stands for any run of characters (also "/"); a host starting "*." also
// matches the domain itself ("*.notion.so" matches notion.so).
func MatchPattern(p string, a Address) bool {
	if CheckPattern(p) != nil || a.Host == "" {
		return false
	}
	c := CanonicalPattern(p)
	s := a.String()
	if glob(c, s) {
		return true
	}
	if strings.HasPrefix(c, "*.") && glob(c[2:], s) {
		return true
	}
	return false
}

// specificity ranks a rule: the characters it fixes (all but "*"); a
// longer literal rule is more specific.
func specificity(p string) int {
	return len(strings.ReplaceAll(CanonicalPattern(p), "*", ""))
}

// glob matches s against pattern, where "*" is any run of characters.
func glob(pattern, s string) bool {
	// Iterative matching with backtracking to the last star.
	p, i := 0, 0
	star, mark := -1, 0
	for i < len(s) {
		switch {
		case p < len(pattern) && pattern[p] == '*':
			star, mark = p, i
			p++
		case p < len(pattern) && pattern[p] == s[i]:
			p++
			i++
		case star >= 0:
			p = star + 1
			mark++
			i = mark
		default:
			return false
		}
	}
	for p < len(pattern) && pattern[p] == '*' {
		p++
	}
	return p == len(pattern)
}
