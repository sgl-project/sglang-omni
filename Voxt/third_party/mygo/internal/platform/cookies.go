package platform

import (
	"net"
	"net/url"
	"strings"
	"time"
)

// Cookie is a normalized snapshot of a cookie in the default page store.
// SameSite is lax, strict, none, or empty (unknown on output only).
// A zero Expires denotes a session cookie.
type Cookie struct {
	Name, Value, Domain, Path string
	Expires                   time.Time
	Secure, HTTPOnly          bool
	SameSite                  string
}

// CookieKey identifies a cookie without request URL matching.
type CookieKey struct{ Name, Domain, Path string }

// CookieMatchesURL reports whether a request to u sends c, by the rules
// of RFC 6265 section 5.4: its domain, path and Secure attribute.
func CookieMatchesURL(c Cookie, u *url.URL) bool {
	host := strings.ToLower(u.Hostname())
	domain := strings.ToLower(c.Domain)
	if d, ok := strings.CutPrefix(domain, "."); ok {
		if host != d && (!strings.HasSuffix(host, "."+d) || net.ParseIP(host) != nil) {
			return false
		}
	} else if host != domain {
		return false
	}
	path := u.EscapedPath()
	if !strings.HasPrefix(path, "/") {
		path = "/"
	}
	p := c.Path
	if p == "" {
		p = "/"
	}
	if path != p && (!strings.HasPrefix(path, p) || !strings.HasSuffix(p, "/") && path[len(p)] != '/') {
		return false
	}
	return !c.Secure || strings.EqualFold(u.Scheme, "https")
}
