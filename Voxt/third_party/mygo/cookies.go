package mygo

import (
	"fmt"
	"net/netip"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo/internal/platform"
)

// CookieSameSite says whether a cookie goes along with requests from
// other sites. Cookies.Set takes an empty one as Lax; a cookie List
// returns has an empty one when the engine does not say.
type CookieSameSite string

const (
	// CookieSameSiteLax sends the cookie along when the user navigates
	// from another site, but not with other requests from it.
	CookieSameSiteLax CookieSameSite = "lax"
	// CookieSameSiteStrict sends the cookie only with the site's own
	// requests.
	CookieSameSiteStrict CookieSameSite = "strict"
	// CookieSameSiteNone sends the cookie with every request; the cookie
	// must be Secure.
	CookieSameSiteNone CookieSameSite = "none"
)

// Cookie is a cookie of the app's pages. Its name, domain and path
// identify it.
type Cookie struct {
	Name  string
	Value string
	// Domain is the host the cookie is sent to, such as "example.com", or
	// with a leading dot, ".example.com", the domain whose subdomains get
	// it too. It is in lowercase ASCII (punycode), without a port.
	Domain string
	// Path is the path the cookie is sent for; empty means "/".
	Path string
	// Expires is when the cookie expires, in whole seconds; zero for a
	// cookie that lasts until the app quits.
	Expires  time.Time
	Secure   bool
	HTTPOnly bool
	SameSite CookieSameSite
}

// CookieFilter chooses cookies for Cookies.List. Its zero value chooses
// every cookie; each field set narrows the choice.
type CookieFilter struct {
	// URL keeps the cookies a request to this http or https URL sends:
	// those of its host's domain and the path, Secure ones only to https.
	URL string
	// Name keeps the cookies of this name.
	Name string
	// Domain keeps the cookies of this domain and of its subdomains, with
	// or without a leading dot: "example.com" keeps those of
	// "example.com", ".example.com" and "www.example.com".
	Domain string
	// Path keeps the cookies of exactly this path.
	Path string
}

// CookiesModule reads and changes the cookies of the app's pages. Use the
// Cookies singleton.
type CookiesModule struct{}

// Cookies reads and changes the cookies of the app's pages, HTTPOnly ones
// included: the default cookie store, which every window shares, and
// which outlives the app as the system webview keeps it. Set one, e.g., to
// sign the pages in with a token the backend got, before the first page
// loads:
//
//	err := mygo.Cookies.Set(mygo.Cookie{Name: "session", Value: token,
//		Domain: "example.com", Secure: true, HTTPOnly: true})
//
// The methods are safe from any goroutine once the app is running; on the
// main thread the app keeps handling events while they wait. Cookies may
// carry the user's credentials: bind a backend method that does what the
// page needs rather than giving the page the cookies.
var Cookies CookiesModule

// List returns the cookies f chooses, sorted by domain, path and name:
//
//	all, err := mygo.Cookies.List(mygo.CookieFilter{})
//	sent, err := mygo.Cookies.List(mygo.CookieFilter{URL: "https://example.com/account"})
//	sessions, err := mygo.Cookies.List(mygo.CookieFilter{Name: "session", Domain: "example.com"})
func (CookiesModule) List(f CookieFilter) ([]Cookie, error) {
	needsApp("Cookies.List")
	if f.URL != "" {
		u, err := url.Parse(f.URL)
		if err != nil || u.Scheme != "http" && u.Scheme != "https" || u.Hostname() == "" {
			return nil, fmt.Errorf("mygo: cookies: List needs an http or https URL, not %q", f.URL)
		}
		u.Host = strings.ToLower(u.Host)
		u.User, u.Fragment, u.RawFragment = nil, "", ""
		f.URL = u.String()
	}
	domain := strings.ToLower(strings.TrimPrefix(f.Domain, "."))
	result := cookieCall(func(done func(cookieResult)) {
		backend().App().ListCookies(f.URL, func(values []platform.Cookie, err error) {
			out := make([]Cookie, 0, len(values))
			for _, c := range values {
				c.Domain = strings.ToLower(c.Domain)
				host := strings.TrimPrefix(c.Domain, ".")
				if f.Name != "" && c.Name != f.Name || f.Path != "" && c.Path != f.Path ||
					domain != "" && host != domain && !strings.HasSuffix(host, "."+domain) {
					continue
				}
				out = append(out, Cookie{c.Name, c.Value, c.Domain, c.Path, c.Expires, c.Secure, c.HTTPOnly, CookieSameSite(c.SameSite)})
			}
			slices.SortFunc(out, func(a, b Cookie) int {
				if n := strings.Compare(a.Domain, b.Domain); n != 0 {
					return n
				}
				if n := strings.Compare(a.Path, b.Path); n != 0 {
					return n
				}
				return strings.Compare(a.Name, b.Name)
			})
			done(cookieResult{out, err})
		})
	})
	return result.cookies, result.err
}

// Set adds the cookie, or replaces the one of the same name, domain and
// path. An empty path is "/" and an empty SameSite Lax; SameSite None
// needs Secure, as do the __Secure- and __Host- name prefixes. The name
// and value take what browsers take: neither has control characters, a
// ";" or spaces at either end, and the name no "=". An Expires in the past
// deletes the cookie. The engine may still drop the cookie or expire it
// earlier, as it would one a server set.
func (CookiesModule) Set(c Cookie) error {
	needsApp("Cookies.Set")
	c, err := normalizeCookie(c)
	if err != nil {
		return err
	}
	if !c.Expires.IsZero() && !c.Expires.After(time.Now()) {
		return Cookies.Delete(c)
	}
	p := platform.Cookie{Name: c.Name, Value: c.Value, Domain: c.Domain, Path: c.Path, Expires: c.Expires, Secure: c.Secure, HTTPOnly: c.HTTPOnly, SameSite: string(c.SameSite)}
	return cookieCall(func(done func(cookieResult)) {
		backend().App().SetCookie(p, func(err error) { done(cookieResult{err: err}) })
	}).err
}

// Delete deletes the cookie of c's name, domain and path; its other fields
// do not matter, so that it deletes a cookie List returned. The domain is
// matched without regard to case and with its leading dot, if any:
// "example.com" and ".example.com" are different cookies. An empty path
// is "/". Deleting a cookie that does not exist is not an error.
func (CookiesModule) Delete(c Cookie) error {
	needsApp("Cookies.Delete")
	key, err := normalizeCookieKey(c.Name, c.Domain, c.Path)
	if err != nil {
		return err
	}
	return cookieCall(func(done func(cookieResult)) {
		backend().App().DeleteCookie(key, func(err error) { done(cookieResult{err: err}) })
	}).err
}

// Clear deletes every cookie, and nothing else the pages stored, unlike
// App.ClearBrowsingData. Pages may set new ones meanwhile.
func (CookiesModule) Clear() error {
	needsApp("Cookies.Clear")
	return cookieCall(func(done func(cookieResult)) {
		backend().App().ClearCookies(func(err error) { done(cookieResult{err: err}) })
	}).err
}

type cookieResult struct {
	cookies []Cookie
	err     error
}

// cookieCalls are the calls waiting for the backend, which may never
// complete them once the app quits: stopCookieCalls fails them. It is
// used on the main thread only.
var cookieCalls = struct {
	stopped bool
	next    uint64
	pending map[uint64]func(cookieResult)
}{pending: make(map[uint64]func(cookieResult))}

func cookieCall(start func(func(cookieResult))) cookieResult {
	ch := make(chan cookieResult, 1)
	if !postMain(func() {
		if cookieCalls.stopped {
			deliver(ch, cookieResult{err: errLoopStopped})
			return
		}
		cookieCalls.next++
		id := cookieCalls.next
		done := func(r cookieResult) {
			if _, ok := cookieCalls.pending[id]; !ok {
				return
			}
			delete(cookieCalls.pending, id)
			deliver(ch, r)
		}
		cookieCalls.pending[id] = done
		start(done)
	}) {
		return cookieResult{err: errLoopStopped}
	}
	return await(ch)
}

func stopCookieCalls() {
	cookieCalls.stopped = true
	for _, done := range cookieCalls.pending {
		done(cookieResult{err: errLoopStopped})
	}
}

func normalizeCookieKey(name, domain, path string) (platform.CookieKey, error) {
	bad := func(what string) (platform.CookieKey, error) {
		return platform.CookieKey{}, fmt.Errorf("mygo: cookies: invalid %s", what)
	}
	for _, c := range []byte(name) {
		if c < 32 || c == 127 {
			return bad("name")
		}
	}
	// Check for ASCII before lowercasing: strings.ToLower maps some non-ASCII
	// runes, such as the Kelvin sign, to ASCII letters.
	for _, c := range []byte(domain) {
		if c >= 0x80 {
			return bad("domain")
		}
	}
	domain = strings.ToLower(domain)
	host := strings.TrimPrefix(domain, ".")
	if host == "" || strings.HasSuffix(host, ".") {
		return bad("domain")
	}
	if ip, err := netip.ParseAddr(host); err == nil {
		if host != domain || ip.Zone() != "" {
			return bad("IP domain")
		}
	} else {
		if len(host) > 253 {
			return bad("domain")
		}
		for _, label := range strings.Split(host, ".") {
			if len(label) == 0 || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
				return bad("domain")
			}
			for _, c := range []byte(label) {
				if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-' || c == '_') {
					return bad("domain")
				}
			}
		}
	}
	if path == "" {
		path = "/"
	}
	if path[0] != '/' {
		return bad("path")
	}
	for _, c := range []byte(path) {
		if c < 32 || c == 127 || c == ';' {
			return bad("path")
		}
	}
	return platform.CookieKey{Name: name, Domain: domain, Path: path}, nil
}

// cookieText reports whether s may be a cookie's name or value, as
// browsers parse them (RFC 6265bis section 5.6): no control character but
// tab, no ";", and no space or tab at either end, which parsing removes.
func cookieText(s string) bool {
	for _, b := range []byte(s) {
		if b < 0x20 && b != '\t' || b == 0x7f || b == ';' {
			return false
		}
	}
	return strings.Trim(s, " \t") == s
}

func normalizeCookie(c Cookie) (Cookie, error) {
	bad := func(what string) (Cookie, error) { return Cookie{}, fmt.Errorf("mygo: cookies: invalid %s", what) }
	if c.Name == "" || !cookieText(c.Name) || strings.Contains(c.Name, "=") {
		return bad("name")
	}
	if !cookieText(c.Value) {
		return bad("value")
	}
	key, err := normalizeCookieKey(c.Name, c.Domain, c.Path)
	if err != nil {
		return Cookie{}, err
	}
	c.Domain = key.Domain
	c.Path = key.Path
	if c.SameSite == "" {
		c.SameSite = CookieSameSiteLax
	}
	switch c.SameSite {
	case CookieSameSiteLax, CookieSameSiteStrict:
	case CookieSameSiteNone:
		if !c.Secure {
			return bad("SameSite=None without Secure")
		}
	default:
		return bad("SameSite")
	}
	if !c.Expires.IsZero() {
		c.Expires = c.Expires.UTC().Truncate(time.Second)
		if c.Expires.Year() < 1601 || c.Expires.Year() > 9999 {
			return bad("expiration year")
		}
	}
	if strings.HasPrefix(c.Name, "__Secure-") || strings.HasPrefix(c.Name, "__Host-") {
		if !c.Secure {
			return bad("cookie prefix without Secure")
		}
	}
	if strings.HasPrefix(c.Name, "__Host-") && (strings.HasPrefix(c.Domain, ".") || c.Path != "/") {
		return bad("__Host- domain or path")
	}
	return c, nil
}
