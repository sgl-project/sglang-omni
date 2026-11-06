package fake

import (
	"net/url"
	"time"

	"github.com/egoist/mygo/internal/platform"
)

// ListCookies returns the cookies that have not expired, those a request
// to rawURL sends unless it is empty.
func (a app) ListCookies(rawURL string, done func([]platform.Cookie, error)) {
	var u *url.URL
	if rawURL != "" {
		var err error
		if u, err = url.Parse(rawURL); err != nil {
			done(nil, err)
			return
		}
	}
	a.b.mu.Lock()
	now := time.Now()
	result := []platform.Cookie{}
	for _, c := range a.b.cookies {
		if (c.Expires.IsZero() || c.Expires.After(now)) && (u == nil || platform.CookieMatchesURL(c, u)) {
			result = append(result, c)
		}
	}
	a.b.mu.Unlock()
	done(result, nil)
}

func (a app) SetCookie(c platform.Cookie, done func(error)) {
	a.b.mu.Lock()
	if a.b.cookies == nil {
		a.b.cookies = map[platform.CookieKey]platform.Cookie{}
	}
	a.b.cookies[platform.CookieKey{Name: c.Name, Domain: c.Domain, Path: c.Path}] = c
	a.b.mu.Unlock()
	done(nil)
}

func (a app) DeleteCookie(key platform.CookieKey, done func(error)) {
	a.b.mu.Lock()
	delete(a.b.cookies, key)
	a.b.mu.Unlock()
	done(nil)
}

func (a app) ClearCookies(done func(error)) {
	a.b.mu.Lock()
	clear(a.b.cookies)
	a.b.CookiesCleared++
	a.b.mu.Unlock()
	done(nil)
}
