//go:build js

package web

import "github.com/egoist/mygo/internal/platform"

func (app) ListCookies(_ string, done func([]platform.Cookie, error)) {
	done(nil, platform.ErrUnsupported)
}
func (app) SetCookie(_ platform.Cookie, done func(error))       { done(platform.ErrUnsupported) }
func (app) DeleteCookie(_ platform.CookieKey, done func(error)) { done(platform.ErrUnsupported) }
func (app) ClearCookies(done func(error))                       { done(platform.ErrUnsupported) }
