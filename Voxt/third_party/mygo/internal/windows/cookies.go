//go:build windows && (amd64 || arm64)

package windows

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

// Mechanically checked against Microsoft's Microsoft.Web.WebView2 NuGet
// 1.0.4191.47, build/native/include/WebView2.h (C interface vtable order).
// https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/1.0.4191.47/microsoft.web.webview2.1.0.4191.47.nupkg
const (
	wv2GetCookieManager                  = 66
	cookieManagerCreate                  = 3
	cookieManagerGet                     = 5
	cookieManagerAdd                     = 6
	cookieManagerDeleteWithDomainAndPath = 9
	cookieManagerDeleteAll               = 10
	cookieListCount                      = 3
	cookieListValue                      = 4
	cookieGetName                        = 3
	cookieGetValue                       = 4
	cookieGetDomain                      = 6
	cookieGetPath                        = 7
	cookieGetExpires                     = 8
	cookiePutExpires                     = 9
	cookieGetHTTPOnly                    = 10
	cookiePutHTTPOnly                    = 11
	cookieGetSameSite                    = 12
	cookiePutSameSite                    = 13
	cookieGetSecure                      = 14
	cookiePutSecure                      = 15
	cookieGetSession                     = 16
)

var iidICoreWebView2_2 = guid("9e8f0cf8-e670-4b5e-b2bc-73e061e3184c")

// cookieStore gives the cookie calls the cookie manager of the profile
// the app's web views share. WebView2 has one only through a web view:
// a call takes that of a window showing a page, or, before there is one,
// that of a hidden web view of the app's own window, made for it and
// closed once windows show pages and no call needs it. A window's first
// navigation waits for the calls under way when its web view is made, so
// that a cookie set meanwhile goes along. All of it is main-thread only.
type cookieStore struct {
	controller uintptr // the hidden web view's, once made
	webview    uintptr
	creating   bool
	closing    bool // a close of the hidden web view is posted
	waiting    []func(error)
	requests   map[*cookieRequest]struct{}
	next       uint64       // the number of the next request
	held       []heldWindow // windows waiting for requests to finish
}

// heldWindow is a window whose first navigation waits for the requests
// numbered below before.
type heldWindow struct {
	w      *window
	before uint64
}

// A cookieRequest is one call; closing a controller does not guarantee
// that WebView2 invokes outstanding completion handlers, so closing a
// window finishes the requests using it.
type cookieRequest struct {
	b        *Backend
	n        uint64
	window   *window // whose web view it uses, or nil for the hidden one
	webview  uintptr // borrowed while the request runs
	manager  uintptr
	finished bool
	done     func([]platform.Cookie, error)
}

func (r *cookieRequest) finish(c []platform.Cookie, err error) {
	if r.finished {
		return
	}
	r.finished = true
	s := &r.b.cookies
	delete(s.requests, r)
	release(r.manager)
	r.webview, r.manager = 0, 0
	r.done(c, err)
	oldest := s.next
	for q := range s.requests {
		oldest = min(oldest, q.n)
	}
	var ready []*window
	s.held = slices.DeleteFunc(s.held, func(h heldWindow) bool {
		if h.before <= oldest {
			ready = append(ready, h.w)
			return true
		}
		return false
	})
	for _, w := range ready {
		w.becomeReady()
	}
	r.b.closeCookieView()
}

func failCookieRequests(w *window, err error) {
	for r := range w.b.cookies.requests {
		if r.window == w {
			r.finish(nil, err)
		}
	}
}

// holdForCookies reports whether w's first navigation must wait for the
// cookie calls under way, which make it ready when they finish.
func (b *Backend) holdForCookies(w *window) bool {
	if len(b.cookies.requests) == 0 {
		return false
	}
	b.cookies.held = append(b.cookies.held, heldWindow{w, b.cookies.next})
	return true
}

// pageWebView returns a window whose web view shows pages, or nil.
func (b *Backend) pageWebView() *window {
	var selected *window
	for hwnd, w := range b.windows {
		if w.closed || w.surface != nil || !w.ready || w.webview == 0 {
			continue
		}
		if selected == nil || hwnd < selected.hwnd {
			selected = w
		}
	}
	return selected
}

// cookieManagerOf returns the cookie manager of webview, a reference.
func cookieManagerOf(webview uintptr) (uintptr, error) {
	view := queryInterface(webview, &iidICoreWebView2_2)
	if view == 0 {
		return 0, fmt.Errorf("mygo: cookies: requires WebView2 ICoreWebView2_2 (SDK 1.0.705.50+): %w", platform.ErrUnsupported)
	}
	var manager uintptr
	hr := comCall(view, wv2GetCookieManager, uintptr(unsafe.Pointer(&manager)))
	release(view)
	if failed(hr) || manager == 0 {
		return 0, hresultError("cookies: get_CookieManager", hr)
	}
	return manager, nil
}

// withCookieManager starts a request, which start makes with the cookie
// manager.
func (a appController) withCookieManager(done func([]platform.Cookie, error), start func(*cookieRequest)) {
	b := a.b
	if b.cookies.requests == nil {
		b.cookies.requests = map[*cookieRequest]struct{}{}
	}
	r := &cookieRequest{b: b, n: b.cookies.next, done: done}
	b.cookies.next++
	b.cookies.requests[r] = struct{}{}
	begin := func(webview uintptr) {
		manager, err := cookieManagerOf(webview)
		if err != nil {
			r.finish(nil, err)
			return
		}
		r.webview, r.manager = webview, manager
		start(r)
	}
	if w := b.pageWebView(); w != nil {
		r.window = w
		begin(w.webview)
		return
	}
	b.cookieView(func(err error) {
		if r.finished {
			return
		}
		if err != nil {
			r.finish(nil, err)
			return
		}
		begin(b.cookies.webview)
	})
}

// cookieView runs done once the hidden web view exists, or with why it
// could not be made; a later call tries again.
func (b *Backend) cookieView(done func(error)) {
	s := &b.cookies
	if s.webview != 0 {
		done(nil)
		return
	}
	s.waiting = append(s.waiting, done)
	if s.creating {
		return
	}
	s.creating = true
	finish := func(err error) {
		s.creating = false
		waiting := s.waiting
		s.waiting = nil
		for _, fn := range waiting {
			fn(err)
		}
	}
	if err := b.startEnvironment(); err != nil {
		finish(err)
		return
	}
	b.whenEnvironment(func() {
		if b.envErr != nil {
			finish(b.envErr)
			return
		}
		hr := withHandler(func(hr, controller uintptr) {
			if failed(hr) || controller == 0 {
				finish(hresultError("cookies: creating a WebView2 controller", hr))
				return
			}
			addRef(controller)
			comCall(controller, ctlPutIsVisible, 0)
			var webview uintptr
			if hr := comCall(controller, ctlGetCoreWebView2, uintptr(unsafe.Pointer(&webview))); failed(hr) || webview == 0 {
				comCall(controller, ctlClose)
				release(controller)
				finish(hresultError("cookies: get_CoreWebView2", hr))
				return
			}
			s.controller, s.webview = controller, webview
			finish(nil)
		}, func(h uintptr) uintptr { return comCall(b.env, envCreateController, b.appHwnd, h) })
		if failed(hr) {
			finish(hresultError("cookies: CreateCoreWebView2Controller", hr))
		}
	})
}

// closeCookieView closes the hidden web view once windows show pages and
// no call needs it. It does so from the message loop, not here: a call
// finishes in a completion handler of the hidden web view, and closing the
// view from inside one crashes the message loop.
func (b *Backend) closeCookieView() {
	s := &b.cookies
	if s.controller == 0 || s.closing {
		return
	}
	s.closing = true
	postMessage(b.appHwnd, wmAppCookieView, 0, 0)
}

// closeCookieViewNow closes the hidden web view, as closeCookieView asks,
// if windows show pages and no call needs it by now.
func (b *Backend) closeCookieViewNow() {
	s := &b.cookies
	s.closing = false
	if s.controller == 0 || len(s.requests) > 0 || b.pageWebView() == nil {
		return
	}
	comCall(s.controller, ctlClose)
	release(s.webview)
	release(s.controller)
	s.controller, s.webview = 0, 0
}

// enumerate gets the cookies a request to uri sends, every cookie when
// uri is empty, and passes them to visit.
func (r *cookieRequest) enumerate(uri string, visit func(uintptr) error) {
	hr := withHandler(func(hr, list uintptr) {
		if r.finished {
			return
		}
		if failed(hr) || list == 0 {
			r.finish(nil, hresultError("cookies: GetCookies completion", hr))
			return
		}
		if err := visit(list); err != nil {
			r.finish(nil, err)
		}
	}, func(handler uintptr) uintptr {
		return comCall(r.manager, cookieManagerGet, uintptr(unsafe.Pointer(u16(uri))), handler)
	})
	if failed(hr) {
		r.finish(nil, hresultError("cookies: GetCookies", hr))
	}
}

func cookieEach(list uintptr, visit func(uintptr) error) error {
	var count uint32
	if hr := comCall(list, cookieListCount, uintptr(unsafe.Pointer(&count))); failed(hr) {
		return hresultError("cookies: get_Count", hr)
	}
	for i := uint32(0); i < count; i++ {
		var c uintptr
		if hr := comCall(list, cookieListValue, uintptr(i), uintptr(unsafe.Pointer(&c))); failed(hr) || c == 0 {
			return hresultError("cookies: GetValueAtIndex", hr)
		}
		err := visit(c)
		release(c)
		if err != nil {
			return err
		}
	}
	return nil
}

func cookieString(c uintptr, slot int) (string, error) {
	var p uintptr
	hr := comCall(c, slot, uintptr(unsafe.Pointer(&p)))
	value := takeWstr(p)
	if failed(hr) {
		return "", hresultError("cookies: string property", hr)
	}
	return value, nil
}

func readWindowsCookie(c uintptr) (platform.Cookie, error) {
	var out platform.Cookie
	for _, field := range []struct {
		slot  int
		value *string
	}{{cookieGetName, &out.Name}, {cookieGetValue, &out.Value}, {cookieGetDomain, &out.Domain}, {cookieGetPath, &out.Path}} {
		v, err := cookieString(c, field.slot)
		if err != nil {
			return out, err
		}
		*field.value = v
	}
	out.Domain = strings.ToLower(out.Domain)
	var secure, httpOnly, session, policy int32
	for _, field := range []struct {
		slot  int
		value *int32
	}{{cookieGetSecure, &secure}, {cookieGetHTTPOnly, &httpOnly}, {cookieGetSession, &session}, {cookieGetSameSite, &policy}} {
		if hr := comCall(c, field.slot, uintptr(unsafe.Pointer(field.value))); failed(hr) {
			return out, hresultError("cookies: property", hr)
		}
	}
	out.Secure = secure != 0
	out.HTTPOnly = httpOnly != 0
	switch policy {
	case 0:
		out.SameSite = "none"
	case 1:
		out.SameSite = "lax"
	case 2:
		out.SameSite = "strict"
	}
	if session == 0 {
		var seconds float64
		if hr := comCall(c, cookieGetExpires, uintptr(unsafe.Pointer(&seconds))); failed(hr) {
			return out, hresultError("cookies: get_Expires", hr)
		}
		if math.IsNaN(seconds) || math.IsInf(seconds, 0) || seconds < 0 || seconds > 253402300799 {
			return out, fmt.Errorf("mygo: cookies: invalid native expiration")
		}
		out.Expires = time.Unix(int64(seconds), 0).UTC()
	}
	return out, nil
}

func (a appController) ListCookies(url string, done func([]platform.Cookie, error)) {
	a.withCookieManager(done, func(r *cookieRequest) {
		r.enumerate(url, func(list uintptr) error {
			result := []platform.Cookie{}
			err := cookieEach(list, func(c uintptr) error {
				value, err := readWindowsCookie(c)
				if err == nil {
					result = append(result, value)
				}
				return err
			})
			r.finish(result, err)
			return nil
		})
	})
}

func (a appController) SetCookie(c platform.Cookie, done func(error)) {
	a.withCookieManager(func(_ []platform.Cookie, err error) { done(err) }, func(r *cookieRequest) {
		var nativeCookie uintptr
		hr := comCall(r.manager, cookieManagerCreate, uintptr(unsafe.Pointer(u16(c.Name))), uintptr(unsafe.Pointer(u16(c.Value))), uintptr(unsafe.Pointer(u16(c.Domain))), uintptr(unsafe.Pointer(u16(c.Path))), uintptr(unsafe.Pointer(&nativeCookie)))
		if failed(hr) || nativeCookie == 0 {
			r.finish(nil, hresultError("cookies: CreateCookie", hr))
			return
		}
		defer release(nativeCookie)
		seconds := -1.0
		if !c.Expires.IsZero() {
			seconds = float64(c.Expires.Unix())
		}
		if hr := comCallFloat(nativeCookie, cookiePutExpires, seconds); failed(hr) {
			r.finish(nil, hresultError("cookies: put_Expires", hr))
			return
		}
		var secure, httpOnly uintptr
		if c.Secure {
			secure = 1
		}
		if c.HTTPOnly {
			httpOnly = 1
		}
		policy := uintptr(1)
		if c.SameSite == "none" {
			policy = 0
		} else if c.SameSite == "strict" {
			policy = 2
		}
		for _, property := range []struct {
			slot  int
			value uintptr
		}{{cookiePutSecure, secure}, {cookiePutHTTPOnly, httpOnly}, {cookiePutSameSite, policy}} {
			if hr := comCall(nativeCookie, property.slot, property.value); failed(hr) {
				r.finish(nil, hresultError("cookies: property setter", hr))
				return
			}
		}
		if hr := comCall(r.manager, cookieManagerAdd, nativeCookie); failed(hr) {
			r.finish(nil, hresultError("cookies: AddOrUpdateCookie", hr))
			return
		}
		want := storedCookie{platform.CookieKey{Name: c.Name, Domain: c.Domain, Path: c.Path}, c.Value}
		r.settle(true, func(s storedCookie) bool { return s == want }, time.Now().Add(cookieSettleTime),
			"the engine did not keep the cookie")
	})
}

func (a appController) DeleteCookie(key platform.CookieKey, done func(error)) {
	a.withCookieManager(func(_ []platform.Cookie, err error) { done(err) }, func(r *cookieRequest) {
		// The cookies of key there are now, which must go; a page may set
		// the cookie again meanwhile.
		r.stored(func(s storedCookie) bool { return s.CookieKey == key }, func(old map[storedCookie]bool) {
			if len(old) == 0 {
				r.finish(nil, nil)
				return
			}
			hr := comCall(r.manager, cookieManagerDeleteWithDomainAndPath, uintptr(unsafe.Pointer(u16(key.Name))), uintptr(unsafe.Pointer(u16(key.Domain))), uintptr(unsafe.Pointer(u16(key.Path))))
			if failed(hr) {
				r.finish(nil, hresultError("cookies: DeleteCookiesWithDomainAndPath", hr))
				return
			}
			r.settle(false, func(s storedCookie) bool { return old[s] }, time.Now().Add(cookieSettleTime),
				"the cookie is still there")
		})
	})
}

// ClearCookies deletes every cookie of the profile: those there now must
// go, pages may set new ones meanwhile.
func (a appController) ClearCookies(done func(error)) {
	a.withCookieManager(func(_ []platform.Cookie, err error) { done(err) }, func(r *cookieRequest) {
		r.stored(func(storedCookie) bool { return true }, func(old map[storedCookie]bool) {
			if hr := comCall(r.manager, cookieManagerDeleteAll); failed(hr) {
				r.finish(nil, hresultError("cookies: DeleteAllCookies", hr))
				return
			}
			r.settle(false, func(s storedCookie) bool { return old[s] }, time.Now().Add(cookieSettleTime),
				"cookies are still there")
		})
	})
}

// cookieSettleTime is how long a change may take to show in GetCookies.
const cookieSettleTime = 2 * time.Second

// storedCookie is what tells a cookie apart from a page's newer one.
type storedCookie struct {
	platform.CookieKey
	value string
}

// stored passes the cookies that match to then.
func (r *cookieRequest) stored(match func(storedCookie) bool, then func(map[storedCookie]bool)) {
	r.enumerate("", func(list uintptr) error {
		found := map[storedCookie]bool{}
		err := cookieEach(list, func(c uintptr) error {
			s, err := readStoredCookie(c)
			if err == nil && match(s) {
				found[s] = true
			}
			return err
		})
		if err != nil {
			return err
		}
		then(found)
		return nil
	})
}

func readStoredCookie(c uintptr) (storedCookie, error) {
	var fields [4]string
	for i, slot := range []int{cookieGetName, cookieGetDomain, cookieGetPath, cookieGetValue} {
		v, err := cookieString(c, slot)
		if err != nil {
			return storedCookie{}, err
		}
		fields[i] = v
	}
	return storedCookie{platform.CookieKey{Name: fields[0], Domain: strings.ToLower(fields[1]), Path: fields[2]}, fields[3]}, nil
}

// settle finishes once a cookie matches, when present, or none does:
// WebView2 changes cookies after its calls return, and GetCookies may not
// see the change for some milliseconds. GetCookies("") lists every
// cookie, HttpOnly ones included, so a change that does not show by the
// deadline did not happen: the request fails with why.
func (r *cookieRequest) settle(present bool, match func(storedCookie) bool, deadline time.Time, why string) {
	r.stored(match, func(found map[storedCookie]bool) {
		if (len(found) > 0) == present {
			r.finish(nil, nil)
		} else if time.Now().Before(deadline) {
			r.settle(present, match, deadline, why)
		} else {
			r.finish(nil, errors.New("mygo: cookies: "+why))
		}
	})
}
