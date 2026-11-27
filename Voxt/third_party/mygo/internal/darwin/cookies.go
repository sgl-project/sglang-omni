//go:build darwin

package darwin

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ebitengine/purego/objc"
	"github.com/egoist/mygo/internal/platform"
)

func cookieStore() (id, error) {
	store := send(class("WKWebsiteDataStore"), "defaultDataStore")
	if !sendBool(store, "respondsToSelector:", uintptr(sel("httpCookieStore"))) || !sendBool(class("NSHTTPCookie"), "instancesRespondToSelector:", uintptr(sel("sameSitePolicy"))) {
		return 0, fmt.Errorf("mygo: cookies: requires macOS 10.15 or newer: %w", platform.ErrUnsupported)
	}
	return send(store, "httpCookieStore"), nil
}

func readCookie(c id) platform.Cookie {
	out := platform.Cookie{Name: goString(send(c, "name")), Value: goString(send(c, "value")), Domain: strings.ToLower(goString(send(c, "domain"))), Path: goString(send(c, "path")), Secure: sendBool(c, "isSecure"), HTTPOnly: sendBool(c, "isHTTPOnly")}
	// Foundation names only Lax and Strict, and macOS 26 gives a cookie
	// set with SameSite=None no policy, which WebKit sends as None
	// (coreSameSitePolicy in CookieCocoa.mm).
	out.SameSite = strings.ToLower(goString(send(c, "sameSitePolicy")))
	if out.SameSite == "" {
		out.SameSite = "none"
	}
	date := send(c, "expiresDate")
	if date != 0 && !sendBool(c, "isSessionOnly") {
		out.Expires = time.Unix(int64(msgFloat(date, sel("timeIntervalSince1970"))), 0).UTC()
	}
	return out
}

// ListCookies lists every cookie, and matches them with the URL in Go:
// WKHTTPCookieStore cannot.
func (a appController) ListCookies(rawURL string, done func([]platform.Cookie, error)) {
	var u *url.URL
	if rawURL != "" {
		var err error
		if u, err = url.Parse(rawURL); err != nil {
			done(nil, err)
			return
		}
	}
	withPool(func() {
		store, err := cookieStore()
		if err != nil {
			done(nil, err)
			return
		}
		block := newBlock(func(_ objc.Block, array id) {
			var result []platform.Cookie
			withPool(func() {
				for _, c := range arrayItems(array) {
					if c := readCookie(c); u == nil || platform.CookieMatchesURL(c, u) {
						result = append(result, c)
					}
				}
			})
			a.b.runOnMain(func() { done(result, nil) })
		})
		send(store, "getAllCookies:", uintptr(block))
		block.Release()
	})
}

func (a appController) SetCookie(c platform.Cookie, done func(error)) {
	withPool(func() {
		store, err := cookieStore()
		if err != nil {
			done(err)
			return
		}
		host := strings.TrimPrefix(c.Domain, ".")
		if strings.Contains(host, ":") {
			host = "[" + host + "]"
		}
		scheme := "http"
		if c.Secure {
			scheme = "https"
		}
		// Foundation parses the attributes, with a stand-in name and value:
		// it would split a value with ", " in it as two cookies.
		header := "n=v; Path=" + c.Path + "; SameSite=" + c.SameSite
		if strings.HasPrefix(c.Domain, ".") {
			header += "; Domain=" + c.Domain
		}
		if !c.Expires.IsZero() {
			header += "; Expires=" + c.Expires.Format(http.TimeFormat)
		}
		if c.Secure {
			header += "; Secure"
		}
		if c.HTTPOnly {
			header += "; HttpOnly"
		}
		headers := send(class("NSDictionary"), "dictionaryWithObject:forKey:", uintptr(nsString(header)), uintptr(nsString("Set-Cookie")))
		array := send(class("NSHTTPCookie"), "cookiesWithResponseHeaderFields:forURL:", uintptr(headers), uintptr(nsURL(scheme+"://"+host+"/")))
		if sendInt(array, "count") != 1 {
			done(fmt.Errorf("mygo: cookies: Set could not construct NSHTTPCookie"))
			return
		}
		props := send(send(send(array, "objectAtIndex:", 0), "properties"), "mutableCopy")
		send(props, "setObject:forKey:", uintptr(nsString(c.Name)), uintptr(constString(libFoundation, "NSHTTPCookieName")))
		send(props, "setObject:forKey:", uintptr(nsString(c.Value)), uintptr(constString(libFoundation, "NSHTTPCookieValue")))
		native := send(class("NSHTTPCookie"), "cookieWithProperties:", uintptr(props))
		release(props)
		if native == 0 {
			done(fmt.Errorf("mygo: cookies: Set could not construct NSHTTPCookie"))
			return
		}
		parsed := readCookie(native)
		if parsed.Name != c.Name || parsed.Domain != c.Domain || parsed.Path != c.Path || parsed.Value != c.Value || parsed.Secure != c.Secure || parsed.HTTPOnly != c.HTTPOnly || parsed.SameSite != c.SameSite {
			// Say what differs, without the value.
			parsed.Value, c.Value = "", ""
			done(fmt.Errorf("mygo: cookies: Set made NSHTTPCookie %+v of %+v", parsed, c))
			return
		}
		block := newBlock(func(objc.Block) { a.b.runOnMain(func() { done(nil) }) })
		send(store, "setCookie:completionHandler:", uintptr(native), uintptr(block))
		block.Release()
	})
}

func (a appController) DeleteCookie(key platform.CookieKey, done func(error)) {
	withPool(func() {
		store, err := cookieStore()
		if err != nil {
			done(err)
			return
		}
		retain(store)
		block := newBlock(func(_ objc.Block, array id) {
			// The borrowed array must survive dispatch to main.
			retain(array)
			a.b.runOnMain(func() {
				withPool(func() {
					defer release(array)
					var match id
					for _, c := range arrayItems(array) {
						if goString(send(c, "name")) == key.Name && strings.ToLower(goString(send(c, "domain"))) == key.Domain && goString(send(c, "path")) == key.Path {
							if match != 0 {
								release(store)
								done(fmt.Errorf("mygo: cookies: ambiguous cookie key"))
								return
							}
							match = c
						}
					}
					if match == 0 {
						release(store)
						done(nil)
						return
					}
					retain(match)
					completion := newBlock(func(objc.Block) { a.b.runOnMain(func() { release(match); release(store); done(nil) }) })
					send(store, "deleteCookie:completionHandler:", uintptr(match), uintptr(completion))
					completion.Release()
				})
			})
		})
		send(store, "getAllCookies:", uintptr(block))
		block.Release()
	})
}

// ClearCookies removes the cookies of the default data store, its
// other data kept.
func (a appController) ClearCookies(done func(error)) {
	withPool(func() {
		if _, err := cookieStore(); err != nil {
			done(err)
			return
		}
		store := send(class("WKWebsiteDataStore"), "defaultDataStore")
		types := send(class("NSSet"), "setWithObject:", uintptr(nsString("WKWebsiteDataTypeCookies")))
		blk := newBlock(func(_ objc.Block) { a.b.runOnMain(func() { done(nil) }) })
		send(store, "removeDataOfTypes:modifiedSince:completionHandler:", uintptr(types), uintptr(send(class("NSDate"), "distantPast")), uintptr(blk))
		blk.Release()
	})
}
