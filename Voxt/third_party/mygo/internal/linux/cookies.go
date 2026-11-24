//go:build linux && (amd64 || arm64)

package linux

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

var (
	soup3                                                                                bool
	cookieManager                                                                        func(ptr) ptr
	cookieList                                                                           func(ptr, ptr, ptr, ptr)
	cookieListFinish                                                                     func(ptr, ptr, *ptr) ptr
	cookieGet                                                                            func(ptr, *byte, ptr, ptr, ptr)
	cookieGetFinish                                                                      func(ptr, ptr, *ptr) ptr
	cookieSetPersistentStorage                                                           func(ptr, *byte, int32)
	cookieAdd, cookieDelete                                                              func(ptr, ptr, ptr, ptr, ptr)
	cookieAddFinish, cookieDeleteFinish                                                  func(ptr, ptr, *ptr) int32
	soupCookieNew                                                                        func(*byte, *byte, *byte, *byte, int32) ptr
	soupCookieFree                                                                       func(ptr)
	soupCookieCopy                                                                       func(ptr) ptr
	soupCookieName, soupCookieValue, soupCookieDomain, soupCookiePath, soupCookieExpires func(ptr) ptr
	soupCookieSecure, soupCookieHTTPOnly, soupCookieSameSite                             func(ptr) int32
	soupCookieSetSecure, soupCookieSetHTTPOnly, soupCookieSetSameSite                    func(ptr, int32)
	soupCookieSetExpires                                                                 func(ptr, ptr)
	cookieDateNew                                                                        func(int64) ptr
	cookieDateUnix                                                                       func(ptr) int64
	cookieDateFree                                                                       func(ptr)
	cookieListFree                                                                       func(ptr)
)

func bindCookies() {
	bind(libWebKit, &cookieManager, "webkit_web_context_get_cookie_manager")
	bind(libWebKit, &cookieList, "webkit_cookie_manager_get_all_cookies")
	bind(libWebKit, &cookieListFinish, "webkit_cookie_manager_get_all_cookies_finish")
	bind(libWebKit, &cookieGet, "webkit_cookie_manager_get_cookies")
	bind(libWebKit, &cookieGetFinish, "webkit_cookie_manager_get_cookies_finish")
	bind(libWebKit, &cookieSetPersistentStorage, "webkit_cookie_manager_set_persistent_storage")
	bind(libWebKit, &cookieAdd, "webkit_cookie_manager_add_cookie")
	bind(libWebKit, &cookieAddFinish, "webkit_cookie_manager_add_cookie_finish")
	bind(libWebKit, &cookieDelete, "webkit_cookie_manager_delete_cookie")
	bind(libWebKit, &cookieDeleteFinish, "webkit_cookie_manager_delete_cookie_finish")
	bind(libSoup, &soupCookieNew, "soup_cookie_new")
	bind(libSoup, &soupCookieFree, "soup_cookie_free")
	bind(libSoup, &soupCookieCopy, "soup_cookie_copy")
	bind(libSoup, &soupCookieName, "soup_cookie_get_name")
	bind(libSoup, &soupCookieValue, "soup_cookie_get_value")
	bind(libSoup, &soupCookieDomain, "soup_cookie_get_domain")
	bind(libSoup, &soupCookiePath, "soup_cookie_get_path")
	bind(libSoup, &soupCookieExpires, "soup_cookie_get_expires")
	bind(libSoup, &soupCookieSecure, "soup_cookie_get_secure")
	bind(libSoup, &soupCookieHTTPOnly, "soup_cookie_get_http_only")
	bind(libSoup, &soupCookieSameSite, "soup_cookie_get_same_site_policy")
	bind(libSoup, &soupCookieSetSecure, "soup_cookie_set_secure")
	bind(libSoup, &soupCookieSetHTTPOnly, "soup_cookie_set_http_only")
	bind(libSoup, &soupCookieSetSameSite, "soup_cookie_set_same_site_policy")
	bind(libSoup, &soupCookieSetExpires, "soup_cookie_set_expires")
	bind(libGLib, &cookieListFree, "g_list_free")
	// GLib 2.80 gdatetime.h and libsoup 2.74 soup-date.h: Soup3 uses
	// refcounted GDateTime, Soup2 copied SoupDate; time_t is 64-bit on our
	// supported Linux amd64/arm64 targets. Never share their pointer types.
	if soup3 {
		bind(libGLib, &cookieDateNew, "g_date_time_new_from_unix_utc")
		bind(libGLib, &cookieDateUnix, "g_date_time_to_unix")
		bind(libGLib, &cookieDateFree, "g_date_time_unref")
	} else {
		bind(libSoup, &cookieDateNew, "soup_date_new_from_time_t")
		bind(libSoup, &cookieDateUnix, "soup_date_to_time_t")
		bind(libSoup, &cookieDateFree, "soup_date_free")
	}
}

var cookiesOnce sync.Once

// persistCookies keeps the cookies of the default web context in
// cookies.sqlite of the app's user data directory, as macOS and Windows
// keep theirs: WebKitGTK keeps them in memory unless told where to store
// them. It runs once WebKitGTK is loaded, before any web view loads a page.
func persistCookies() {
	if cookieManager == nil || cookieSetPersistentStorage == nil {
		return
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		log.Printf("mygo: cookies: %v", err)
		return
	}
	dir = filepath.Join(dir, theBackend.name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		log.Printf("mygo: cookies: %v", err)
		return
	}
	const sqlite = 1 // WEBKIT_COOKIE_PERSISTENT_STORAGE_SQLITE
	cookieSetPersistentStorage(cookieManager(webkitWebContextGetDefault()), cs(filepath.Join(dir, "cookies.sqlite")), sqlite)
}

// linuxCookieManager loads WebKitGTK and returns the cookie manager of
// the default web context, when the functions available checks, once
// bound, are there.
func linuxCookieManager(operation, needs string, available func() bool) (ptr, error) {
	if err := webKit(); err != nil {
		return 0, fmt.Errorf("mygo: cookies: %s: %w", operation, err)
	}
	if !available() || cookieManager == nil || soupCookieNew == nil || soupCookieFree == nil {
		return 0, fmt.Errorf("mygo: cookies: %s needs %s: %w", operation, needs, platform.ErrUnsupported)
	}
	return cookieManager(webkitWebContextGetDefault()), nil
}

// allCookies lists the cookies of the default web context, those a
// request to uri sends unless it is empty. done owns the SoupCookies and
// frees them.
func allCookies(operation, uri string, done func([]ptr, error)) {
	manager, err := linuxCookieManager(operation, "WebKitGTK 2.42 or newer", func() bool {
		return cookieList != nil && cookieListFinish != nil && cookieGet != nil && cookieGetFinish != nil && cookieListFree != nil && soupCookieName != nil && soupCookieValue != nil && soupCookieDomain != nil && soupCookiePath != nil && soupCookieExpires != nil && soupCookieSecure != nil && soupCookieHTTPOnly != nil && cookieDateUnix != nil
	})
	if err != nil {
		done(nil, err)
		return
	}
	request := pending.add(func(source, result ptr) {
		var ge ptr
		finish := cookieListFinish
		if uri != "" {
			finish = cookieGetFinish
		}
		list := finish(source, result, &ge)
		// WebKit transfers the GList (data, next, prev) and its cookies.
		var cookies []ptr
		for node := list; node != 0; {
			entry := *(**[3]ptr)(unsafe.Pointer(&node))
			cookies = append(cookies, entry[0])
			node = entry[1]
		}
		cookieListFree(list)
		if err := gErr(ge); err != nil {
			freeCookies(cookies)
			done(nil, fmt.Errorf("mygo: cookies: %s: %w", operation, err))
			return
		}
		done(cookies, nil)
	})
	if uri != "" {
		cookieGet(manager, cs(uri), 0, cbAsyncReady, request)
		return
	}
	cookieList(manager, 0, cbAsyncReady, request)
}

func freeCookies(cookies []ptr) {
	for _, c := range cookies {
		soupCookieFree(c)
	}
}

func (a appController) ListCookies(uri string, done func([]platform.Cookie, error)) {
	allCookies("List", uri, func(cookies []ptr, err error) {
		if err != nil {
			done(nil, err)
			return
		}
		values := make([]platform.Cookie, 0, len(cookies))
		for _, c := range cookies {
			out := platform.Cookie{Name: goStr(soupCookieName(c)), Value: goStr(soupCookieValue(c)), Domain: goStr(soupCookieDomain(c)), Path: goStr(soupCookiePath(c)), Secure: soupCookieSecure(c) != 0, HTTPOnly: soupCookieHTTPOnly(c) != 0}
			if d := soupCookieExpires(c); d != 0 {
				out.Expires = time.Unix(cookieDateUnix(d), 0).UTC()
			}
			if soupCookieSameSite != nil {
				switch soupCookieSameSite(c) {
				case 0:
					out.SameSite = "none"
				case 1:
					out.SameSite = "lax"
				case 2:
					out.SameSite = "strict"
				}
			}
			values = append(values, out)
		}
		freeCookies(cookies)
		done(values, nil)
	})
}

func (a appController) SetCookie(c platform.Cookie, done func(error)) {
	manager, err := linuxCookieManager("Set", "WebKitGTK 2.20 and libsoup 2.70 or newer", func() bool {
		return cookieAdd != nil && cookieAddFinish != nil && soupCookieSetSecure != nil && soupCookieSetHTTPOnly != nil && soupCookieSetSameSite != nil && soupCookieSetExpires != nil && cookieDateNew != nil && cookieDateFree != nil
	})
	if err != nil {
		done(err)
		return
	}
	native := soupCookieNew(cs(c.Name), cs(c.Value), cs(c.Domain), cs(c.Path), -1)
	if native == 0 {
		done(fmt.Errorf("mygo: cookies: Set could not construct SoupCookie"))
		return
	}
	var secure, httpOnly int32
	if c.Secure {
		secure = 1
	}
	if c.HTTPOnly {
		httpOnly = 1
	}
	soupCookieSetSecure(native, secure)
	soupCookieSetHTTPOnly(native, httpOnly)
	policy := int32(1)
	if c.SameSite == "none" {
		policy = 0
	} else if c.SameSite == "strict" {
		policy = 2
	}
	soupCookieSetSameSite(native, policy)
	if !c.Expires.IsZero() {
		date := cookieDateNew(c.Expires.Unix())
		if date == 0 {
			soupCookieFree(native)
			done(fmt.Errorf("mygo: cookies: Set cannot represent expiration"))
			return
		}
		soupCookieSetExpires(native, date)
		cookieDateFree(date)
	}
	request := pending.add(func(source, result ptr) {
		var ge ptr
		ok := cookieAddFinish(source, result, &ge)
		soupCookieFree(native)
		done(cookieFinishError("Set", ok, ge))
	})
	cookieAdd(manager, native, 0, cbAsyncReady, request)
}

func (a appController) DeleteCookie(key platform.CookieKey, done func(error)) {
	if err := webKit(); err != nil {
		done(fmt.Errorf("mygo: cookies: Delete: %w", err))
		return
	}
	if cookieDelete == nil || cookieDeleteFinish == nil || soupCookieCopy == nil {
		done(fmt.Errorf("mygo: cookies: Delete needs WebKitGTK 2.42 or newer: %w", platform.ErrUnsupported))
		return
	}
	// libsoup deletes the cookie of the same name, path and value: find
	// the stored one, whose value the key lacks.
	allCookies("Delete", "", func(cookies []ptr, err error) {
		if err != nil {
			done(err)
			return
		}
		defer freeCookies(cookies)
		var match ptr
		for _, c := range cookies {
			if goStr(soupCookieName(c)) == key.Name && strings.ToLower(goStr(soupCookieDomain(c))) == key.Domain && goStr(soupCookiePath(c)) == key.Path {
				if match != 0 {
					done(fmt.Errorf("mygo: cookies: ambiguous cookie key"))
					return
				}
				match = c
			}
		}
		if match == 0 {
			done(nil)
			return
		}
		match = soupCookieCopy(match)
		request := pending.add(func(source, result ptr) {
			var ge ptr
			ok := cookieDeleteFinish(source, result, &ge)
			soupCookieFree(match)
			done(cookieFinishError("Delete", ok, ge))
		})
		cookieDelete(cookieManager(webkitWebContextGetDefault()), match, 0, cbAsyncReady, request)
	})
}

// ClearCookies clears the cookies of the default web context's website
// data, its other data kept.
func (a appController) ClearCookies(done func(error)) {
	if err := webKit(); err != nil {
		done(fmt.Errorf("mygo: cookies: Clear: %w", err))
		return
	}
	const cookies = 1 << 8 // WEBKIT_WEBSITE_DATA_COOKIES
	manager := webkitWebContextGetWebsiteDataManager(webkitWebContextGetDefault())
	id := pending.add(func(source, res ptr) {
		var ge ptr
		var ok int32
		if webkitWebsiteDataManagerClearFinish(source, res, &ge) {
			ok = 1
		}
		done(cookieFinishError("Clear", ok, ge))
	})
	webkitWebsiteDataManagerClear(manager, cookies, 0, 0, cbAsyncReady, id)
}

func cookieFinishError(operation string, ok int32, ge ptr) error {
	if err := gErr(ge); err != nil {
		return fmt.Errorf("mygo: cookies: %s: %w", operation, err)
	}
	if ok == 0 {
		return fmt.Errorf("mygo: cookies: %s failed", operation)
	}
	return nil
}
