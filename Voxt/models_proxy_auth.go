package main

// A custom proxy's user name and password (the Network… dialog), as Voxt
// 1.x's: Voxt's only, never the family's. The user name is in the
// settings (Network.ProxyUser), the password in the Keychain
// ("proxy.<host>") and in memory, both for the proxy host they were saved
// with (Network.ProxyAuthHost). Downloads and the remote services send
// them to a custom proxy of that host when both are set; automatic and no
// proxy keep them for when the proxy is custom again.

import (
	"log"
	"net/url"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// proxySecret is the password of the proxy host it was saved for.
type proxySecret struct{ host, password string }

// proxyAccount is the Keychain account of proxy host's password.
func proxyAccount(host string) string { return "proxy." + host }

// proxyHost is the host of a custom proxy's URL, "" for automatic or no
// proxy.
func proxyHost(mode string) string {
	u, err := url.Parse(mode)
	if err != nil {
		return ""
	}
	return u.Hostname()
}

// proxyUser is what the proxy of mode gets as user name and password:
// those saved, when mode is a custom proxy of the host they are for and
// both are set; else nil.
func (a *app) proxyUser(prefs settings.Settings, mode string) *url.Userinfo {
	n := prefs.Network
	secret := a.proxyPassword.Load()
	if n.ProxyUser == "" || n.ProxyAuthHost == "" || secret == nil || secret.password == "" || secret.host != n.ProxyAuthHost || proxyHost(mode) != n.ProxyAuthHost {
		return nil
	}
	return url.UserPassword(n.ProxyUser, secret.password)
}

// loadProxyPassword reads the saved password off the main thread, then
// gives the downloads and the remote services the network again (at the
// start, once the keyring is there).
func (a *app) loadProxyPassword() {
	host, keys := a.prefs.Network.ProxyAuthHost, a.keys
	if host == "" || keys == nil {
		return
	}
	a.async(func() {
		password, err := keys.Get(proxyAccount(host))
		if err != nil {
			log.Printf("network: the proxy's password: %v", err)
			return
		}
		a.post(func() {
			a.proxyPassword.Store(&proxySecret{host: host, password: password})
			a.applyNetwork()
		})
	})
}

// saveProxyAuth keeps user and password for the custom proxy host: the
// password in memory at once, in the Keychain off the main thread, where
// the item of the host they were for before goes.
func (a *app) saveProxyAuth(host, user, password string) {
	old, keys := a.prefs.Network.ProxyAuthHost, a.keys
	a.updateSettings(func(s *settings.Settings) { s.Network.ProxyUser, s.Network.ProxyAuthHost = user, host })
	a.proxyPassword.Store(&proxySecret{host: host, password: password})
	a.applyNetwork()
	if keys == nil {
		return
	}
	a.async(func() {
		if old != "" && old != host {
			if err := keys.Set(proxyAccount(old), "", ""); err != nil {
				log.Printf("network: the old proxy's password: %v", err)
			}
		}
		if err := keys.Set(proxyAccount(host), "Voxt proxy "+host, password); err != nil {
			log.Printf("network: the proxy's password: %s", strings.ReplaceAll(err.Error(), password, "[password]"))
			a.post(func() { a.toast(i18n.T("Could not save the proxy's password in the Keychain.")) })
		}
	})
}

// applyNetwork gives the downloads and the remote services the network
// of the settings again (a password changed, which the settings do not
// show).
func (a *app) applyNetwork() {
	if a.models != nil {
		a.models.SetNetwork(a.networkFor(a.prefs))
	}
	a.syncRemote()
}
