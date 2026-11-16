package main

import (
	"slices"
	"strings"

	famsettings "github.com/hehehai/actnowkit/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// The download sources, as the dialog's segments (a list of URLs another
// app set is a fourth, shown while it is the value).
var networkSources = []string{famsettings.EndpointsAuto, famsettings.EndpointsOfficial, famsettings.EndpointsMirror}

// openNetwork opens the Network dialog with the values VoxtNew uses.
func (a *app) openNetwork() {
	a.netForm = networkForm{proxyUser: a.prefs.Network.ProxyUser}
	if secret := a.proxyPassword.Load(); secret != nil && secret.host == a.prefs.Network.ProxyAuthHost {
		a.netForm.proxyPassword = secret.password
	}
	a.fillNetwork(actnow.KeyProxy)
	a.fillNetwork(actnow.KeyHFEndpoints)
	a.netWas = a.netForm
	a.netOpen = true
}

// fillNetwork sets the form's fields of key to the value VoxtNew uses.
func (a *app) fillNetwork(key string) {
	f := &a.netForm
	v := a.fam.Effective(a.prefs, key)
	switch key {
	case actnow.KeyProxy:
		f.proxyErr = ""
		switch v {
		case famsettings.ProxyAuto:
			f.proxyMode, f.proxyURL = proxyAuto, ""
		case famsettings.ProxyDirect:
			f.proxyMode, f.proxyURL = proxyNone, ""
		default:
			f.proxyMode, f.proxyURL = proxyCustom, v
		}
	case actnow.KeyHFEndpoints:
		if v == "" {
			v = famsettings.EndpointsAuto
		}
		f.source, f.sourceURL = slices.Index(networkSources, v), ""
		if f.source < 0 {
			f.source, f.sourceURL = len(networkSources), v
		}
	}
}

// saveNetwork checks the form and makes it VoxtNew's settings (the
// family's, for the keys VoxtNew follows); it reports whether it did.
func (a *app) saveNetwork() bool {
	f := &a.netForm
	proxy := famsettings.ProxyAuto
	switch f.proxyMode {
	case proxyNone:
		proxy = famsettings.ProxyDirect
	case proxyCustom:
		v, err := famsettings.Normalize(actnow.KeyProxy, f.proxyURL)
		if err != nil || v == famsettings.ProxyAuto || v == famsettings.ProxyDirect {
			f.proxyErr = i18n.T("Enter a proxy address that starts with http://, https:// or socks5://, such as http://127.0.0.1:7890.")
			return false
		}
		if strings.ContainsAny(f.proxyPassword, "\"\\\n\r") { // what the Keychain's security tool cannot take (internal/secrets)
			f.proxyErr = i18n.T("The password cannot have quotes, backslashes or line breaks.")
			return false
		}
		proxy = v
	}
	source := f.sourceURL
	if f.source < len(networkSources) {
		source = networkSources[f.source]
	}
	a.updateSettings(func(s *settings.Settings) {
		actnow.SetOwn(s, actnow.KeyProxy, proxy)
		actnow.SetOwn(s, actnow.KeyHFEndpoints, source)
	})
	if f.proxyMode == proxyCustom {
		a.saveProxyAuth(proxyHost(proxy), strings.TrimSpace(f.proxyUser), strings.TrimSpace(f.proxyPassword)) // models_proxy_auth.go
	}
	return true
}

// followNetwork makes VoxtNew follow the family for key (on: "Use
// theirs", "Use the same setting…"), or keep its own value (off).
func (a *app) followNetwork(key string, on bool) {
	fam := a.fam
	a.updateSettings(func(s *settings.Settings) { fam.Follow(s, key, on) })
	a.fillNetwork(key)
}

// closeNetwork closes the dialog; the one-time notices of a different
// value were seen, so they go.
func (a *app) closeNetwork() {
	a.netOpen = false
	if slices.ContainsFunc(a.prefs.Family.Conflicts, isNetworkKey) {
		a.updateSettings(func(s *settings.Settings) {
			s.Family.Conflicts = slices.DeleteFunc(slices.Clone(s.Family.Conflicts), isNetworkKey)
		})
	}
}

func isNetworkKey(key string) bool { return key == actnow.KeyProxy || key == actnow.KeyHFEndpoints }
