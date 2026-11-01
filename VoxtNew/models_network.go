package main

// The Local models page's Network… dialog (voxtnew://models/network):
// how model downloads reach the Hub, two of the ActNow family's shared
// settings (family spec §9.1): the proxy (automatic: the environment's,
// else the Mac's; none; or a proxy's URL) and the download source
// (Hugging Face and hf-mirror.com raced, or one of them). While VoxtNew
// follows the family (the default), a change applies to every ActNow app
// (family.go writes it to the family); "Use a different setting for this
// app" keeps it to VoxtNew. A key whose value differed from the family's
// at the first start shows "Use theirs" once (spec §9.4).

import (
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	famsettings "github.com/hehehai/actnowkit/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/actnow"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// networkPath is the Network dialog's deep link (actnow.Links).
const networkPath = modelsPath + "/network"

// The proxy's modes, as the dialog's segments.
const (
	proxyAuto = iota
	proxyNone
	proxyCustom
)

// The download sources, as the dialog's segments (a list of URLs another
// app set is a fourth, shown while it is the value).
var networkSources = []string{famsettings.EndpointsAuto, famsettings.EndpointsOfficial, famsettings.EndpointsMirror}

// networkForm is the Network dialog's fields.
type networkForm struct {
	proxyMode int
	proxyURL  string
	proxyErr  string
	source    int    // index in networkSources, or len(networkSources): sourceURLs
	sourceURL string // a list of endpoints another app set ("" none); settings.json does not say which app
}

// openNetwork opens the Network dialog with the values VoxtNew uses.
func (a *app) openNetwork() {
	a.netForm = networkForm{}
	a.fillNetwork(actnow.KeyProxy)
	a.fillNetwork(actnow.KeyHFEndpoints)
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
			f.proxyErr = i18n.T("Enter a proxy's address that starts with http://, https:// or socks5://, like http://127.0.0.1:7890.")
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

// networkDialog is the Network dialog.
func (a *app) networkDialog(c *ui.Context) {
	if !a.netOpen {
		return
	}
	t := c.Theme()
	f := &a.netForm
	open := true
	dialogFrame(c, "net-dlg", &open, i18n.T("Network"), i18n.T("How model downloads reach Hugging Face"), 540, func() {
		ui.Text(c, i18n.T("Proxy")).FontWeight(600)
		segmentedFit(c.Key("net-proxy"), &f.proxyMode, i18n.T("Automatic"), i18n.T("No proxy"), i18n.T("Custom")).Label(i18n.T("Proxy"))
		switch f.proxyMode {
		case proxyAuto:
			caption(c, i18n.T("The proxy in the Mac's network settings, or the one the environment names."))
		case proxyNone:
			caption(c, i18n.T("Downloads connect to the server directly."))
		default:
			ui.TextInput(c.Key("net-proxy-url"), &f.proxyURL).Label(i18n.T("Proxy address")).Placeholder("http://127.0.0.1:7890").Font("monospace").Width(px(t, 320))
			if f.proxyErr != "" {
				caption(c, f.proxyErr).TextColor(t.Danger)
			}
		}
		a.networkFamilyLine(c, actnow.KeyProxy, i18n.T("Other ActNow apps use a different proxy."))

		ui.Box(c).Height(px(t, sp2)).Shrink(0)
		ui.Text(c, i18n.T("Download source")).FontWeight(600)
		labels := []string{i18n.T("Automatic"), i18n.T("Hugging Face only"), i18n.T("hf-mirror.com only")}
		if f.sourceURL != "" {
			labels = append(labels, i18n.T("Other"))
		}
		segmentedFit(c.Key("net-source"), &f.source, labels...).Label(i18n.T("Download source"))
		switch {
		case f.source == 0:
			caption(c, i18n.T("Hugging Face and its mirror hf-mirror.com, whichever answers first. Every file is checked, so either is safe."))
		case f.source < len(networkSources):
			caption(c, i18n.T("Only this server: when it cannot be reached, downloads fail."))
		default:
			caption(c, i18n.T("Custom servers: %s", f.sourceURL))
		}
		a.networkFamilyLine(c, actnow.KeyHFEndpoints, i18n.T("Other ActNow apps use a different download source."))
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("net-cancel"), i18n.T("Cancel")).Clicked() {
			open = false
		}
		if primaryButton(c.Key("net-save"), i18n.T("Save")).Clicked() && a.saveNetwork() {
			open = false
		}
	})
	if !open {
		a.closeNetwork()
	}
}

// networkFamilyLine says under a shared setting whether it applies to
// every ActNow app, with the link that changes it (spec §9.3), or, once
// after the first start, that the family uses another value (§9.4). With
// no other ActNow app installed it says nothing but the conflict (which
// another app's value implies) or a newer app's read-only directory.
func (a *app) networkFamilyLine(c *ui.Context, key, other string) {
	t := c.Theme()
	follows := a.prefs.Family.Follows(key)
	if slices.Contains(a.prefs.Family.Conflicts, key) && !follows {
		notice(c, "net-conflict-"+keySuffix(key), noticeInfo, other, "", func() {
			if ui.Button(c.Key("theirs"), i18n.T("Use theirs")).Clicked() {
				a.followNetwork(key, true)
			}
		}, nil)
		return
	}
	if !a.familyOthers() && !(follows && a.fam.ReadOnly()) {
		return
	}
	ui.Row(c.Key("net-family-" + keySuffix(key))).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
		if follows && a.fam.ReadOnly() {
			caption(c, i18n.T("Another ActNow app updated the shared format. Update this app to change these settings."))
			return
		}
		if follows {
			caption(c, i18n.T("Applies to all ActNow apps"))
			if ui.Link(c.Key("own"), i18n.T("Use a different setting for this app"), "").FontSize(px(t, fsCaption)).Clicked() {
				a.followNetwork(key, false)
			}
			return
		}
		caption(c, i18n.T("Only VoxtNew uses this setting."))
		if ui.Link(c.Key("follow"), i18n.T("Use the same setting as the other ActNow apps"), "").FontSize(px(t, fsCaption)).Clicked() {
			a.followNetwork(key, true)
		}
	})
}

// keySuffix is a shared key as a part of an element key.
func keySuffix(key string) string { return strings.ReplaceAll(key, ".", "-") }
