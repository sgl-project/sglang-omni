package main

// The Speech models page's Network… dialog. Proxy credentials stay in
// networkForm, which is local widget state and never enters a fixture.

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// networkPath is the Network dialog's deep link (actnow.Links).
const networkPath = modelsPath + "/network"

// The proxy's modes, as the dialog's segments.
const (
	proxyAuto = iota
	proxyNone
	proxyCustom
)

// networkForm is the Network dialog's fields.
type networkForm struct {
	proxyMode int
	proxyURL  string
	proxyErr  string
	source    int    // index in networkSources, or len(networkSources): sourceURLs
	sourceURL string // a list of endpoints another app set ("" none); settings.json does not say which app
	// proxyUser and proxyPassword are a custom proxy's, those saved until
	// changed (shown only while the proxy is custom).
	proxyUser, proxyPassword string
}

func networkDialog(c *ui.Context, state *modelsNetworkViewState, actions modelsActions, local *modelsLocalUI) {
	if !local.netOpen {
		return
	}
	t := c.Theme()
	form := &local.netForm
	open := true
	if dialogFrame(c, "net-dlg", &open, i18n.T("Network"), i18n.T("How model downloads reach Hugging Face"), 540, func() {
		ui.Text(c, i18n.T("Proxy")).FontWeight(600)
		segmentedFit(c.Key("net-proxy"), &form.proxyMode, i18n.T("Automatic"), i18n.T("No proxy"), i18n.T("Custom")).Label(i18n.T("Proxy"))
		switch form.proxyMode {
		case proxyAuto, proxyNone:
			caption(c, state.ProxyCaption)
		default:
			ui.TextInput(c.Key("net-proxy-url"), &form.proxyURL).Label(i18n.T("Proxy address")).Placeholder("http://127.0.0.1:7890").Font("monospace").Width(px(t, 320))
			ui.TextInput(c.Key("net-proxy-user"), &form.proxyUser).Label(i18n.T("Proxy user name")).Placeholder(i18n.T("User name (optional)")).Width(px(t, 320))
			ui.TextInput(c.Key("net-proxy-password"), &form.proxyPassword).Password().Label(i18n.T("Proxy password")).Placeholder(i18n.T("Password (optional)")).Width(px(t, 320))
			caption(c, i18n.T("The password is kept in the Keychain."))
			if state.ProxyError != "" {
				caption(c, state.ProxyError).TextColor(t.Danger)
			}
		}
		networkFamilyLine(c, state.ProxyFamily, actions)

		ui.Box(c).Height(px(t, sp2)).Shrink(0)
		ui.Text(c, i18n.T("Download source")).FontWeight(600)
		segmentedFit(c.Key("net-source"), &form.source, state.SourceLabels...).Label(i18n.T("Download source"))
		caption(c, state.SourceCaption)
		networkFamilyLine(c, state.SourceFamily, actions)
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("net-cancel"), i18n.T("Cancel")).Clicked() {
			open = false
		}
		if primaryButton(c.Key("net-save"), i18n.T("Save")).Clicked() && actions.saveNetwork() {
			open = false
		}
	}).Dismissed() && local.netForm != local.netWas {
		open = true // a stray click or Escape: keep what was typed
	}
	if !open {
		actions.closeNetwork()
	}
}

func networkFamilyLine(c *ui.Context, state modelsNetworkFamilyViewState, actions modelsActions) {
	t := c.Theme()
	switch state.Kind {
	case "":
		return
	case "conflict":
		notice(c, "net-conflict-"+keySuffix(state.Key), noticeInfo, state.Text, "", func() {
			if ui.Button(c.Key("theirs"), state.ActionLabel).Clicked() {
				actions.followNetwork(state.Key, state.Follow)
			}
		}, nil)
		return
	}
	ui.Row(c.Key("net-family-" + keySuffix(state.Key))).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
		caption(c, state.Text)
		if state.Kind == "readonly" {
			return
		}
		key := "own"
		if state.Follow {
			key = "follow"
		}
		if ui.Link(c.Key(key), state.ActionLabel, "").FontSize(px(t, fsCaption)).Clicked() {
			actions.followNetwork(state.Key, state.Follow)
		}
	})
}

// keySuffix is a shared key as a part of an element key.
func keySuffix(key string) string { return strings.ReplaceAll(key, ".", "-") }
