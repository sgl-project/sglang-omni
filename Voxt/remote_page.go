package main

// The Language models page (/language-models; Auralink's Model services
// page) and the remote services' cards, which the Speech models page shows
// too (models_cards.go): a card per service, those set up first, each with
// its state, where it runs (on this Mac or remote), model, saved key and a
// connection test; a streaming speech service's card is tagged Realtime. A
// card opens the service's dialog (remote_dialog.go): its key, address and
// model. A feature uses a remote model once the user chooses it on its
// page (feature_models.go), or when no local model of its kind is
// installed. What to build and test is internal/providers'.

import (
	"net/url"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func (a *app) remotePage(c *ui.Context) {
	t := c.Theme()
	pageHeader(c, i18n.T("Language models"), nil)
	ui.Scroll(c.Key("remote-scroll")).Grow(1).MinHeight(0).Children(func() {
		ui.Column(c).Padding(0, px(t, sp6), px(t, sp6)).Gap(px(t, sp4)).MaxWidth(px(t, 880)).Children(func() {
			a.obBackBar(c, remotePath) // onboarding.go: left from the setup guide
			caption(c, i18n.T("Transcribe's clean-up, Translate and Rewrite use the language model you set up here, over its OpenAI-compatible API. Voxt sends it the recognised or selected text, never your audio; keys stay in this Mac's Keychain."))
			a.providerCards(c, local.LLM)
		})
	})
	a.providerDialogView(c) // remote_dialog.go
}

// providerCards lists kind's services: those set up first, then the
// servers that run on this Mac (Voxt 1.x's 本地), then the online
// services (远程), each group with what it takes; the speech services, all
// online by default, as Set up and Online services on the Speech models
// page.
func (a *app) providerCards(c *ui.Context, kind local.Kind) {
	t := c.Theme()
	var mine, here, online []service
	for _, sv := range services(kind) {
		_, set := a.setupOf(kind, sv.id)
		switch {
		case set:
			mine = append(mine, sv)
		case onThisMac(sv.endpoint):
			here = append(here, sv)
		default:
			online = append(online, sv)
		}
	}
	room := min(a.pageWidth(c), px(t, 880)) - px(t, 2*sp6) - 1 // the column's padding
	// Cards at least 260 wide: two columns in the default window (800
	// wide) and the smallest.
	cols := max(1, int((room+px(t, sp3))/(px(t, 260)+px(t, sp3))))
	cardW := (room - px(t, sp3)*float32(cols-1)) / float32(cols)
	grid := func(key string, list []service) {
		ui.Row(c.Key(key)).Wrap().Gap(px(t, sp3)).AlignItems(ui.Start).Children(func() {
			for _, sv := range list {
				a.providerCard(c, sv, cardW)
			}
		})
	}
	if kind == local.ASR {
		if len(mine) > 0 {
			sectionTitle(c, i18n.T("Set up"))
			grid("remote-asr-mine", mine)
		}
		if rest := append(here, online...); len(rest) > 0 {
			sectionTitle(c, i18n.T("Online services"))
			caption(c, i18n.T("Buy an API key from the service, then set it up here; a Whisper server on this Mac needs no key. Your recording goes to the service only for the features you choose it on."))
			grid("remote-asr-online", rest)
		}
		return
	}
	if len(mine) > 0 {
		sectionTitle(c, i18n.T("Set up"))
		grid("remote-mine", mine)
	}
	if len(here) > 0 {
		sectionTitle(c, i18n.T("On this Mac"))
		caption(c, i18n.T("Free, and the text stays here: install Ollama, LM Studio or oMLX, download a model in it, then set it up here."))
		grid("remote-here", here)
	}
	if len(online) > 0 {
		sectionTitle(c, i18n.T("Online services"))
		caption(c, i18n.T("Buy an API key from the service, then set it up here."))
		grid("remote-online", online)
	}
}

// providerState is a card's state line: its dot's colour and text.
func (a *app) providerState(c *ui.Context, kind local.Kind, p settings.Provider, configured bool) (ui.Color, string) {
	k := tok(c.Theme())
	if !configured {
		return k.G[5], i18n.T("Not set up")
	}
	switch a.setupMissing(kind, p) {
	case providers.MissingKey:
		return k.Warning, i18n.T("API key needed")
	case providers.MissingEndpoint:
		return k.Warning, i18n.T("No address")
	case providers.MissingModel:
		return k.Warning, i18n.T("No model")
	}
	if tr, ok := a.provTests[providers.Account(kind, p.ID)]; ok && !tr.running && tr.res.Err != nil {
		return k.Danger, i18n.T("Test failed")
	}
	return k.Success, i18n.T("Ready")
}

// providerCard is a service's card: its state dot, name, where it runs
// and model; once set up, where it is used, its key's mask, its last test
// and a Test button. A speech service's card follows the local speech
// cards' order, its labels at the foot. A click opens its dialog.
func (a *app) providerCard(c *ui.Context, sv service, w float32) {
	t := c.Theme()
	k := tok(t)
	kind, acct := sv.kind, providers.Account(sv.kind, sv.id)
	p, configured := a.setupOf(kind, sv.id)
	dot, state := a.providerState(c, kind, p, configured)
	missing := a.setupMissing(kind, p)
	b := ui.ButtonBase(c.Key("remote-card-"+acct)).Width(w).Padding(px(t, sp3), px(t, sp4)).Radius(px(t, radiusCard)).Border(1, k.Line).
		Label(sv.name).Tooltip(i18n.T("Set up %s", sv.name)).AlignItems(ui.Stretch).Shrink(0).
		Role(ui.RoleGroup) // a group, not a button: assistive technology reaches its Test button too; pressing it still opens the dialog
	if b.Hovered() {
		b.Background(k.Subtle)
	}
	model := setupModel(kind, p)
	if model == "" {
		model = i18n.T("Model not chosen")
	}
	used := ""
	if configured {
		used = a.usedBy(kind, p.ID)
	}
	keyRow := func() {
		ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).Height(px(t, 30)).Children(func() {
			mask, read := a.keyMasks[acct]
			keyText, keyCol := mask, k.Text2
			switch {
			case mask == "" && !read:
				keyText = "…"
			case missing != "":
				keyText, keyCol = state, k.Warning
			case mask == "":
				keyText, keyCol = i18n.T("No key"), k.Text3
			}
			ui.Icon(c, icons.KeyMinimalisticSquare.Linear).Size(px(t, 14), px(t, 14)).TextColor(keyCol).Shrink(0)
			ui.Text(c, keyText).FontSize(px(t, fsCaption)).TextColor(keyCol).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
			switch missing {
			case "":
				if tr, ok := a.provTests[acct]; ok {
					testStatus(c, tr, false)
				}
				if ghostButton(c.Key("remote-test-"+acct), i18n.T("Test")).Label(i18n.T("Test %s", sv.name)).Disabled(a.provTests[acct].running).Clicked() {
					kr := a.keys
					a.testProvider(kind, p, func() (string, error) { return kr.Get(acct) })
				}
			default:
				if ghostButton(c.Key("remote-complete-"+acct), i18n.T("Complete setup")).Label(i18n.T("Complete setup of %s", sv.name)).Clicked() {
					a.openProviderDialog(kind, sv.id)
				}
			}
		})
	}
	b.Children(func() {
		if kind == local.ASR {
			// The local speech cards' order (Voxt 1.x's): the name, then the
			// model and where it is used, the key, and the labels at the foot.
			ui.Column(c).Gap(px(t, sp2)).Grow(1).MinWidth(0).Children(func() {
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Start).MinHeight(px(t, 22)).Children(func() {
					ui.Row(c).Height(px(t, 22)).AlignItems(ui.Center).Shrink(0).Children(func() { statusDot(c, dot) })
					ui.Text(c, sv.name).FontWeight(600).Grow(1).MinWidth(0).MaxLines(2).Ellipsis("…").Margin(px(t, 2), 0, 0, 0)
				})
				meta := model
				if used != "" {
					meta += " · " + used
				}
				caption(c, meta).MinWidth(0)
				if configured {
					keyRow()
				}
				ui.Row(c.Key("tags")).Gap(px(t, 6)).AlignItems(ui.Center).Wrap().Children(func() {
					tagChip(c, whereText(sv, p))
					if sv.speech.Streams(setupModel(kind, p)) {
						tagChip(c, i18n.T("Realtime"))
					}
				})
			})
			return
		}
		ui.Column(c).Gap(px(t, 6)).Grow(1).MinWidth(0).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Start).MinHeight(px(t, 22)).Children(func() {
				ui.Row(c).Height(px(t, 22)).AlignItems(ui.Center).Shrink(0).Children(func() { statusDot(c, dot) })
				ui.Text(c, sv.name).FontWeight(600).Grow(1).MinWidth(0).MaxLines(2).Ellipsis("…").Margin(px(t, 2), 0, 0, 0)
				ui.Row(c).Height(px(t, 22)).AlignItems(ui.Center).Shrink(0).Children(func() { tagChip(c, whereText(sv, p)) })
			})
			truncText(c, model).FontSize(px(t, fsCaption)).TextColor(k.Text2)
			if !configured {
				return // the rest shows once it is set up
			}
			if used != "" {
				truncText(c, used).FontSize(px(t, fsCaption)).TextColor(k.Text2)
			}
			keyRow()
		})
	})
	if b.Clicked() {
		a.openProviderDialog(kind, sv.id)
	}
}

// whereText is a service's label: on this Mac (a server on the loopback:
// Ollama, LM Studio, oMLX, a custom one) or remote.
func whereText(sv service, p settings.Provider) string {
	ep := strings.TrimSpace(p.Endpoint)
	if ep == "" {
		ep = sv.endpoint
	}
	if onThisMac(ep) {
		return i18n.T("On this Mac")
	}
	return i18n.T("Remote")
}

// onThisMac says whether a service's address is a server on this Mac.
func onThisMac(base string) bool {
	u, err := url.Parse(base)
	return err == nil && u.Host != "" && core.IsLoopback(u.Hostname())
}

// testStatus is a connection test's state: a spinner while it runs, a
// green tick and its time once it works, a red mark and a short reason
// when it failed, with what the service said in a tooltip (and an ⓘ,
// when long is set: the dialog's).
func testStatus(c *ui.Context, tr providerTest, long bool) {
	t := c.Theme()
	k := tok(t)
	switch {
	case tr.running:
		ui.Spinner(c).Size(px(t, 14), px(t, 14)).Shrink(0)
		if long {
			caption(c, i18n.T("Testing…")).SingleLine().Ellipsis("…").MinWidth(0)
		}
	case tr.res.At.IsZero():
	case tr.res.Err == nil:
		text := i18n.T("%d ms", tr.res.Latency.Milliseconds())
		if long {
			text = i18n.T("Connection OK · %d ms", tr.res.Latency.Milliseconds())
		}
		statusMark(c, icons.CheckCircle, k.Success, text).Shrink(1).Tooltip(text)
	default:
		msg, _ := uitext.ModelErrorText(tr.res.Err)
		tip := msg + "\n\n" + tr.detail
		if !long {
			statusMark(c, icons.CloseCircle, k.Danger, i18n.T("Failed")).Shrink(0).Tooltip(tip)
			return
		}
		statusMark(c, icons.CloseCircle, k.Danger, msg).Shrink(1).Tooltip(tip)
		infoTip(c, tr.detail).Shrink(0)
	}
}
