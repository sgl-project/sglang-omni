package main

// The Language models page (/language-models; Auralink's Model services page,
// language models only): a card per OpenAI-compatible service, those set
// up first, each with its state, model, saved key and a connection test.
// A card opens the service's dialog (remote_dialog.go): its key, address
// and model. A feature uses a remote model once the user chooses it on
// its page (feature_models.go), or when no local language model is
// installed. What to build and test is internal/providers'.

import (
	"net/url"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func (a *app) remotePage(c *ui.Context) {
	t := c.Theme()
	pageHeader(c, i18n.T("Language models"), nil)
	ui.Scroll(c.Key("remote-scroll")).Grow(1).MinHeight(0).Children(func() {
		ui.Column(c).Padding(0, px(t, sp6), px(t, sp6)).Gap(px(t, sp4)).MaxWidth(px(t, 880)).Children(func() {
			caption(c, i18n.T("Transcribe's clean-up, Translate and Rewrite use a language model of a service you set up here, over its OpenAI-compatible API. To keep text on this Mac, run a model with Ollama, LM Studio or oMLX (free); or buy an API key from a provider. Voxt sends the service the recognised or selected text, never your audio. Keys stay in this Mac's Keychain."))
			a.providerCards(c)
		})
	})
	a.providerDialogView(c) // remote_dialog.go
}

// providerCards lists the services: those set up first, then the rest.
func (a *app) providerCards(c *ui.Context) {
	t := c.Theme()
	var mine, rest []llm.Preset
	for _, pr := range llm.Presets() {
		if _, ok := providers.Setup(a.prefs, pr.ID); ok {
			mine = append(mine, pr)
		} else {
			rest = append(rest, pr)
		}
	}
	room := min(a.pageWidth(c), px(t, 880)) - px(t, 2*sp6) - 1 // the column's padding
	// Cards at least 260 wide: two columns in the default window (800
	// wide) and the smallest.
	cols := max(1, int((room+px(t, sp3))/(px(t, 260)+px(t, sp3))))
	cardW := (room - px(t, sp3)*float32(cols-1)) / float32(cols)
	grid := func(key string, list []llm.Preset) {
		ui.Row(c.Key(key)).Wrap().Gap(px(t, sp3)).AlignItems(ui.Start).Children(func() {
			for _, pr := range list {
				a.providerCard(c, pr, cardW)
			}
		})
	}
	if len(mine) > 0 {
		sectionTitle(c, i18n.T("Set up"))
		grid("remote-mine", mine)
	}
	title := i18n.T("Services")
	if len(mine) > 0 {
		title = i18n.T("Available")
	}
	sectionTitle(c, title)
	grid("remote-rest", rest)
}

// providerState is a card's state line: its dot's colour and text.
func (a *app) providerState(c *ui.Context, p settings.Provider, configured bool) (ui.Color, string) {
	k := tok(c.Theme())
	if !configured {
		return k.G[5], i18n.T("Not set up")
	}
	switch providers.Missing(p, a.hasKey(p.ID)) {
	case providers.MissingKey:
		return k.Warning, i18n.T("No key saved")
	case providers.MissingEndpoint:
		return k.Warning, i18n.T("No address")
	case providers.MissingModel:
		return k.Warning, i18n.T("No model")
	}
	if tr, ok := a.provTests[p.ID]; ok && !tr.running && tr.res.Err != nil {
		return k.Danger, i18n.T("Test failed")
	}
	return k.Success, i18n.T("Ready")
}

// providerCard is a service's card: its state dot, name and model; once
// set up, its key's mask, its last test and a Test button. A click opens
// its dialog.
func (a *app) providerCard(c *ui.Context, pr llm.Preset, w float32) {
	t := c.Theme()
	k := tok(t)
	p, configured := providers.Setup(a.prefs, pr.ID)
	name := uitext.PresetName(pr.ID)
	dot, state := a.providerState(c, p, configured)
	missing := providers.Missing(p, a.hasKey(p.ID))
	b := ui.ButtonBase(c.Key("remote-card-"+pr.ID)).Width(w).Padding(px(t, sp3), px(t, sp4)).Radius(px(t, radiusCard)).Border(1, k.Line).
		Label(name).Tooltip(i18n.T("Set up %s", name)).AlignItems(ui.Stretch).Shrink(0)
	if b.Hovered() {
		b.Background(k.Subtle)
	}
	b.Children(func() {
		ui.Column(c).Gap(px(t, 6)).Grow(1).MinWidth(0).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Start).MinHeight(px(t, 22)).Children(func() {
				ui.Row(c).Height(px(t, 22)).AlignItems(ui.Center).Shrink(0).Children(func() { statusDot(c, dot) })
				ui.Text(c, name).FontWeight(600).Grow(1).MinWidth(0).MaxLines(2).Ellipsis("…").Margin(px(t, 2), 0, 0, 0)
				ui.Row(c).Height(px(t, 22)).AlignItems(ui.Center).Shrink(0).Children(func() { tagChip(c, whereText(pr, p)) })
			})
			model := providers.Model(p)
			if model == "" {
				model = i18n.T("Model not chosen")
			}
			truncText(c, model).FontSize(px(t, fsCaption)).TextColor(k.Text2)
			if !configured {
				return // the rest shows once it is set up
			}
			ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).Height(px(t, 30)).Children(func() {
				mask, read := a.keyMasks[p.ID]
				keyText, keyCol := mask, k.Text2
				switch {
				case mask == "" && !read:
					keyText = "…"
				case mask == "" && missing == providers.MissingKey:
					keyText, keyCol = state, k.Warning
				case mask == "":
					keyText, keyCol = i18n.T("No key"), k.Text3
				}
				ui.Icon(c, icons.KeyMinimalisticSquare.Linear).Size(px(t, 14), px(t, 14)).TextColor(keyCol).Shrink(0)
				ui.Text(c, keyText).FontSize(px(t, fsCaption)).TextColor(keyCol).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
				switch missing {
				case "":
					if tr, ok := a.provTests[p.ID]; ok {
						testStatus(c, tr, false)
					}
					if ghostButton(c.Key("remote-test-"+p.ID), i18n.T("Test")).Label(i18n.T("Test %s", name)).Disabled(a.provTests[p.ID].running).Clicked() {
						kr := a.keys
						a.testProvider(p, func() (string, error) { return kr.Get(p.ID) })
					}
				case providers.MissingKey:
				default:
					ui.Text(c, state).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine().Shrink(0)
				}
			})
		})
	})
	if b.Clicked() {
		a.openProviderDialog(pr.ID)
	}
}

// whereText is a service's label: on this Mac (a server on the loopback:
// Ollama, LM Studio, oMLX, a custom one) or remote.
func whereText(pr llm.Preset, p settings.Provider) string {
	if resolved, err := providers.Config(p).Resolve(); err == nil {
		pr = resolved
	}
	if u, err := url.Parse(pr.BaseURL); err == nil && u.Host != "" && llm.IsLoopback(u.Hostname()) {
		return i18n.T("On this Mac")
	}
	return i18n.T("Remote")
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
