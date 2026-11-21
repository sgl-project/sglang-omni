package main

// The setup guide's models step (onboarding.go): the recommended pack (a
// speech model for this Mac's memory with its voice detection), which the
// footer's primary button downloads, then the language model: the one set
// up, or a way to Settings → Language models (they are remote). The models another
// ActNow app already put on this Mac come first (family spec §7.10.1): a
// card offers to use them ("Use them" claims them for VoxtNew, nothing
// downloads), and while the recommended speech model is missing and one
// of its alternatives is there, the guide offers that one for now, the
// download staying the default (it is a downgrade).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// obModelsStep is the step: the reuse card, a row by kind, and the way to
// the other models. The download itself is the footer's; under the reuse
// card, whose Use them is the primary button, it stays here.
func obModelsStep(c *ui.Context, state onboardingModelsViewState, actions onboardingActions) {
	t := c.Theme()
	if state.ReuseShown {
		obReuseCard(c, state, actions)
	}
	obModelRow(c, state.Speech, actions)
	if state.Speech.Downloading {
		caption(c, i18n.T("The model downloads in the background, so you can continue. It is ready when the bar fills."))
	}
	ui.Row(c).Margin(px(t, sp1), 0, 0, 0).Children(func() {
		if ghostButton(c.Key("ob-choose-other"), i18n.T("Choose another model…")).Clicked() {
			actions.obLeave(modelsPath)
		}
	})
	obLanguageRow(c, state, actions)
}

// obLanguageRow is the guide's language model: the one the features use,
// or the way to set one up (they are remote: a service's API, or a
// server such as Ollama on this Mac).
func obLanguageRow(c *ui.Context, state onboardingModelsViewState, actions onboardingActions) {
	t := c.Theme()
	k := tok(t)
	ui.Column(c.Key("ob-models-llm")).Gap(px(t, sp1)).Margin(px(t, sp4), 0, 0, 0).Children(func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			sectionTitleText(c, i18n.T("Language model"))
			if state.LanguageReady {
				statusMark(c, icons.CheckCircle, k.Success, i18n.T("Ready: %s", state.LanguageName))
			}
		})
		if state.LanguageReady {
			return
		}
		caption(c, i18n.Platform("Cleans up, translates and rewrites the text. Run one on this Mac with Ollama, LM Studio or oMLX, or use a service's API key.", "Cleans up, translates and rewrites the text. Run one on this PC with Ollama, LM Studio or oMLX, or use a service's API key."))
		ui.Row(c).Children(func() {
			if ui.Button(c.Key("ob-llm-setup"), i18n.T("Set up a language model")).Clicked() {
				actions.obLeave(remotePath)
			}
		})
	})
}

// obReuseCard is the card of the models already on this Mac: how many,
// who installed them and their size, their names, Use them and Choose
// models….
func obReuseCard(c *ui.Context, state onboardingModelsViewState, actions onboardingActions) {
	t := c.Theme()
	k := tok(t)
	card(c.Key("ob-reuse")).Children(func() {
		ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Start).Children(func() {
			ui.Icon(c, icons.CheckCircle.Linear).Size(px(t, 20), px(t, 20)).TextColor(k.Success).Shrink(0)
			ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp2)).Children(func() {
				ui.Column(c).Gap(px(t, 2)).Children(func() {
					ui.Text(c, i18n.PlatformN(len(state.ReuseIDs), "%d model is already on this Mac", "%d models are already on this Mac", "%d model is already on this PC", "%d models are already on this PC", len(state.ReuseIDs))).FontWeight(600)
					caption(c, state.ReuseSubtitle)
				})
				ui.Text(c, state.ReuseNames).FontSize(px(t, fsCaption))
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
					if primaryButton(c.Key("ob-reuse-use"), i18n.T("Use them")).Clicked() {
						actions.obUse(state.ReuseIDs...)
					}
					if ghostButton(c.Key("ob-reuse-choose"), i18n.T("Choose models…")).Clicked() {
						actions.obLeave(modelsPath)
					}
				})
			})
		})
	})
}

// obModelRow is the guide's speech model: the one VoxtNew uses (ready),
// or the recommended one: on this Mac already (the card's), or its size
// (what is still to download, under the card), its download, and the
// alternative on this Mac.
func obModelRow(c *ui.Context, state onboardingSpeechViewState, actions onboardingActions) {
	t := c.Theme()
	k := tok(t)
	ui.Column(c.Key("ob-models-asr")).Gap(px(t, sp1)).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			sectionTitleText(c, i18n.T("Speech recognition"))
			switch {
			case state.Ready:
				statusMark(c, icons.CheckCircle, k.Success, i18n.T("Ready: %s", state.Name))
			case state.OnDisk:
				ui.Text(c, state.Name).FontWeight(600)
				statusMark(c, icons.CheckCircle, k.Text2, state.InstalledBy)
			default:
				ui.Text(c, state.Name).FontWeight(600)
				tagChip(c, state.SizeLabel)
			}
		})
		if state.Ready || state.OnDisk {
			return
		}
		caption(c, state.About)
		if state.Alternative != nil {
			obAlternativeLine(c, *state.Alternative, actions)
		}
		if state.Downloading {
			ui.Progress(c, state.DownloadFraction).Label(i18n.T("Downloading %s", state.Name))
		}
		if state.Error != nil && !state.Downloading {
			notice(c, "ob-model-err", noticeError, state.Error.Text, state.Error.More, nil, nil)
		}
	})
	if state.ShowDownload {
		ui.Row(c).Margin(px(t, sp2), 0, 0, 0).Children(func() {
			if secondaryIconButton(c.Key("ob-download"), icons.DownloadMinimalistic, state.DownloadLabel).Clicked() {
				actions.downloadModel(state.ID)
			}
		})
	}
}

// obAlternativeLine offers an alternative on this Mac for the recommended
// model: Download the recommended model (the default: the alternative is
// a downgrade) or use the alternative for now.
func obAlternativeLine(c *ui.Context, state onboardingAlternativeViewState, actions onboardingActions) {
	t := c.Theme()
	fillCard(c.Key("ob-alt-"+state.WantID)).Padding(px(t, sp2), px(t, sp3)).Gap(px(t, sp2)).Children(func() {
		ui.Text(c, state.Message).FontSize(px(t, fsCaption))
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			if secondaryIconButton(c.Key("ob-alt-download"), icons.DownloadMinimalistic, state.DownloadLabel).Clicked() {
				actions.downloadModel(state.WantID)
			}
			if ghostButton(c.Key("ob-alt-use"), state.UseLabel).Clicked() {
				actions.obUse(state.UseID)
			}
		})
	})
}
