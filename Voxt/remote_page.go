package main

// The Language models section in Settings (/settings/language-models;
// /language-models before) and the remote service cards shared with the
// Speech models section. Rendering depends only on view state,
// actions, local widget state and the UI kit.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// remoteSettingsView is the Language models section in Settings
// (/settings/language-models); the settings dialog gives it its scroll and
// title, and shell.go draws the provider dialog over the window.
func remoteSettingsView(c *ui.Context, state *remoteViewState, actions remoteActions, local *remoteLocalUI, width float32) {
	if state.BackBar != nil {
		notice(c, "ob-back", noticeInfo, state.BackBar.Text, "", func() {
			if ui.Button(c.Key("ob-back-open"), i18n.T("Back to the setup guide")).Clicked() {
				actions.openOnboarding()
			}
		}, actions.dismissOnboardingBack)
	}
	caption(c, i18n.T("Voxt sends the language model the recognised or selected text, never your audio."))
	llmWaitView(c, &state.Wait, actions, local.wait)
	on := state.Switch
	settingRow(c, "llm-switch", i18n.T("Retry with another model on this Mac"), i18n.T("If the language model does not answer, Voxt tries once more with another model you set up on this Mac. Translate tries its own fallback model first. Voxt never switches to an external API on its own."), func() {
		if ui.Switch(c.Key("llm-switch-switch"), &on).Label(i18n.T("Retry with another model on this Mac")).Changed() {
			actions.setLLMSwitch(on)
		}
	})
	remoteProviderCards(c, &state.Cards, actions, width)
	if ui.Button(c.Key("remote-debug"), i18n.T("Try prompts on a model…")).AlignSelf(ui.Start).Clicked() {
		actions.openLLMDebug()
	}
}

// settingsSectionWidth is the width of a Settings section's content: the
// dialog less its sidebar column.
func settingsSectionWidth(c *ui.Context) float32 {
	t := c.Theme()
	w, h := c.Size()
	dw, _ := settingsDialogSize(t, w, h)
	return dw - px(t, settingsColumnW)
}

// remoteProviderCards lists configured services first, then loopback and
// online services. Card width and grid layout stay on the UI side.
func remoteProviderCards(c *ui.Context, state *remoteCardsViewState, actions remoteCardsActions, width float32) {
	t := c.Theme()
	room := width - px(t, 2*sp6) - 1 // the content's padding
	// Cards at least 260 wide, in as many columns as the section fits.
	cols := max(1, int((room+px(t, sp3))/(px(t, 260)+px(t, sp3))))
	cardWidth := (room - px(t, sp3)*float32(cols-1)) / float32(cols)
	for _, group := range state.Groups {
		sectionTitle(c, group.Title)
		if group.Caption != "" {
			caption(c, group.Caption)
		}
		ui.Row(c.Key(group.Key)).Wrap().Gap(px(t, sp3)).AlignItems(ui.Start).Children(func() {
			for _, card := range group.Cards {
				remoteProviderCard(c, card, actions, cardWidth)
			}
		})
	}
}

func remoteDotColor(c *ui.Context, status string) ui.Color {
	colors := tok(c.Theme())
	switch status {
	case remoteStatusWarning:
		return colors.Warning
	case remoteStatusDanger:
		return colors.Danger
	case remoteStatusSuccess:
		return colors.Success
	}
	return colors.G[5]
}

func remoteKeyColor(c *ui.Context, status string) ui.Color {
	colors := tok(c.Theme())
	switch status {
	case remoteStatusWarning:
		return colors.Warning
	case "muted":
		return colors.Text3
	}
	return colors.Text2
}

// remoteProviderCard is one service's display and its open/test intents.
func remoteProviderCard(c *ui.Context, state remoteProviderCardState, actions remoteCardsActions, width float32) {
	t := c.Theme()
	colors := tok(t)
	button := ui.ButtonBase(c.Key("remote-card-"+state.Account)).Width(width).Padding(px(t, sp3), px(t, sp4)).Radius(px(t, radiusCard)).Border(1, colors.Line).
		Label(state.Name).Tooltip(i18n.T("Set up %s", state.Name)).AlignItems(ui.Stretch).Shrink(0).
		Role(ui.RoleGroup)
	if button.Hovered() {
		button.Background(colors.Subtle)
	}
	keyRow := func() {
		ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).Height(px(t, 30)).Children(func() {
			ui.Icon(c, icons.KeyMinimalisticSquare.Linear).Size(px(t, 14), px(t, 14)).TextColor(remoteKeyColor(c, state.KeyStatus)).Shrink(0)
			ui.Text(c, state.KeyText).FontSize(px(t, fsCaption)).TextColor(remoteKeyColor(c, state.KeyStatus)).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
			if state.Missing == "" {
				if state.Test != nil {
					remoteTestStatus(c, *state.Test, false)
				}
				if ghostButton(c.Key("remote-test-"+state.Account), i18n.T("Test")).Label(i18n.T("Test %s", state.Name)).Shrink(0).Disabled(state.TestDisabled).Clicked() {
					actions.testRemoteProvider(state.Kind, state.ID)
				}
				if ghostButton(c.Key("remote-assign-"+state.Account), i18n.T("Apply to")).Label(i18n.T("Apply %s to features", state.Name)).Shrink(0).Tooltip(i18n.T("Choose the features that use this service")).Clicked() {
					actions.openModelAssignment(state.Kind, state.ID)
				}
			} else if ghostButton(c.Key("remote-complete-"+state.Account), i18n.T("Complete setup")).Label(i18n.T("Complete setup of %s", state.Name)).Clicked() {
				actions.openRemoteProvider(state.Kind, state.ID)
			}
		})
	}
	button.Children(func() {
		if state.Kind == "asr" {
			ui.Column(c).Gap(px(t, sp2)).Grow(1).MinWidth(0).Children(func() {
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Start).MinHeight(px(t, 22)).Children(func() {
					ui.Row(c).Height(px(t, 22)).AlignItems(ui.Center).Shrink(0).Children(func() { statusDot(c, remoteDotColor(c, state.Status)) })
					ui.Text(c, state.Name).FontWeight(600).Grow(1).MinWidth(0).MaxLines(2).Ellipsis("…").Margin(px(t, 2), 0, 0, 0)
				})
				meta := state.Model
				if state.Used != "" {
					meta += " · " + state.Used
				}
				caption(c, meta).MinWidth(0)
				if state.Configured {
					keyRow()
				}
				ui.Row(c.Key("tags")).Gap(px(t, 6)).AlignItems(ui.Center).Wrap().Children(func() {
					tagChip(c, state.Where)
					if state.Realtime {
						tagChip(c, i18n.T("Realtime"))
					}
				})
			})
			return
		}
		ui.Column(c).Gap(px(t, 6)).Grow(1).MinWidth(0).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Start).MinHeight(px(t, 22)).Children(func() {
				ui.Row(c).Height(px(t, 22)).AlignItems(ui.Center).Shrink(0).Children(func() { statusDot(c, remoteDotColor(c, state.Status)) })
				ui.Text(c, state.Name).FontWeight(600).Grow(1).MinWidth(0).MaxLines(2).Ellipsis("…").Margin(px(t, 2), 0, 0, 0)
				ui.Row(c).Height(px(t, 22)).AlignItems(ui.Center).Shrink(0).Children(func() { tagChip(c, state.Where) })
			})
			truncText(c, state.Model).FontSize(px(t, fsCaption)).TextColor(colors.Text2)
			if !state.Configured {
				return
			}
			if state.Used != "" {
				truncText(c, state.Used).FontSize(px(t, fsCaption)).TextColor(colors.Text2)
			}
			keyRow()
		})
	})
	if button.Clicked() {
		actions.openRemoteProvider(state.Kind, state.ID)
	}
}

// remoteTestStatus renders a predetermined connection-test state and its
// redacted detail. The view chooses the dialog's long or the card's short form.
func remoteTestStatus(c *ui.Context, state remoteTestViewState, long bool) {
	t := c.Theme()
	colors := tok(t)
	text, tooltip := state.Text, state.Tooltip
	if long {
		text, tooltip = state.LongText, state.LongTooltip
	}
	switch state.State {
	case "running":
		ui.Spinner(c).Size(px(t, 14), px(t, 14)).Shrink(0)
		if long {
			caption(c, text).SingleLine().Ellipsis("…").MinWidth(0)
		}
	case "success":
		statusMark(c, icons.CheckCircle, colors.Success, text).Shrink(1).Tooltip(tooltip)
	case "failure":
		if !long {
			statusMark(c, icons.CloseCircle, colors.Danger, text).Shrink(0).Tooltip(tooltip)
			return
		}
		statusMark(c, icons.CloseCircle, colors.Danger, text).Shrink(1).Tooltip(tooltip)
		infoTip(c, tooltip).Shrink(0)
	}
}
