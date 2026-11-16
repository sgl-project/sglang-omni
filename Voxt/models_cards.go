package main

// The Speech models page's cards. All catalogue, family, download and
// feature-selection decisions live in models_view_state.go.

import (
	"strconv"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

func modelCards(c *ui.Context, cards []modelCardViewState, actions modelsActions) {
	for _, model := range cards {
		modelCard(c, model, actions)
	}
}

func modelCard(c *ui.Context, model modelCardViewState, actions modelsActions) {
	t := c.Theme()
	colors := tok(t)
	card(c.Key(model.Key)).Children(func() {
		if model.FamilyName != "" {
			ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Wrap().Children(func() {
				ui.Icon(c, icons.Soundwave.Linear).Size(px(t, 20), px(t, 20)).TextColor(colors.Text2).Shrink(0)
				ui.Text(c, model.FamilyName).FontWeight(600)
				caption(c, model.CountText)
			})
			modelLanguages(c, model.Languages)
		}
		for _, row := range model.Rows {
			if model.FamilyName != "" {
				ui.Box(c.Key("line-" + row.ID)).Height(1).Background(colors.Line).Shrink(0)
				ui.Column(c.Key("model-" + row.ID)).Gap(px(t, sp3)).Children(func() { modelRow(c, row, false, actions) })
			} else {
				modelRow(c, row, true, actions)
			}
		}
	})
}

// modelLanguages is the line saying which languages a model knows.
func modelLanguages(c *ui.Context, languages string) {
	t := c.Theme()
	if languages != "" {
		ui.Row(c).Gap(px(t, 4)).AlignItems(ui.Center).Children(func() {
			ui.Icon(c, icons.Global.Linear).Size(px(t, 14), px(t, 14)).TextColor(tok(t).Text2).Shrink(0)
			caption(c, languages).MinWidth(0)
		})
	}
}

func hiddenModelCards(c *ui.Context, cards []modelCardViewState, open *bool, actions modelsActions) {
	if len(cards) == 0 {
		return
	}
	collapsible(c.Key("models-hidden"), i18n.T("Hidden models"), open, func() { caption(c, strconv.Itoa(len(cards))) }, func() {
		modelCards(c, cards, actions)
	})
}

// modelRow is one model: its name, size and uses, what it is good at, its
// languages, labels, and available actions.
func modelRow(c *ui.Context, row modelRowViewState, alone bool, actions modelsActions) {
	t := c.Theme()
	colors := tok(t)
	icon := icons.Soundwave
	if row.Kind == "llm" {
		icon = icons.ChatRoundLine
	}
	ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Start).Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp2)).Children(func() {
			ui.Column(c).Gap(px(t, 2)).Children(func() {
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
					if alone {
						ui.Icon(c, icon.Linear).Size(px(t, 20), px(t, 20)).TextColor(colors.Text2).Shrink(0)
					}
					ui.Text(c, row.Name).FontWeight(600)
					if row.Default {
						tagChip(c, i18n.T("Default"))
					}
					if row.Installed {
						statusMark(c, icons.CheckCircle, colors.Success, i18n.T("Installed"))
					}
				})
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
					meta := row.SizeText
					if row.UsedBy != "" {
						meta += " · " + row.UsedBy
					}
					caption(c, meta)
					if row.MemoryText != "" {
						ui.Row(c.Key("loaded")).Gap(px(t, 6)).AlignItems(ui.Center).Children(func() {
							statusDot(c, colors.Accent)
							caption(c, row.MemoryText)
						})
					}
				})
				if row.FamilyText != "" {
					caption(c.Key("family"), row.FamilyText)
				}
			})
			ui.Text(c, row.About)
			if alone {
				modelLanguages(c, row.Languages)
			}
			if len(row.Tags) > 0 {
				ui.Row(c.Key("tags")).Gap(px(t, 6)).AlignItems(ui.Center).Wrap().Children(func() {
					for _, tag := range row.Tags {
						tagChip(c, tag)
					}
				})
			}
		})
		ui.Row(c.Key("actions")).Gap(px(t, sp2)).Shrink(0).Children(func() {
			modelActions(c, row.Actions, actions)
		})
	})
	modelProgress(c, "model-err", row.ID, row.Progress, row.Error, actions)
	if row.Draft != nil {
		draftRow(c, row.Draft, actions)
	}
}

func modelProgress(c *ui.Context, key, modelID string, progress *modelProgressViewState, downloadError *modelErrorViewState, actions modelsActions) {
	if progress != nil {
		ui.Progress(c, progress.Fraction).Label(progress.Label)
	}
	if downloadError != nil {
		notice(c, key, noticeError, downloadError.Message, downloadError.Detail, func() {
			if ui.Button(c.Key("retry"), downloadError.RetryLabel).Clicked() {
				actions.downloadModel(modelID)
			}
		}, nil)
	}
}

func modelActions(c *ui.Context, actions []modelActionViewState, intents modelsActions) {
	for _, action := range actions {
		clicked := false
		switch action.Style {
		case "ghost":
			clicked = ghostButton(c.Key(string(action.Kind)), action.Label).Label(action.AccessibilityLabel).Tooltip(action.Tooltip).Disabled(action.Disabled).Clicked()
		case "secondary-icon":
			clicked = secondaryIconButton(c.Key(string(action.Kind)), icons.DownloadMinimalistic, action.Label).Label(action.AccessibilityLabel).Disabled(action.Disabled).Clicked()
		default:
			clicked = ui.Button(c.Key(string(action.Kind)), action.Label).Label(action.AccessibilityLabel).Disabled(action.Disabled).Clicked()
		}
		if !clicked {
			continue
		}
		switch action.Kind {
		case modelActionHide:
			intents.setModelHidden(action.ModelID, true)
		case modelActionShow:
			intents.setModelHidden(action.ModelID, false)
		case modelActionCancel:
			intents.cancelModel(action.ModelID)
		case modelActionDelete:
			intents.askDeleteModel(action.ModelID)
		case modelActionDiscard:
			intents.discardModelDownload(action.ModelID)
		case modelActionUse:
			intents.useModel(action.ModelID)
		case modelActionDownload:
			intents.downloadModel(action.ModelID)
		}
	}
}

func draftRow(c *ui.Context, draft *modelDraftViewState, actions modelsActions) {
	t := c.Theme()
	colors := tok(t)
	ui.Row(c.Key("draft")).Gap(px(t, sp3)).AlignItems(ui.Start).Margin(px(t, sp2), 0, 0, px(t, 32)).Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp1)).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
				ui.Text(c, i18n.T("Faster transcription")).FontWeight(600)
				infoTip(c.Key("how"), draft.Help)
				if draft.Installed {
					statusMark(c, icons.CheckCircle, colors.Success, i18n.T("Installed"))
				}
			})
			if draft.NeedsDownload {
				caption(c.Key("needs"), i18n.T("Download the draft model to use this."))
			}
			caption(c.Key("size"), draft.SizeText)
		})
		ui.Row(c.Key("actions")).Gap(px(t, sp2)).Shrink(0).AlignItems(ui.Center).Children(func() {
			modelActions(c, draft.Actions, actions)
			on := draft.On
			if ui.Switch(c.Key("faster"), &on).Label(draft.SwitchLabel).Changed() {
				actions.setModelFaster(draft.ParentID, on)
			}
		})
	})
	if draft.Progress != nil || draft.Error != nil {
		ui.Column(c.Key("draft-progress")).Margin(0, 0, 0, px(t, 32)).Children(func() {
			modelProgress(c, "draft-err", draft.ModelID, draft.Progress, draft.Error, actions)
		})
	}
}
