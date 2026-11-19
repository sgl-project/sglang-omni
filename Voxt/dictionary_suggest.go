package main

// Pending dictionary suggestions above the terms list.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

const (
	suggestionRowsShown  = 5
	suggestionRowsHeight = 240
)

func suggestionsSection(c *ui.Context, state *dictionaryViewState, actions dictionaryActions, local *dictionaryLocalUI) {
	if len(state.Suggestions) == 0 {
		return
	}
	t := c.Theme()
	list := state.Suggestions
	if !local.allSuggestions {
		list = list[:min(len(list), suggestionRowsShown)]
	}
	ui.Column(c.Key("dict-suggestions")).Shrink(0).Padding(0, 0, px(t, sp2)).Children(func() {
		ui.Row(c).Padding(0, px(t, sp6)).AlignItems(ui.Center).Children(func() {
			sectionTitleText(c, i18n.T("Suggestions")).Grow(1)
			if len(list) < state.SuggestionCount && ghostButton(c.Key("dict-suggest-all"), i18n.T("All (%d)", state.SuggestionCount)).Clicked() {
				local.allSuggestions = true
			}
		})
		caption(c, i18n.T("Voxt suggests a word here when you correct one it wrote in the %d seconds after inserting, or when Scan history finds one. Nothing goes in the dictionary until you click Add. To stop them, turn off Learn from my corrections.", state.LearnWatchSeconds)).Padding(0, px(t, sp6))
		ui.Scroll(c.Key("dict-suggest-rows")).MaxHeight(px(t, suggestionRowsHeight)).Children(func() {
			for _, suggestion := range list {
				suggestionRow(c, suggestion, actions)
			}
		})
	})
}

func suggestionRow(c *ui.Context, suggestion dictionarySuggestionViewState, actions dictionaryActions) {
	t := c.Theme()
	colors := tok(t)
	row := ui.Row(c.Key("suggest-"+suggestion.Key)).Padding(px(t, 6), px(t, sp2)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
		Gap(px(t, sp2)).AlignItems(ui.Center)
	if row.Hovered() {
		row.Background(colors.Hover)
	}
	row.Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
			ui.Text(c, suggestion.Term).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0)
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				if suggestion.HeardText != "" {
					truncText(c, i18n.T("Heard: %s", suggestion.HeardText)).FontSize(px(t, fsCaption)).TextColor(colors.Text2).Shrink(1)
				}
				caption(c, suggestion.SourceText).SingleLine().Shrink(0)
			})
		})
		if ghostButton(c.Key("suggest-add-"+suggestion.Key), i18n.T("Add")).Label(i18n.T("Add %s to the dictionary", suggestion.Term)).Clicked() {
			actions.acceptDictionarySuggestion(suggestion.Key)
		}
		if ghostButton(c.Key("suggest-edit-"+suggestion.Key), i18n.T("Edit…")).Label(i18n.T("Edit %s first, then add it", suggestion.Term)).Clicked() {
			actions.editDictionarySuggestion(suggestion.Key)
		}
		if ghostButton(c.Key("suggest-ignore-"+suggestion.Key), i18n.T("Ignore")).Label(i18n.T("Ignore %s", suggestion.Term)).Clicked() {
			actions.dismissSuggestion(suggestion.Key)
		}
	})
}
