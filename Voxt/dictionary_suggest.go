package main

// Pending dictionary suggestions above the terms list: a filled card, so
// they read apart from the terms, each with Add as its button.

import (
	"strconv"

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
	k := tok(t)
	list := state.Suggestions
	if !local.allSuggestions {
		list = list[:min(len(list), suggestionRowsShown)]
	}
	fillCard(c.Key("dict-suggestions")).Shrink(0).Padding(px(t, sp3), px(t, sp4)).Gap(px(t, sp2)).Children(func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).MinHeight(px(t, 28)).Children(func() {
			ui.Text(c, i18n.T("Suggestions")).FontWeight(600).SingleLine().Shrink(0)
			caption(c, strconv.Itoa(state.SuggestionCount)).TextColor(k.Text3).SingleLine().Shrink(0)
			infoTip(c, i18n.T("Voxt suggests a word when you correct one it wrote, or when Scan history finds one. Nothing is added until you click Add. To stop them, turn off Learn from my corrections in Dictionary settings.")).Shrink(0)
			ui.Spacer(c)
			if len(list) < state.SuggestionCount && ghostButton(c.Key("dict-suggest-all"), i18n.T("All (%d)", state.SuggestionCount)).Clicked() {
				local.allSuggestions = true
			}
		})
		ui.Scroll(c.Key("dict-suggest-rows")).MaxHeight(px(t, suggestionRowsHeight)).Children(func() {
			ui.Column(c).Gap(px(t, 2)).Children(func() {
				for _, suggestion := range list {
					suggestionRow(c, suggestion, actions)
				}
			})
		})
	})
}

func suggestionRow(c *ui.Context, suggestion dictionarySuggestionViewState, actions dictionaryActions) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c.Key("suggest-" + suggestion.Key)).MinHeight(px(t, 44)).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
			ui.Text(c, suggestion.Term).SingleLine().Ellipsis("…").MinWidth(0)
			detail := suggestion.SourceText
			if suggestion.HeardText != "" {
				detail = i18n.T("heard as %s", suggestion.HeardText) + "  ·  " + suggestion.SourceText
			}
			truncText(c, detail).FontSize(px(t, fsCaption)).TextColor(k.Text2)
		})
		if ghostButton(c.Key("suggest-ignore-"+suggestion.Key), i18n.T("Ignore")).Label(i18n.T("Ignore %s", suggestion.Term)).Clicked() {
			actions.dismissSuggestion(suggestion.Key)
		}
		if ghostButton(c.Key("suggest-edit-"+suggestion.Key), i18n.T("Edit…")).Label(i18n.T("Edit %s first, then add it", suggestion.Term)).Clicked() {
			actions.editDictionarySuggestion(suggestion.Key)
		}
		if ui.Button(c.Key("suggest-add-"+suggestion.Key), i18n.T("Add")).Label(i18n.T("Add %s to the dictionary", suggestion.Term)).Clicked() {
			actions.acceptDictionarySuggestion(suggestion.Key)
		}
	})
}
