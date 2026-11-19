package main

import (
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// trEndWordRow draws the streaming speech model's end-word setting.
func trEndWordRow(c *ui.Context, state *featureEndWordViewState, actions featureActions, local *keyRow) {
	if state == nil {
		return
	}
	t := c.Theme()
	if state.Custom && !local.custom {
		local.custom, local.text = true, state.CustomText
	}
	if local.custom {
		local.sel = i18n.T("Custom…")
	} else {
		local.sel = state.Value
	}
	ui.Column(c.Key("tr-endword")).Gap(px(t, 6)).Children(func() {
		settingRow(c, "tr-endword-row", i18n.T("End with a word"), i18n.T("Say this word and pause to stop recording. It is left out of the text. Works only with speech models that show text as you speak."), func() {
			options := make([]string, len(state.Options))
			for index, option := range state.Options {
				options[index] = option.Label
			}
			if !slices.Contains(options, local.sel) {
				local.sel = state.Value
			}
			if ui.Select(c.Key("tr-endword-sel"), &local.sel, options).Label(i18n.T("Word that ends the recording")).Width(px(t, 180)).Changed() {
				for _, option := range state.Options {
					if option.Label != local.sel {
						continue
					}
					if option.Custom {
						local.custom, local.text = true, ""
						return
					}
					local.custom = false
					actions.setFeatureEndWord(option.Value)
					return
				}
			}
		})
		if !local.custom {
			return
		}
		input := ui.TextInput(c.Key("tr-endword-text"), &local.text).Placeholder("over").Label(i18n.T("Your word that ends the recording")).MaxWidth(px(t, 240))
		if input.Changed() {
			local.edited = true
		}
		if local.edited && (input.Submitted() || !input.Focused()) {
			local.edited = false
			actions.setFeatureEndWord(local.text)
		}
	})
}
