package main

// The Speech models page's Advanced section. Runtime idle timers and the
// saved setting are owned by models_controller.go.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func modelsAdvanced(c *ui.Context, state *modelsAdvancedViewState, actions modelsActions, local *modelsLocalUI) {
	t := c.Theme()
	collapsible(c.Key("models-advanced"), i18n.T("Advanced"), &local.modelAdvanced, nil, func() {
		help := i18n.T("Loaded models start at once but keep using memory. Automatic unloads them after 10 minutes unused.")
		settingRow(c, "models-keep", i18n.T("Keep models loaded"), help, func() {
			// The label follows the setting and the language, unless a
			// choice of this select is on its way to the setting.
			if want := state.Value + "\x00" + state.Locale; local.modelKeepFor != want {
				local.modelKeepSel, local.modelKeepFor = state.Selection, want
			}
			options := make([]string, len(state.Options))
			for index, option := range state.Options {
				options[index] = option.Label
			}
			if ui.Select(c.Key("models-keep-select"), &local.modelKeepSel, options).Label(i18n.T("Keep models loaded")).Width(px(t, 200)).Shrink(1).MinWidth(px(t, 140)).Changed() {
				for _, option := range state.Options {
					if option.Label == local.modelKeepSel {
						local.modelKeepFor = option.Value + "\x00" + state.Locale
						actions.setModelsKeepLoaded(option.Value)
						break
					}
				}
			}
		})
	})
}
