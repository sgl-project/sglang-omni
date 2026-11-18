package main

// The History settings dialog's Export text row.

import (
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func histExportTextRow(c *ui.Context, state *historySettingsViewState, actions historySettingsActions, local *historySettingsLocalUI) {
	t := c.Theme()
	settingRow(c, "hist-set-text", i18n.T("Export text"), "", func() {
		if !slices.Contains(state.TextScopes, local.textScope) {
			local.textScope = state.TextScope
		}
		if !slices.Contains(state.TextFormats, local.textFormat) {
			local.textFormat = state.TextFormat
		}
		ui.Select(c.Key("hist-set-text-scope"), &local.textScope, state.TextScopes).Label(i18n.T("Which entries")).Width(px(t, 150)).Shrink(1).MinWidth(px(t, 100))
		ui.Select(c.Key("hist-set-text-format"), &local.textFormat, state.TextFormats).Label(i18n.T("Format")).Width(px(t, 110)).Shrink(0)
		if ui.Button(c.Key("hist-set-text-go"), i18n.T("Export text…")).Disabled(state.TextBusy).Clicked() {
			format := historyTextMarkdown
			if local.textFormat == "JSONL" {
				format = historyTextJSONL
			}
			shown := len(state.TextScopes) > 1 && local.textScope == state.TextScopes[1]
			actions.exportShownHistoryText(shown, format)
		}
	})
}
