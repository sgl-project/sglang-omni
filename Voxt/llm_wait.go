package main

// The Language models page's wait setting. The controller supplies the
// saved value and localized choices; this file owns only widget state.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// llmWaitLocalUI is the select's widget state. Its field names stay
// compatible with the existing modelsState embedding and tests.
type llmWaitLocalUI struct {
	llmWaitSel, llmWaitFor string
}

func remoteOptionTitles(options []remoteOptionViewState) []string {
	titles := make([]string, len(options))
	for index, option := range options {
		titles[index] = option.Title
	}
	return titles
}

func remoteOptionValue(options []remoteOptionViewState, title string) (string, bool) {
	for _, option := range options {
		if option.Title == title {
			return option.Value, option.Custom
		}
	}
	return "", false
}

// llmWaitView is the wait select: a preset, or the custom wait an earlier
// version saved, which shows but is not offered again. The label follows
// the setting and the language, unless a choice of this select is on its
// way to the setting (models_keep.go's).
func llmWaitView(c *ui.Context, state *remoteWaitViewState, actions remoteWaitActions, local *llmWaitLocalUI) {
	t := c.Theme()
	help := i18n.T("If the language model does not answer by then, Voxt inserts the text as recognised, without the clean-up. Automatic waits 20 seconds to a minute, depending on the text's length.")
	settingRow(c, "llm-wait", i18n.T("Wait for the language model"), help, func() {
		if local.llmWaitFor != state.Token {
			local.llmWaitFor = state.Token
			local.llmWaitSel = state.Selected
		}
		if ui.Select(c.Key("llm-wait-select"), &local.llmWaitSel, remoteOptionTitles(state.Options)).Label(i18n.T("Wait for the language model")).Width(px(t, 200)).Shrink(1).MinWidth(px(t, 140)).Changed() {
			if value, saved := remoteOptionValue(state.Options, local.llmWaitSel); !saved {
				actions.chooseLLMWait(value)
			}
		}
	})
}
