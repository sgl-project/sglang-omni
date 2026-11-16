package main

import (
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func featureModels(c *ui.Context, feature string, state *featureModelsViewState, actions featureActions, local *featureUI) {
	prefix := state.Prefix
	sectionTitleTip(c, i18n.T("Models"), i18n.T("Automatic uses the recommended speech model and your first language model. With no speech model installed, it uses a remote speech service you set up."))
	for _, item := range state.Before {
		featureModelNotice(c, prefix, item, actions)
	}
	featureModelSelectRow(c, prefix+"-asr", i18n.T("Speech recognition"), state.ASRLabel, &local.asr, &state.ASR, func(id string) {
		actions.selectFeatureModel(feature, "asr", id)
	})
	featureModelSelectRow(c, prefix+"-llm", i18n.T("Language model"), state.LLMLabel, &local.llm, &state.LLM, func(id string) {
		actions.selectFeatureModel(feature, "llm", id)
	})
	if state.Fallback != nil {
		featureModelSelectRow(c, prefix+"-fallback", i18n.T("Fallback model"), i18n.T("Fallback language model for Translate"), &local.fallback, state.Fallback, actions.selectFeatureFallback)
	}
	for _, item := range state.After {
		featureModelNotice(c, prefix, item, actions)
	}
}

func trEnhance(c *ui.Context, state *featureCleanupViewState, actions featureActions) {
	sectionTitle(c, i18n.T("Clean-up"))
	enhance, cleanup, structure := state.Enhance, state.Cleanup, state.Structure
	settingRow(c, "tr-enhance", i18n.T("Clean up with a language model"),
		i18n.T("When off, or if the model fails, the text is typed as recognised."), func() {
			if ui.Switch(c.Key("tr-enhance-switch"), &enhance).Label(i18n.T("Clean up with a language model")).Changed() {
				actions.setFeatureEnhance(enhance)
			}
		})
	settingRow(c, "tr-cleanup", i18n.T("Basic clean-up"),
		i18n.T("Removes fillers and false starts, fixes punctuation, and turns spoken punctuation (\"comma\") into marks."), func() {
			if ui.Switch(c.Key("tr-cleanup-switch"), &cleanup).Label(i18n.T("Basic clean-up")).Disabled(!enhance).Changed() {
				actions.setFeatureCleanup(cleanup)
			}
		})
	settingRow(c, "tr-structure", i18n.T("Structure"),
		i18n.T("Makes lists and paragraphs when you dictate steps or a long text."), func() {
			if ui.Switch(c.Key("tr-structure-switch"), &structure).Label(i18n.T("Structure")).Disabled(!enhance).Changed() {
				actions.setFeatureStructure(structure)
			}
		})
}

func featureModelNoticeAction(actions featureActions, action string) {
	if action == "models" {
		actions.openFeatureSpeechModels()
		return
	}
	actions.openFeatureLanguageModels()
}

func featureModelNotice(c *ui.Context, prefix string, state featureModelNoticeViewState, actions featureActions) {
	kind := noticeInfo
	if state.Warning {
		kind = noticeWarn
	}
	notice(c, state.Key, kind, state.Text, "", func() {
		suffix := "llm"
		if state.Action == "models" {
			suffix = "models"
		}
		if ui.Button(c.Key(prefix+"-open-"+suffix), state.ActionLabel).Clicked() {
			featureModelNoticeAction(actions, state.Action)
		}
	}, nil)
}

func featureModelSelectRow(c *ui.Context, key, title, label string, value *string, state *featureSelectViewState, save func(string)) {
	t := c.Theme()
	help := ""
	if key == "tl-fallback" {
		help = i18n.T("Used when the language model fails.")
	}
	settingRow(c, key, title, help, func() {
		options := make([]string, len(state.Options))
		for index, option := range state.Options {
			options[index] = option.Label
		}
		if !slices.Contains(options, *value) {
			*value = state.Value
		}
		selectControl := ui.Select(c.Key(key+"-sel"), value, options).Label(label).Width(px(t, 260)).Shrink(1).MinWidth(px(t, 160)).Disabled(state.Disabled)
		if selectControl.Changed() {
			for _, option := range state.Options {
				if option.Label == *value {
					save(option.Value)
					return
				}
			}
		}
	})
}
