package main

import (
	"math"
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const promptMinLines = 4

func translateSettings(c *ui.Context, state *featurePromptViewState, actions featureActions, local *featureUI) {
	sectionTitle(c, i18n.T("Translation"))
	caption(c, i18n.T("How Translate works: speech recognition writes what you said in your own language, then the language model translates it into the target language with the prompt in Your rules. A prompt of your own replaces the built-in one and gets the target language only where it says {{TARGET_LANGUAGE}}. Selected text already in the target language goes into your main language instead."))
	local.lang = state.Target.Value
	featureModelSelectRow(c, "tl-target", i18n.T("Target language"), i18n.T("Target language for Translate"), &local.lang, state.Target, actions.selectFeatureTarget)
	if state.TargetWarning {
		notice(c, "tl-no-target", noticeWarn, i18n.T("Your prompt does not use {{TARGET_LANGUAGE}}, so this menu does nothing."), "", func() {
			if ui.Button(c.Key("tl-add-target"), i18n.T("Add {{TARGET_LANGUAGE}} to the prompt")).Clicked() {
				actions.addFeatureTargetVariable()
			}
		}, nil)
	}
	cleanup := state.Cleanup
	settingRow(c, "tl-cleanup", i18n.T("Basic clean-up"),
		i18n.T("Cleans up what you say before translating it. Selected text is translated as it is."), func() {
			if ui.Switch(c.Key("tl-cleanup-switch"), &cleanup).Label(i18n.T("Basic clean-up")).Changed() {
				actions.setFeatureTranslateCleanup(cleanup)
			}
		})
}

func rewriteSettings(c *ui.Context, state *featurePromptViewState, actions featureActions) {
	sectionTitle(c, i18n.T("Rewriting"))
	cleanup := state.Cleanup
	settingRow(c, "rw-cleanup", i18n.T("Clean up the instruction"),
		i18n.T("Ignores fillers and false starts in what you say. The selected text is not changed."), func() {
			if ui.Switch(c.Key("rw-cleanup-switch"), &cleanup).Label(i18n.T("Clean up the instruction")).Changed() {
				actions.setFeatureRewriteCleanup(cleanup)
			}
		})
}

func cardSetting(c *ui.Context, prefix, feature string, state *featureCardViewState, actions featureActions) {
	sectionTitle(c, i18n.T("Answer card"))
	help := i18n.T("Shows the text in the answer card instead of typing it.")
	switch feature {
	case "translate":
		help = i18n.T("Shows translations in the answer card instead of typing them.")
	case "rewrite":
		help = i18n.T("Shows answers in the answer card, where you can follow up. When off, the answer is typed or replaces the selection.")
	}
	show := state.Show
	settingRow(c, prefix+"-card", i18n.T("Show the result in a card"), help, func() {
		if ui.Switch(c.Key(prefix+"-card-switch"), &show).Label(i18n.T("Show the result in a card")).Changed() {
			actions.setFeatureShowCard(feature, show)
		}
	})
	size := state.Size
	settingRow(c, prefix+"-card-size", i18n.T("Card size"), "", func() {
		if ui.Segmented(c.Key(prefix+"-card-size-seg"), &size, i18n.T("Small"), i18n.T("Medium"), i18n.T("Large"), i18n.T("Fill")).Label(i18n.T("Card size")).Changed() {
			actions.setCardSize(feature, size)
		}
	})
}

func featureRules(c *ui.Context, prefix, feature string, state *featureRulesViewState, actions featureActions, local *featureUI) {
	t := c.Theme()
	colors := tok(t)
	if !state.SettingsOpen && c.Shortcut(ui.Cmd, ui.KeyF) {
		local.rules, local.findFocus = true, true
	}
	collapsible(c.Key(prefix+"-rules"), i18n.T("Your rules"), &local.rules, func() {
		caption(c, state.Summary).SingleLine().Shrink(0)
	}, func() {
		var label, placeholder string
		switch feature {
		case "translate":
			label = i18n.T("System prompt for Translate")
			placeholder = i18n.T("For example: use British spelling; keep product names in English; write dates as 15 October 2026.")
		case "rewrite":
			label = i18n.T("System prompt for Rewrite")
			placeholder = i18n.T("For example: write plainly, without emoji; sign mails with my first name.")
		default:
			label = i18n.T("System prompt for Transcribe")
			placeholder = i18n.T("For example: write numbers as digits; keep English product names as they are; end every sentence with a full stop.")
		}
		finder := &local.finder
		moved := finder.advance(len(state.Matches))
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			find := finder.field(c, prefix+"-find", &local.find, i18n.T("Find in the prompt"), px(t, 320))
			if local.findFocus {
				local.findFocus = false
				find.Focus()
			}
			finder.count(c, local.find, len(state.Matches))
		})
		input := ui.TextArea(c.Key(prefix+"-prompt"), &local.prompt).Label(label).Placeholder(placeholder).Lines(promptMinLines, math.MaxInt32)
		if len(state.Matches) > 0 {
			ranges := make([]ui.TextRange, len(state.Matches))
			for index, match := range state.Matches {
				ranges[index] = ui.TextRange{Start: match[0], End: match[1], Color: colors.Danger, Weight: 700}
				if index == finder.cur-1 {
					ranges[index].Weight = 900
				}
			}
			input.TextRanges(ranges...)
		}
		if moved {
			input.SetTextSelection(state.Matches[finder.cur-1][0], state.Matches[finder.cur-1][1])
		}
		if input.Changed() {
			local.promptDirty = true
			local.promptSeq++
			actions.scheduleFeaturePromptSave(feature, local.promptSeq)
		} else if local.promptDirty && !input.Focused() {
			actions.savePrompt(feature)
		}
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			caption(c, i18n.T("Variables (click to copy):")).Grow(1).MinWidth(0)
			if ghostIconButton(c.Key(prefix+"-prompt-reset"), icons.Refresh, i18n.T("Reset to default")).Label(i18n.T("Reset to default prompt")).Disabled(state.ResetDisabled).Clicked() {
				actions.resetFeaturePrompt(feature)
			}
		})
		ui.Column(c.Key(prefix + "-vars")).Gap(px(t, 4)).Children(func() {
			for _, variable := range state.Variables {
				ui.Row(c.Key("var-" + variable.Name)).Wrap().GapX(px(t, sp2)).GapY(px(t, 2)).AlignItems(ui.Center).Children(func() {
					button := ui.ButtonBase(c.Key("copy")).Padding(px(t, 2), px(t, 6)).Radius(px(t, 4)).Background(colors.Hover).Label(variable.Token).Tooltip(i18n.T("Copy")).Shrink(0)
					if button.Hovered() {
						button.Background(colors.Pressed)
					}
					button.Children(func() {
						ui.Text(c, variable.Token).FontSize(px(t, fsCaption)).Font("monospace").SingleLine()
					})
					if button.Clicked() {
						actions.copyFeatureText(variable.Token)
					}
					caption(c, variable.Text).Grow(1).Basis(px(t, 200)).MinWidth(0)
				})
			}
		})
		collapsible(c.Key(prefix+"-preview"), i18n.T("What the language model receives"), &local.preview, nil, func() {
			featurePreviewBody(c, prefix, state)
		})
	})
}

func featurePreviewBody(c *ui.Context, prefix string, state *featureRulesViewState) {
	t := c.Theme()
	if state.EnhanceOff {
		caption(c, i18n.T("Clean-up is off, so nothing is sent to a language model."))
	}
	if state.PreviewError != "" {
		caption(c, state.PreviewError)
		return
	}
	caption(c, i18n.T("The whole prompt for a sample input: your rules, then your personal dictionary when it is on, then the input. The values of the variables, filled in from your settings now, are marked."))
	fillCard(c.Key(prefix + "-preview-prompt")).Padding(px(t, sp3)).Children(func() {
		ui.RichText(c, markValues(tok(t).AccentInk, state.PreviewPrompt, state.PreviewValues)...).Label(state.PreviewPrompt).FontSize(px(t, fsCaption)).Font("monospace").Selectable()
	})
}

// markValues splits text into spans with every occurrence of values
// marked, the longest value first where they overlap.
// shortcut: a value that is also a word of the dictionary or the sample is
// marked there too; mark from the rendering if that misleads.
func markValues(accent ui.Color, text string, values []string) []ui.Span {
	values = slices.DeleteFunc(slices.Clone(values), func(v string) bool { return strings.TrimSpace(v) == "" })
	slices.SortFunc(values, func(x, y string) int { return len(y) - len(x) })
	var out []ui.Span
	plain := 0
	// Byte steps are safe: a value starts with a rune's first byte, which
	// no continuation byte equals.
	for i := 0; i < len(text); {
		n := 0
		for _, v := range values {
			if strings.HasPrefix(text[i:], v) {
				n = len(v)
				break
			}
		}
		if n == 0 {
			i++
			continue
		}
		if i > plain {
			out = append(out, ui.Span{Text: text[plain:i]})
		}
		out = append(out, ui.Span{Text: text[i : i+n], Weight: 700, Color: accent, Background: accent.Alpha(0.12)})
		i += n
		plain = i
	}
	if plain < len(text) || len(out) == 0 {
		out = append(out, ui.Span{Text: text[plain:]})
	}
	return out
}
