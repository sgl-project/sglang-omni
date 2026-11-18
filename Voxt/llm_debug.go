package main

// The language model debug page's view. Request construction, model calls
// and prompt persistence live in llm_debug_controller.go.

import (
	"fmt"
	"math"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// llmDebugPromptLines is the prompt field's height when the prompt is
// short; it grows with its text from there.
const llmDebugPromptLines = 8

func llmDebugView(c *ui.Context, state *llmDebugViewState, actions llmDebugActions, local *llmDebugUI) {
	t := c.Theme()
	colors := tok(t)
	columnPageHeader(c, i18n.T("Language model debug"), func() {
		if iconButton(c.Key("llmdbg-close"), icons.Close, i18n.T("Close")).Clicked() || c.Shortcut(0, ui.KeyEscape) {
			actions.openRemoteLanguageModels()
		}
	})
	ui.Scroll(c.Key("llmdbg-scroll")).Grow(1).MinHeight(0).Children(func() {
		pageColumn(c).Gap(px(t, sp4)).Children(func() {
			caption(c, i18n.T("Try a prompt on your own text. Nothing is saved until you click Use this prompt."))
			if !state.Ready {
				caption(c, i18n.T("Set up a language model first."))
				if ui.Button(c.Key("llmdbg-setup"), i18n.T("Open Language models")).AlignSelf(ui.Start).Clicked() {
					actions.openRemoteLanguageModels()
				}
				return
			}
			ui.Row(c).Gap(px(t, sp3)).Wrap().Children(func() {
				ui.Select(c.Key("llmdbg-model"), &local.model, state.Models).Label(i18n.T("Language model")).Width(px(t, 300))
				ui.Select(c.Key("llmdbg-preset"), &local.preset, state.Presets).Label(i18n.T("Request")).Width(px(t, 300))
			})
			ui.TextArea(c.Key("llmdbg-prompt"), &local.prompt).Label(state.PromptLabel).Lines(llmDebugPromptLines, math.MaxInt32)
			ui.Row(c).Gap(px(t, sp2)).Children(func() {
				if ui.Button(c.Key("llmdbg-use"), i18n.T("Use this prompt")).Disabled(!state.Edited).Clicked() {
					actions.useRemoteDebugPrompt()
				}
				if ghostButton(c.Key("llmdbg-reset"), i18n.T("Back to the saved prompt")).Disabled(!state.Edited).Clicked() {
					local.prompt = state.SavedPrompt
				}
			})
			if state.Instruction {
				ui.TextInput(c.Key("llmdbg-instruction"), &local.instruction).Label(i18n.T("Spoken instruction")).
					Placeholder(i18n.T("um make it shorter, no, more formal"))
			}
			ui.TextArea(c.Key("llmdbg-input"), &local.input).Label(state.InputLabel).Lines(4, math.MaxInt32)
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				if primaryButton(c.Key("llmdbg-run"), i18n.T("Run")).Disabled(!state.CanRun).Clicked() {
					actions.runRemoteDebug()
				}
				if state.Running {
					ui.Spinner(c).Size(px(t, 16), px(t, 16))
					caption(c, i18n.T("Waiting for the answer…"))
				}
			})
			result := state.Result
			if result == nil {
				return
			}
			if result.Error != "" {
				notice(c, "llmdbg-err", noticeError, result.Error, "", nil, actions.dismissLLMDebugResult)
			} else {
				sectionTitle(c, result.AnswerTitle)
				fillCard(c.Key("llmdbg-answer")).Padding(px(t, sp3)).Children(func() {
					ui.Text(c, result.Answer).Selectable()
				})
			}
			if len(result.Messages) == 0 {
				return
			}
			sectionTitle(c, i18n.T("Messages sent"))
			for index, message := range result.Messages {
				ui.Column(c.Key(fmt.Sprintf("llmdbg-msg-%d", index))).Gap(px(t, 4)).Children(func() {
					ui.Text(c, message.Role).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(colors.Text2)
					fillCard(c).Padding(px(t, sp3)).Children(func() {
						ui.Text(c, message.Content).FontSize(px(t, fsCaption)).Font("monospace").Selectable()
					})
				})
			}
		})
	})
}
