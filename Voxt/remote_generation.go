package main

// A remote language model's generation controls. Capabilities, choices and
// display text arrive in view state; genForm remains local widget state.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func remoteGenerationFields(c *ui.Context, state *remoteGenerationViewState, form *genForm) {
	t := c.Theme()
	num := func(key, title, placeholder string, value *string) {
		fieldInfo(c, title, "", func() {
			ui.TextInput(c.Key(key), value).Placeholder(placeholder).Label(title)
		})
	}
	// A numeric field alone in a pair's row needs a growing column so its
	// Basis(0) does not collapse the row's height.
	half := func(key, title, placeholder string, value *string) {
		ui.Column(c).Grow(1).Basis(0).MinWidth(0).Children(func() { num(key, title, placeholder, value) })
	}
	pair := func(fn func()) { ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Start).Children(fn) }

	sectionTitle(c, i18n.T("Generation settings"))
	if len(state.ThinkOptions) > 0 {
		fieldInfo(c, i18n.T("Thinking"), "", func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
				ui.Select(c.Key("gen-think"), &form.think, remoteOptionTitles(state.ThinkOptions)).Label(i18n.T("Thinking")).Width(px(t, 240))
				thinking, _ := remoteOptionValue(state.ThinkOptions, form.think)
				switch thinking {
				case "effort":
					if len(state.EffortOptions) == 0 {
						ui.TextInput(c.Key("gen-effort"), &form.effort).Placeholder(i18n.T("e.g. low")).Label(i18n.T("Effort")).Width(px(t, 160))
					} else {
						effort := state.EffortSelection
						if ui.Select(c.Key("gen-effort"), &effort, remoteOptionTitles(state.EffortOptions)).Label(i18n.T("Effort")).Width(px(t, 180)).Changed() {
							form.effort, _ = remoteOptionValue(state.EffortOptions, effort)
						}
					}
				case "budget":
					ui.TextInput(c.Key("gen-budget"), &form.budget).Placeholder(i18n.T("Budget tokens")).Label(i18n.T("Budget tokens")).Width(px(t, 160))
				}
			})
		})
		thinking, _ := remoteOptionValue(state.ThinkOptions, form.think)
		if thinking == "on" || thinking == "effort" || thinking == "budget" {
			fieldInfo(c, i18n.T("Off for short inputs"), "", func() {
				ui.Switch(c.Key("gen-short"), &form.shortOff).Label(i18n.T("Off for short inputs"))
			})
			if form.shortOff {
				num("gen-short-under", i18n.T("Fewer characters than"), state.ShortUnderPlaceholder, &form.shortUnder)
			}
		}
	}
	if state.ShowMax {
		num("gen-max", i18n.T("Max output tokens (optional)"), i18n.T("e.g. 4096"), &form.maxTokens)
	}
	if len(state.VerbosityOptions) > 0 {
		fieldInfo(c, i18n.T("Verbosity"), "", func() {
			ui.Select(c.Key("gen-verbosity"), &form.verbosity, remoteOptionTitles(state.VerbosityOptions)).Label(i18n.T("Verbosity")).Width(px(t, 180))
		})
	}
	if state.ShowKeepAlive {
		num("gen-keepalive", i18n.T("Keep the model loaded for"), i18n.T("e.g. 5m"), &form.keepAlive)
	}
	if state.ShowUsage {
		ui.Switch(c.Key("gen-usage"), &form.usage).Label(i18n.T("Include usage in streamed responses"))
	}

	collapsible(c, i18n.T("Advanced"), &form.advanced, nil, func() {
		if len(state.APIOptions) > 0 {
			fieldInfo(c, i18n.T("API format"), i18n.T("Chat Completions works with nearly every service. Responses is OpenAI's newer API, and a custom service must support POST /responses."), func() {
				ui.Select(c.Key("gen-api"), &form.api, remoteOptionTitles(state.APIOptions)).Label(i18n.T("API format")).Width(px(t, 240))
			})
		}
		api, _ := remoteOptionValue(state.APIOptions, form.api)
		responses := api == "responses"
		pair(func() {
			if state.ShowTemperature {
				half("gen-temp", i18n.T("Temperature"), state.ServiceDefault, &form.temperature)
			}
			if state.ShowTopP {
				half("gen-topp", "Top P", state.ServiceDefault, &form.topP)
			}
		})
		pair(func() {
			if state.ShowTopK {
				half("gen-topk", "Top K", state.ServiceDefault, &form.topK)
			}
			if state.ShowMinP {
				half("gen-minp", "Min P", state.ServiceDefault, &form.minP)
			}
		})
		switch {
		case state.FrequencyOnly:
			num("gen-frequency", i18n.T("Frequency penalty"), state.ServiceDefault, &form.frequency)
		case state.ShowPenalties:
			pair(func() {
				half("gen-presence", i18n.T("Presence penalty"), state.ServiceDefault, &form.presence)
				half("gen-frequency", i18n.T("Frequency penalty"), state.ServiceDefault, &form.frequency)
			})
			num("gen-repetition", i18n.T("Repetition penalty"), state.ServiceDefault, &form.repetition)
		}
		if state.ShowSeed {
			num("gen-seed", i18n.T("Seed"), "42", &form.seed)
		}
		if len(state.ResponseFormatOptions) > 0 {
			fieldInfo(c, i18n.T("Response format"), "", func() {
				ui.Select(c.Key("gen-format"), &form.format, remoteOptionTitles(state.ResponseFormatOptions)).Label(i18n.T("Response format")).Width(px(t, 240))
			})
			format, _ := remoteOptionValue(state.ResponseFormatOptions, form.format)
			if len(state.ResponseFormatOptions) == 3 && format == "jsonSchema" {
				ui.TextArea(c.Key("gen-schema"), &form.schema).Label("JSON Schema").Placeholder(`{"type": "object", …}`).MinHeight(px(t, 96))
			}
		}
		if state.LogprobsCap && !responses {
			fieldInfo(c, "Logprobs", "", func() {
				ui.Switch(c.Key("gen-logprobs"), &form.logprobs).Label("Logprobs")
			})
			if form.logprobs {
				num("gen-toplogprobs", i18n.T("Top logprobs (optional)"), "5", &form.topLogprobs)
			}
		}
		if state.StopCap && !responses {
			fieldInfo(c, i18n.T("Stop sequences"), "", func() {
				ui.TextArea(c.Key("gen-stop"), &form.stop).Label(i18n.T("Stop sequences")).Placeholder(i18n.T("One stop sequence per line")).MinHeight(px(t, 56))
			})
		}
		if state.ShowExtraBody {
			fieldInfo(c, i18n.T("Request parameters"), i18n.T("A JSON object merged into the request. Its keys override Voxt's, nested objects are merged, and null leaves a field out."), func() {
				ui.TextArea(c.Key("gen-extra"), &form.extraBody).Label(i18n.T("Request parameters")).Placeholder(state.ExtraBodyExample).MinHeight(px(t, 72))
			})
		}
	})
}
