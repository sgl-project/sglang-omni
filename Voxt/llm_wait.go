package main

// How long Transcribe, Translate and Rewrite wait at least for a language
// model's answer (settings.Models.LLMWait, the Language models page,
// #3821): Automatic keeps the session's own limits; a long text gets
// longer either way (internal/session's llmDeadlines).

import (
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// waitLabels are the choices' labels, in settings.WaitChoices' order.
func waitLabels() []string {
	return []string{i18n.T("Automatic"), i18n.T("1 minute"), i18n.T("2 minutes"), i18n.T("5 minutes"), i18n.T("10 minutes")}
}

// llmWaitRow is the Language models page's wait setting: a preset, or
// Custom… and a whole number of minutes (#3941). The label follows the
// setting and the language, unless a choice of this select is on its way
// to the setting (models_keep.go's).
func (a *app) llmWaitRow(c *ui.Context) {
	t := c.Theme()
	help := i18n.T("How long Voxt waits for an answer before it inserts the recognised text instead. Automatic waits 20 seconds to a minute, longer for long text.")
	settingRow(c, "llm-wait", i18n.T("Wait for the language model"), help, func() {
		custom := i18n.T("Custom…")
		opts := append(waitLabels(), custom)
		save := func(v string) {
			a.llmWaitFor = v + "\x00" + i18n.Locale()
			a.updateSettings(func(s *settings.Settings) { s.Models.LLMWait = v })
		}
		v := a.prefs.Models.LLMWait
		if want := v + "\x00" + i18n.Locale(); a.llmWaitFor != want {
			i := slices.Index(settings.WaitChoices, v)
			a.llmWaitFor, a.llmWaitSel, a.llmWaitCustom = want, opts[max(i, 0)], false
			if d := a.prefs.Models.LLMWaitFor(); i < 0 && d > 0 {
				a.llmWaitSel, a.llmWaitCustom, a.llmWaitMin = custom, true, d.Minutes()
			}
		}
		if ui.Select(c.Key("llm-wait-select"), &a.llmWaitSel, opts).Label(i18n.T("Wait for the language model")).Width(px(t, 200)).Shrink(1).MinWidth(px(t, 140)).Changed() {
			a.llmWaitCustom = a.llmWaitSel == custom
			if !a.llmWaitCustom {
				save(settings.WaitChoices[max(slices.Index(opts, a.llmWaitSel), 0)])
			} else {
				// the field starts at the wait it replaces, 1 for Automatic
				a.llmWaitMin = max(a.prefs.Models.LLMWaitFor().Minutes(), 1)
				save(settings.WaitMinutes(int(a.llmWaitMin)))
			}
		}
		if a.llmWaitCustom {
			over := false
			if numberInput(c.Key("llm-wait-min"), &a.llmWaitMin, 1, settings.WaitMaxMinutes, 1, numOpts{Label: i18n.T("Wait for the language model in minutes"), Over: &over}) {
				save(settings.WaitMinutes(int(a.llmWaitMin)))
			}
			if over { // typed past the most: say so rather than drop it silently (#3955)
				caption(c, i18n.T("At most %d minutes", settings.WaitMaxMinutes))
			} else {
				caption(c, i18n.T("min (%d–%d)", 1, settings.WaitMaxMinutes))
			}
		}
	})
}
