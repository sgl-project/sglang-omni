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

// llmWaitRow is the Language models page's wait setting. The label
// follows the setting and the language, unless a choice of this select
// is on its way to the setting (models_keep.go's).
func (a *app) llmWaitRow(c *ui.Context) {
	t := c.Theme()
	help := i18n.T("How long Voxt waits for an answer before it inserts the recognised text instead. Automatic waits 20 seconds to a minute, longer for long text.")
	settingRow(c, "llm-wait", i18n.T("Wait for the language model"), help, func() {
		v := a.prefs.Models.LLMWait
		if want := v + "\x00" + i18n.Locale(); a.llmWaitFor != want {
			a.llmWaitSel, a.llmWaitFor = waitLabels()[max(slices.Index(settings.WaitChoices, v), 0)], want
		}
		opts := waitLabels()
		if ui.Select(c.Key("llm-wait-select"), &a.llmWaitSel, opts).Label(i18n.T("Wait for the language model")).Width(px(t, 200)).Shrink(1).MinWidth(px(t, 140)).Changed() {
			v = settings.WaitChoices[max(slices.Index(opts, a.llmWaitSel), 0)]
			a.llmWaitFor = v + "\x00" + i18n.Locale()
			a.updateSettings(func(s *settings.Settings) { s.Models.LLMWait = v })
		}
	})
}
