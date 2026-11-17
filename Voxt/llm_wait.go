package main

// How long Transcribe, Translate and Rewrite wait at least for a language
// model's answer (settings.Models.LLMWait, the Language models page,
// #3821): Automatic keeps the session's own limits; a long text gets
// longer either way (internal/session's llmDeadlines).

import (
	"slices"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// waitLabels are the choices' labels, in settings.WaitChoices' order.
func waitLabels() []string {
	return []string{i18n.T("Automatic"), i18n.T("1 minute"), i18n.T("2 minutes"), i18n.T("5 minutes"), i18n.T("10 minutes"), i18n.T("30 minutes"), i18n.T("60 minutes"), i18n.T("No limit")}
}

// llmWaitRow is the Language models page's wait setting. The label follows
// the setting and the language, unless a choice of this select is on its way
// to the setting (models_keep.go's).
func (a *app) llmWaitRow(c *ui.Context) {
	t := c.Theme()
	help := i18n.T("How long Voxt waits for an answer before it inserts the recognised text instead. Automatic waits 20 seconds to a minute, longer for long text.")
	settingRow(c, "llm-wait", i18n.T("Wait for the language model"), help, func() {
		opts := waitLabels()
		v := a.prefs.Models.LLMWait
		i := slices.Index(settings.WaitChoices, v)
		// A custom wait saved by an earlier version shows as what it is;
		// the nearest preset would say another wait than the one in effect.
		if wait := a.prefs.Models.LLMWaitFor(); i < 0 && wait > 0 {
			n := int(wait / time.Minute)
			opts = append(opts, i18n.N(n, "%d minute", "%d minutes", n))
			i = len(opts) - 1
		}
		save := func(v string) {
			a.llmWaitFor = v + "\x00" + i18n.Locale()
			a.updateSettings(func(s *settings.Settings) { s.Models.LLMWait = v })
		}
		if want := v + "\x00" + i18n.Locale(); a.llmWaitFor != want {
			a.llmWaitFor, a.llmWaitSel = want, opts[max(i, 0)]
		}
		if ui.Select(c.Key("llm-wait-select"), &a.llmWaitSel, opts).Label(i18n.T("Wait for the language model")).Width(px(t, 200)).Shrink(1).MinWidth(px(t, 140)).Changed() {
			if choice := slices.Index(opts, a.llmWaitSel); choice >= 0 && choice < len(settings.WaitChoices) {
				save(settings.WaitChoices[choice])
			}
		}
	})
}
