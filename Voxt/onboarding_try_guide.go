package main

// What a setup-guide try step still needs before Start (language model,
// speech model still downloading), using sessionBlocker like a real try.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func (a *app) obTryPrereqs(c *ui.Context, kind store.Kind, prefix string) {
	if n := a.sessionBlocker(kind); n != nil && n.action == actionLanguage {
		a.obLanguageGuide(c, kind, prefix)
		return
	}
	a.obSpeechDownloadGuide(c, kind, prefix)
}

func (a *app) obLanguageGuide(c *ui.Context, kind store.Kind, prefix string) {
	setUp := func() {
		if ui.Button(c.Key(prefix+"-llm-setup"), i18n.T("Set up a language model")).Clicked() {
			a.showMainWindow()
			a.navigate(remotePath)
		}
	}
	var detail string
	switch kind {
	case store.KindTranslate:
		detail = i18n.T("No language model is set up: set one up to translate.")
	case store.KindRewrite:
		detail = i18n.T("No language model is set up: set one up to rewrite.")
	default:
		detail = i18n.T("No language model is set up.")
	}
	more := i18n.T("Set one up on the Language models page, then come back and try again. You can also skip this step.")
	notice(c, prefix+"-llm-guide", noticeWarn, detail, more, setUp, nil)
}

func (a *app) obSpeechDownloadGuide(c *ui.Context, kind store.Kind, prefix string) {
	if kind == store.KindTranslate {
		return
	}
	n := a.speechWaits(kind)
	if n == nil {
		return
	}
	more := i18n.T("The model downloads in the background. You can continue the guide and try again when the bar fills, or skip this step.")
	notice(c, prefix+"-asr-dl", noticeInfo, n.text, more, nil, nil)
}
