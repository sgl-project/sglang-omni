package main

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type replacementEditor struct {
	text    string
	saved   string
	dirty   bool
	badLine int
}

func (a *app) replacementsView(c *ui.Context) {
	t := c.Theme()
	editor := &a.dv.replacements
	if saved := a.prefs.Dictionary.Replacements; !editor.dirty && saved != editor.saved {
		editor.text, editor.saved = saved, saved
		_, editor.badLine = dict.ParseReplacements(saved)
	}
	ui.Scroll(c.Key("dict-replacements-scroll")).Grow(1).MinHeight(0).Children(func() {
		ui.Column(c).Padding(px(t, sp2), px(t, sp6), px(t, sp6)).Gap(px(t, sp2)).Children(func() {
			caption(c, i18n.T("Write one rule per line. Voxt replaces the text on this Mac before Transcribe, Translate or Rewrite sends it to a language model."))
			caption(c, i18n.T("Speech recognition receives the original audio. Choose an on-device speech model to keep the audio on this Mac."))
			if ui.TextArea(c.Key("dict-replacements"), &editor.text).Label(i18n.T("Replacement rules")).Placeholder(i18n.T("Project Atlas → Project A")).Lines(5, 12).Changed() {
				editor.dirty = editor.text != editor.saved
				_, editor.badLine = dict.ParseReplacements(editor.text)
			}
			if editor.badLine != 0 {
				caption(c, i18n.T("Use a source and a replacement on line %d, separated by →.", editor.badLine))
			}
			if primaryButton(c.Key("dict-replacements-save"), i18n.T("Save")).Disabled(!editor.dirty || editor.badLine != 0).Clicked() {
				saved := editor.text
				editor.saved, editor.dirty = saved, false
				a.updateSettings(func(s *settings.Settings) { s.Dictionary.Replacements = saved })
			}
		})
	})
}
