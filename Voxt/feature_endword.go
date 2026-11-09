package main

// The Transcribe page's "End with a word" row (settings.Features.EndWord,
// Voxt 1.x's Voice End Command): the word said last ends the recording
// and is left out of the text (internal/session). Only a speech model
// that streams hears it, so the row shows only with one; the word works
// in every feature whose speech model streams.

import (
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// endWordPresets are the words Voxt 1.x offered.
var endWordPresets = []string{"over", "end", "完毕", "好了"}

// trEndWordRow is the end word: Off, one of endWordPresets, or a word
// typed after Custom….
func (a *app) trEndWordRow(c *ui.Context) {
	if m, ok := a.featureModel(local.ASR, hotkey.Transcribe); !ok || !m.Stream {
		return
	}
	t := c.Theme()
	r := &a.fui(hotkey.Transcribe).endWord
	cur := a.prefs.Features.EndWord
	off, custom := i18n.T("Off"), i18n.T("Custom…")
	switch {
	case r.custom:
		r.sel = custom
	case cur == "":
		r.sel = off
	case slices.Contains(endWordPresets, cur):
		r.sel = cur
	default:
		r.sel, r.custom, r.text = custom, true, cur
	}
	ui.Column(c.Key("tr-endword")).Gap(px(t, 6)).Children(func() {
		settingRow(c, "tr-endword-row", i18n.T("End with a word"), i18n.T("Say this word last and pause: the recording stops as if you pressed the shortcut, and the word is left out of the text. Only speech models that show the text as you speak hear it."), func() {
			opts := slices.Concat([]string{off}, endWordPresets, []string{custom})
			if ui.Select(c.Key("tr-endword-sel"), &r.sel, opts).Label(i18n.T("Word that ends the recording")).Width(px(t, 180)).Changed() {
				switch r.sel {
				case custom:
					r.custom, r.text = true, ""
				case off:
					r.custom = false
					a.setEndWord("")
				default:
					r.custom = false
					a.setEndWord(r.sel)
				}
			}
		})
		if !r.custom {
			return
		}
		in := ui.TextInput(c.Key("tr-endword-text"), &r.text).Placeholder("over").Label(i18n.T("Your word that ends the recording")).MaxWidth(px(t, 240))
		if in.Changed() {
			r.edited = true
		}
		if r.edited && (in.Submitted() || !in.Focused()) {
			r.edited = false
			a.setEndWord(strings.Join(strings.Fields(r.text), " "))
		}
	})
}

func (a *app) setEndWord(word string) {
	a.updateSettings(func(s *settings.Settings) { s.Features.EndWord = word })
}
