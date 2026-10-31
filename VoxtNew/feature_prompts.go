package main

// Translate's and Rewrite's own settings and every feature's rules, with
// the preview of the whole prompt (feature_page.go).

import (
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// trySample is the input the prompt preview shows.
func trySample() string {
	return i18n.T("um so the meeting moves to Thursday at three, no, at four")
}

// ---- Translate and Rewrite ----

// translateSettings are Translate's own: the target language, the
// clean-up of speech, the card.
func (a *app) translateSettings(c *ui.Context) {
	t := c.Theme()
	u := a.fui(hotkey.Translate)
	f := &a.prefs.Features.Translate
	sectionTitle(c, i18n.T("Translation"))
	opts := []string{i18n.T("Automatic: %s", uitext.TargetLanguageLabel(session.TranslateTarget(settings.Settings{General: a.prefs.General}, i18n.Locale(), "")))}
	for _, code := range compose.TargetLanguages {
		opts = append(opts, uitext.TargetLanguageLabel(code))
	}
	cur := opts[0]
	if j := slices.Index(compose.TargetLanguages, f.Target); j >= 0 {
		cur = opts[j+1]
	} else if f.Target != "" {
		cur = uitext.TargetLanguageLabel(f.Target)
		opts = append(opts, cur)
	}
	u.lang = cur // the settings say; a choice saves at once
	settingRow(c, "tl-target", i18n.T("Translate into"),
		i18n.T("Automatic is English, or Chinese when your main language is English (Settings → General). The answer card can translate the same text into another language."), func() {
			if ui.Select(c.Key("tl-target-sel"), &u.lang, opts).Label(i18n.T("Target language")).Width(px(t, 260)).Shrink(1).MinWidth(px(t, 160)).Changed() {
				code := ""
				if j := slices.Index(opts, u.lang); j > 0 && j <= len(compose.TargetLanguages) {
					code = compose.TargetLanguages[j-1]
				}
				a.updateSettings(func(s *settings.Settings) { s.Features.Translate.Target = code })
			}
		})
	settingRow(c, "tl-cleanup", i18n.T("Basic clean-up"),
		i18n.T("Cleans up what you say before translating it: fillers, false starts and corrections go, spoken punctuation becomes marks. Selected text is translated as it is."), func() {
			if ui.Switch(c.Key("tl-cleanup-switch"), &f.Cleanup).Label(i18n.T("Basic clean-up")).Changed() {
				on := f.Cleanup
				a.updateSettings(func(s *settings.Settings) { s.Features.Translate.Cleanup = on })
			}
		})
	settingRow(c, "tl-card", i18n.T("Show the result in a card"),
		i18n.T("Shows the translation in the answer card, to read, copy or insert, instead of typing it (or replacing the selection). Selected text that cannot be edited, such as a web page, always goes to the card."), func() {
			if ui.Switch(c.Key("tl-card-switch"), &f.ShowCard).Label(i18n.T("Show the result in a card")).Changed() {
				on := f.ShowCard
				a.updateSettings(func(s *settings.Settings) { s.Features.Translate.ShowCard = on })
			}
		})
}

// rewriteSettings are Rewrite's own: the instruction's clean-up. Every
// answer shows in the answer card (Voxt's), nothing to choose there.
func (a *app) rewriteSettings(c *ui.Context) {
	f := &a.prefs.Features.Rewrite
	sectionTitle(c, i18n.T("Rewriting"))
	settingRow(c, "rw-cleanup", i18n.T("Clean up the instruction"),
		i18n.T("Reads past fillers, false starts and corrections in what you say (\"shorter, no, more formal\"). It never changes the selected text by itself."), func() {
			if ui.Switch(c.Key("rw-cleanup-switch"), &f.Cleanup).Label(i18n.T("Clean up the instruction")).Changed() {
				on := f.Cleanup
				a.updateSettings(func(s *settings.Settings) { s.Features.Rewrite.Cleanup = on })
			}
		})
}

// ---- Your rules ----

// variableInfo says what a prompt variable holds.
func variableInfo(name string) string {
	switch name {
	case "USER_MAIN_LANGUAGE":
		return i18n.T("The language you mostly speak (Settings → General).")
	case "USER_OTHER_LANGUAGES":
		return i18n.T("The other languages you mix in.")
	case "TARGET_LANGUAGE":
		return i18n.T("The language to translate into (Translate only).")
	case "APP_NAME":
		return i18n.T("The app in front (with App enhancement).")
	}
	return ""
}

// promptOf is a feature's saved rules.
func promptOf(s settings.Settings, feature string) string {
	return featureOf(s, feature).Prompt
}

// promptSync sets the rules from the settings when they changed
// elsewhere and nothing typed waits to be saved.
func (a *app) promptSync(feature string) {
	u := a.fui(feature)
	saved := promptOf(a.prefs, feature)
	if !u.promptDirty && saved != u.promptSeen {
		u.prompt, u.promptSeen = saved, saved
	}
}

// savePrompt saves the rules as typed.
func (a *app) savePrompt(feature string) {
	u := a.fui(feature)
	u.promptDirty = false
	v := u.prompt
	if strings.TrimSpace(v) == "" {
		v = ""
	}
	u.promptSeen = v
	if v == promptOf(a.prefs, feature) {
		return
	}
	a.updateSettings(func(s *settings.Settings) {
		switch feature {
		case hotkey.Translate:
			s.Features.Translate.Prompt = v
		case hotkey.Rewrite:
			s.Features.Rewrite.Prompt = v
		default:
			s.Features.Transcribe.Prompt = v
		}
	})
}

// promptSaveDelay is how long the rules wait after the last key before
// they are saved (they are saved at once when the field loses focus).
const promptSaveDelay = 800 * time.Millisecond

func (a *app) featureRules(c *ui.Context, feature string) {
	t := c.Theme()
	k := tok(t)
	p := uitext.FeaturePrefix(feature)
	u := a.fui(feature)
	a.promptSync(feature)
	sectionTitle(c, i18n.T("Your rules"))
	var label, placeholder string
	switch feature {
	case hotkey.Translate:
		caption(c, i18n.T("These rules are merged with the built-in translation rules; where they disagree, yours win."))
		label = i18n.T("Your rules for Translate")
		placeholder = i18n.T("For example: use British spelling; keep product names in English; write dates as 15 October 2026.")
	case hotkey.Rewrite:
		caption(c, i18n.T("These rules are merged with the built-in writing rules; where they disagree, yours win."))
		label = i18n.T("Your rules for Rewrite")
		placeholder = i18n.T("For example: write plainly, without emoji; sign mails with my first name.")
	default:
		caption(c, i18n.T("These rules are merged with the built-in clean-up rules; where they disagree, yours win."))
		label = i18n.T("Your rules for Transcribe")
		placeholder = i18n.T("For example: write numbers as digits; keep English product names as they are; end every sentence with a full stop.")
	}
	in := ui.TextArea(c.Key(p+"-prompt"), &u.prompt).Label(label).Placeholder(placeholder).MinHeight(px(t, 88))
	if in.Changed() {
		u.promptDirty = true
		u.promptSeq++
		if seq := u.promptSeq; a.win != nil {
			time.AfterFunc(promptSaveDelay, func() {
				a.post(func() {
					if u.promptDirty && u.promptSeq == seq {
						a.savePrompt(feature)
					}
				})
			})
		}
	} else if u.promptDirty && !in.Focused() {
		a.savePrompt(feature) // it lost the focus
	}
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
		caption(c, i18n.T("Variables (click one to copy it):")).Grow(1).MinWidth(0)
		empty := u.prompt == "" && promptOf(a.prefs, feature) == ""
		if ghostIconButton(c.Key(p+"-prompt-reset"), icons.Refresh, i18n.T("Reset to default")).Label(i18n.T("Reset your rules")).Disabled(empty).Clicked() {
			u.prompt = ""
			a.savePrompt(feature)
		}
	})
	ui.Column(c.Key(p + "-vars")).Gap(px(t, 4)).Children(func() {
		for _, name := range compose.VariableNames {
			token := "{{" + name + "}}"
			ui.Row(c.Key("var-" + name)).Wrap().GapX(px(t, sp2)).GapY(px(t, 2)).AlignItems(ui.Center).Children(func() {
				b := ui.ButtonBase(c.Key("copy")).Padding(px(t, 2), px(t, 6)).Radius(px(t, 4)).Background(k.Hover).Label(token).Tooltip(i18n.T("Copy")).Shrink(0)
				if b.Hovered() {
					b.Background(k.Pressed)
				}
				b.Children(func() {
					ui.Text(c, token).FontSize(px(t, fsCaption)).Font("monospace").SingleLine()
				})
				if b.Clicked() {
					c.Services().WriteClipboard(token)
					a.toast(i18n.T("Copied"))
				}
				caption(c, variableInfo(name)).Grow(1).Basis(px(t, 200)).MinWidth(0)
			})
		}
	})
	collapsible(c.Key(p+"-preview"), i18n.T("Show the full prompt"), &u.preview, nil, func() {
		a.previewBody(c, feature)
	})
}

// rewriteSample is the instruction and the selection Rewrite's preview
// shows.
func rewriteSample() (instruction, selection string) {
	return i18n.T("um make it shorter, no, more formal"), i18n.T("hey, the release slips to thursday because the tests are still running, sorry!")
}

// composedPreview is the request a session would send with the settings
// and the rules as typed, for the sample text.
func (a *app) composedPreview(feature string) (compose.Output, error) {
	g := a.prefs.General
	in := compose.Input{
		TaskPrompt: a.fui(feature).prompt, MainLanguage: session.MainLanguage(a.prefs, i18n.Locale()), OtherLanguages: g.OtherLanguages,
	}
	switch feature {
	case hotkey.Translate:
		f := a.prefs.Features.Translate
		in.Task, in.Cleanup, in.Text = compose.Translate, f.Cleanup, trySample()
		in.TargetLanguage = session.TranslateTarget(a.prefs, i18n.Locale(), "")
	case hotkey.Rewrite:
		in.Task, in.Cleanup = compose.Rewrite, a.prefs.Features.Rewrite.Cleanup
		in.Instruction, in.Text = rewriteSample()
	default:
		f := a.prefs.Features.Transcribe
		in.Task, in.Cleanup, in.Structure, in.Text = compose.Transcribe, f.Cleanup, f.Structure, trySample()
	}
	return compose.Build(in)
}

func (a *app) previewBody(c *ui.Context, feature string) {
	t := c.Theme()
	k := tok(t)
	p := uitext.FeaturePrefix(feature)
	switch feature {
	case hotkey.Transcribe:
		if !a.prefs.Features.Transcribe.Enhance {
			caption(c, i18n.T("Clean-up is off: nothing is sent to a language model."))
		}
	case hotkey.Translate:
		caption(c, i18n.T("Spoken text, as below; selected text is translated without the clean-up and keeps its form."))
	case hotkey.Rewrite:
		caption(c, i18n.T("With a selection, as below; with nothing selected the model writes from the instruction and answers with a title and the text."))
	}
	out, err := a.composedPreview(feature)
	if err != nil {
		caption(c, err.Error())
		return
	}
	caption(c, i18n.T("Your dictionary's terms and App enhancement's scene are added when they apply."))
	block := func(key, title, text string) {
		ui.Column(c.Key(key)).Gap(px(t, 4)).Children(func() {
			ui.Text(c, title).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2)
			fillCard(c).Padding(px(t, sp3)).Children(func() {
				ui.Text(c, text).FontSize(px(t, fsCaption)).Font("monospace").Selectable()
			})
		})
	}
	block(p+"-preview-system", i18n.T("System prompt"), out.System)
	block(p+"-preview-user", i18n.T("User message (an example)"), out.User)
}
