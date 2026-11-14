package main

// Translate's and Rewrite's own settings and every feature's rules, with
// the preview of the whole prompt (feature_page.go).

import (
	"math"
	"slices"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// trySample is the input the prompt preview shows.
func trySample() string {
	return i18n.T("um so the meeting moves to Thursday at three, no, at four")
}

// ---- Translate and Rewrite ----

// translateSettings are Translate's own: the target language and the
// clean-up of speech.
func (a *app) translateSettings(c *ui.Context) {
	t := c.Theme()
	u := a.fui(hotkey.Translate)
	f := &a.prefs.Features.Translate
	sectionTitle(c, i18n.T("Translation"))
	caption(c, i18n.T("How Translate works: speech recognition writes what you said in your own language, then the language model translates it into the target language with the prompt in Your rules. A prompt of your own replaces the built-in one and gets the target language only where it says {{TARGET_LANGUAGE}}. Selected text already in the target language goes into your main language instead."))
	opts := []string{i18n.T("Automatic (now: %s)", uitext.TargetLanguageLabel(session.TranslateTarget(settings.Settings{General: a.prefs.General}, i18n.Locale(), "")))}
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
	settingRow(c, "tl-target", i18n.T("Target language"),
		i18n.T("The language model translates into this. Speech recognition writes what you said in your own language. Automatic is English, or Chinese when your main language is English."), func() {
			if ui.Select(c.Key("tl-target-sel"), &u.lang, opts).Label(i18n.T("Target language for Translate")).Width(px(t, 260)).Shrink(1).MinWidth(px(t, 160)).Changed() {
				code := ""
				if j := slices.Index(opts, u.lang); j > 0 && j <= len(compose.TargetLanguages) {
					code = compose.TargetLanguages[j-1]
				}
				a.updateSettings(func(s *settings.Settings) { s.Features.Translate.Target = code })
			}
		})
	// A prompt of its own replaces the built-in one (compose.buildFullPrompt):
	// without the variable, "into English" in it wins over the menu.
	if own := promptOf(a.prefs, hotkey.Translate); own != "" && !strings.Contains(own, targetVar) {
		notice(c, "tl-no-target", noticeWarn, i18n.T("Your prompt does not use {{TARGET_LANGUAGE}}, so this menu does nothing."), "", func() {
			if ui.Button(c.Key("tl-add-target"), i18n.T("Add {{TARGET_LANGUAGE}} to the prompt")).Clicked() {
				u := a.fui(hotkey.Translate) // its text: what was typed since the last save too
				u.prompt = strings.TrimRight(u.prompt, "\n") + "\n\n" + i18n.T("Translate the text into {{TARGET_LANGUAGE}}.")
				a.savePrompt(hotkey.Translate)
			}
		}, nil)
	}
	settingRow(c, "tl-cleanup", i18n.T("Basic clean-up"),
		i18n.T("Cleans up what you say before translating it. Selected text is translated as it is."), func() {
			if ui.Switch(c.Key("tl-cleanup-switch"), &f.Cleanup).Label(i18n.T("Basic clean-up")).Changed() {
				on := f.Cleanup
				a.updateSettings(func(s *settings.Settings) { s.Features.Translate.Cleanup = on })
			}
		})
}

// cardSetting is feature's Answer card section (p is the page's prefix):
// the feature's own card switch and card size. Rewrite's card does more
// (follow-ups): its page says so, and what turning the switch off gives
// up.
func (a *app) cardSetting(c *ui.Context, p, feature string) {
	sectionTitle(c, i18n.T("Answer card"))
	f := &a.prefs.Features
	var on *bool
	var help string
	switch feature {
	case hotkey.Translate:
		on = &f.Translate.ShowCard
		help = i18n.T("Shows translations in the answer card instead of typing them.")
	case hotkey.Rewrite:
		on = &f.Rewrite.ShowCard
		help = i18n.T("Shows answers in the answer card, where you can follow up. When off, the answer is typed or replaces the selection.")
	default:
		on = &f.Transcribe.ShowCard
		help = i18n.T("Shows the text in the answer card instead of typing it.")
	}
	settingRow(c, p+"-card", i18n.T("Show the result in a card"), help, func() {
		if ui.Switch(c.Key(p+"-card-switch"), on).Label(i18n.T("Show the result in a card")).Changed() {
			v := *on
			a.updateSettings(func(s *settings.Settings) {
				switch feature {
				case hotkey.Translate:
					s.Features.Translate.ShowCard = v
				case hotkey.Rewrite:
					s.Features.Rewrite.ShowCard = v
				default:
					s.Features.Transcribe.ShowCard = v
				}
			})
		}
	})
	size := f.CardSize(feature)
	settingRow(c, p+"-card-size", i18n.T("Card size"), "", func() {
		if ui.Segmented(c.Key(p+"-card-size-seg"), &size, i18n.T("Small"), i18n.T("Medium"), i18n.T("Large")).Label(i18n.T("Card size")).Changed() {
			a.setCardSize(feature, size)
		}
	})
}

// setCardSize saves feature's answer card size (settings.Features.CardSizes;
// the small one is the default, kept as no entry) and shows it on a card
// that shows.
func (a *app) setCardSize(feature string, step int) {
	a.updateSettings(func(s *settings.Settings) {
		if step == 0 {
			delete(s.Features.CardSizes, feature)
			return
		}
		if s.Features.CardSizes == nil {
			s.Features.CardSizes = map[string]int{}
		}
		s.Features.CardSizes[feature] = step
	})
	a.overlayRestyle()
}

// rewriteSettings are Rewrite's own: the instruction's clean-up.
func (a *app) rewriteSettings(c *ui.Context) {
	f := &a.prefs.Features.Rewrite
	sectionTitle(c, i18n.T("Rewriting"))
	settingRow(c, "rw-cleanup", i18n.T("Clean up the instruction"),
		i18n.T("Ignores fillers and false starts in what you say. The selected text is not changed."), func() {
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
		return i18n.T("Your main language")
	case "USER_OTHER_LANGUAGES":
		return i18n.T("Your other languages")
	case "TARGET_LANGUAGE":
		return i18n.T("The target language (Translate only)")
	case "APP_NAME":
		return i18n.T("The frontmost app")
	}
	return ""
}

// promptOf is a feature's saved rules.
func promptOf(s settings.Settings, feature string) string {
	return featureOf(s, feature).Prompt
}

// targetVar is the variable that takes Translate's target language.
const targetVar = "{{TARGET_LANGUAGE}}"

// promptBuiltin is the built-in system prompt before user overrides, the
// target language left as {{TARGET_LANGUAGE}}: the rules field starts
// from it, and a prompt edited from it keeps following the menu.
func (a *app) promptBuiltin(feature string) string {
	u := a.fui(feature)
	was := u.prompt
	wasDirty := u.promptDirty
	u.prompt = ""
	u.promptDirty = false
	in := a.previewInput(feature)
	u.prompt, u.promptDirty = was, wasDirty
	in.KeepTarget = true
	out, err := compose.Build(in)
	if err != nil {
		return ""
	}
	return out.System
}

// promptSync sets the rules from the settings when they changed
// elsewhere and nothing typed waits to be saved.
func (a *app) promptSync(feature string) {
	u := a.fui(feature)
	saved := promptOf(a.prefs, feature)
	builtin := a.promptBuiltin(feature)
	if !u.promptDirty {
		u.promptSeen = saved
		if saved == "" {
			u.prompt = builtin
		} else {
			u.prompt = saved
		}
	}
}

// savePrompt saves the rules as typed.
func (a *app) savePrompt(feature string) {
	u := a.fui(feature)
	u.promptDirty = false
	v := u.prompt
	builtin := a.promptBuiltin(feature)
	if strings.TrimSpace(v) == "" || strings.TrimSpace(v) == strings.TrimSpace(builtin) {
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

// promptMinLines is the rules field's height when it is empty or short; it
// grows with its text from there.
const promptMinLines = 4

// findFolded returns where query is in text, ignoring case, as rune
// offsets [start, end), the text fields' unit; none for a blank query.
func findFolded(text, query string) [][2]int {
	if strings.TrimSpace(query) == "" {
		return nil
	}
	haystack, needle := []rune(text), []rune(query)
	var out [][2]int
	for i := 0; i+len(needle) <= len(haystack); i++ {
		j := 0
		for j < len(needle) && unicode.ToLower(haystack[i+j]) == unicode.ToLower(needle[j]) {
			j++
		}
		if j == len(needle) {
			out = append(out, [2]int{i, i + j})
			i += j - 1
		}
	}
	return out
}

func (a *app) featureRules(c *ui.Context, feature string) {
	t := c.Theme()
	k := tok(t)
	p := uitext.FeaturePrefix(feature)
	u := a.fui(feature)
	a.promptSync(feature)
	if !a.settingsOpen && c.Shortcut(ui.Cmd, ui.KeyF) {
		u.rules, u.findFocus = true, true
	}
	// Folded by default: long rules would push the rest of the page down.
	// Its title tells how long the rules are.
	summary := i18n.T("Not set")
	if n := utf8.RuneCountInString(strings.TrimSpace(u.prompt)); n > 0 {
		summary = i18n.N(n, "%d character", "%d characters", n)
	}
	collapsible(c.Key(p+"-rules"), i18n.T("Your rules"), &u.rules, func() {
		caption(c, summary).SingleLine().Shrink(0)
	}, func() {
		var label, placeholder string
		switch feature {
		case hotkey.Translate:
			label = i18n.T("System prompt for Translate")
			placeholder = i18n.T("For example: use British spelling; keep product names in English; write dates as 15 October 2026.")
		case hotkey.Rewrite:
			label = i18n.T("System prompt for Rewrite")
			placeholder = i18n.T("For example: write plainly, without emoji; sign mails with my first name.")
		default:
			label = i18n.T("System prompt for Transcribe")
			placeholder = i18n.T("For example: write numbers as digits; keep English product names as they are; end every sentence with a full stop.")
		}
		matches := findFolded(u.prompt, u.find)
		f := &u.finder
		moved := f.advance(len(matches))
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			find := f.field(c, p+"-find", &u.find, i18n.T("Find in the prompt"), px(t, 320))
			if u.findFocus {
				u.findFocus = false
				find.Focus()
			}
			f.count(c, u.find, len(matches))
		})
		// As high as the rules, however long: a long prompt reads down the
		// page, which scrolls to the caret and to what Find selects (MyGo
		// patch 0012), instead of in a box that scrolls inside the page.
		in := ui.TextArea(c.Key(p+"-prompt"), &u.prompt).Label(label).Placeholder(placeholder).Lines(promptMinLines, math.MaxInt32)
		if len(matches) > 0 {
			ranges := make([]ui.TextRange, len(matches))
			for i, m := range matches {
				// A text area's runs take no ground: the current match is
				// heavier, and selected below.
				ranges[i] = ui.TextRange{Start: m[0], End: m[1], Color: k.Danger, Weight: 700}
				if i == f.cur-1 {
					ranges[i].Weight = 900
				}
			}
			in.TextRanges(ranges...)
		}
		if moved {
			in.SetTextSelection(matches[f.cur-1][0], matches[f.cur-1][1])
		}
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
			caption(c, i18n.T("Variables (click to copy):")).Grow(1).MinWidth(0)
			builtin := a.promptBuiltin(feature)
			empty := promptOf(a.prefs, feature) == "" && strings.TrimSpace(u.prompt) == strings.TrimSpace(builtin)
			if ghostIconButton(c.Key(p+"-prompt-reset"), icons.Refresh, i18n.T("Reset to default")).Label(i18n.T("Reset to default prompt")).Disabled(empty).Clicked() {
				u.prompt = builtin
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
		collapsible(c.Key(p+"-preview"), i18n.T("Example user message"), &u.preview, nil, func() {
			a.previewBody(c, feature)
		})
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
	return compose.Build(a.previewInput(feature))
}

// previewInput is composedPreview's input.
func (a *app) previewInput(feature string) compose.Input {
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
	return in
}

func (a *app) previewBody(c *ui.Context, feature string) {
	t := c.Theme()
	k := tok(t)
	p := uitext.FeaturePrefix(feature)
	if feature == hotkey.Transcribe && !a.prefs.Features.Transcribe.Enhance {
		caption(c, i18n.T("Clean-up is off, so nothing is sent to a language model."))
	}
	out, err := a.composedPreview(feature)
	if err != nil {
		caption(c, err.Error())
		return
	}
	ui.Column(c.Key(p + "-preview-user")).Gap(px(t, 4)).Children(func() {
		ui.Text(c, i18n.T("Example user message")).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2)
		fillCard(c).Padding(px(t, sp3)).Children(func() {
			ui.Text(c, out.User).FontSize(px(t, fsCaption)).Font("monospace").Selectable()
		})
	})
}
