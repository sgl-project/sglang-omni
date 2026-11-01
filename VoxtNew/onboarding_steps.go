package main

// The setup guide's steps (onboarding.go): welcome, a try, and what
// else there is; the models' step is onboarding_models.go.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// obRow is a line of the guide: an icon, a title and a line under it.
func obRow(c *ui.Context, key string, ic icons.Icon, title, detail string) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c.Key(key)).Gap(px(t, sp3)).AlignItems(ui.Start).Padding(px(t, 6), 0).Children(func() {
		ui.Icon(c, ic.Linear).Size(px(t, 20), px(t, 20)).TextColor(k.Text2).Shrink(0)
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
			ui.Text(c, title).FontWeight(600)
			if detail != "" {
				ui.Text(c, detail).FontSize(px(t, fsCaption)).TextColor(k.Text2)
			}
		})
	})
}

// obWelcomeStep is the privacy promise.
func (a *app) obWelcomeStep(c *ui.Context) {
	obRow(c, "ob-p-local-models", modelsIcon, i18n.T("Your audio and text stay on this Mac"),
		i18n.T("Speech recognition and the language models run here. VoxtNew has no server of its own, no account and no analytics."))
	obRow(c, "ob-p-local", icons.Database, i18n.T("History stays on this Mac"),
		i18n.T("Results (and, if you choose, their recordings) are kept here only. Turn them off in Settings → Privacy."))
	obRow(c, "ob-p-urls", appEnhancementIcon, i18n.T("Browser addresses only pick your rules"),
		i18n.T("With App enhancement, the address of the tab in front chooses a group of rules; the language model never sees it."))
}

// obTryStep is a try step: the shortcut, a sample, the Try area, and
// the result once it passed.
func (a *app) obTryStep(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	kind, _ := obStepKind(a.ob.step)
	feature := string(kind)
	step := a.ob.step
	// The shortcut.
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
		hs := hotkey.HotkeysOf(a.prefs.Features, feature)
		if len(hs) == 0 {
			caption(c, i18n.T("%s has no shortcut: use Start, and add one on its page.", uitext.KindTitle(kind)))
			return
		}
		caption(c, i18n.T("Shortcut"))
		h := hs[0]
		ui.Text(c, uitext.HotkeyText(h.Keys)).FontSize(px(t, fsCaption)).Font("monospace").Padding(px(t, 2), px(t, 8)).Radius(px(t, 4)).Background(k.Hover)
		caption(c, uitext.TriggerHowText(h.Trigger))
		if hotkey.NeedsTap(h) && !a.hotkeys.Installed {
			warnLine(c, i18n.T("This shortcut works once VoxtNew may use Accessibility; Start works now."))
		}
	})
	problemLines(c, a.conflictLines(feature, 0)) // another ActNow app's keys too
	if a.ob.passed[step] {
		notice(c, "ob-passed", noticeOK, i18n.T("It works. Continue, or try again."), "", nil, nil)
	}
	// A sample.
	switch kind {
	case store.KindTranscribe:
		obSample(c, "ob-sample", i18n.T("Read this, or say anything"), i18n.T("Um, so the design review moves to Thursday afternoon, no, Friday morning, and I'll send the agenda tonight."), nil)
	case store.KindTranslate:
		target := session.TranslateTarget(a.prefs, i18n.Locale(), "")
		sample := i18n.T("Let's go camping in the mountains this weekend; remember to bring a warm jacket and a torch.")
		obSample(c, "ob-sample", i18n.T("Sample text · into %s", uitext.TargetLanguageLabel(target)), sample, func() {
			if ui.Button(c.Key("ob-translate-sample"), i18n.T("Translate this text")).Disabled(a.sess.State.Live()).Clicked() {
				a.obTry(sample)
			}
		})
	case store.KindRewrite:
		obSample(c, "ob-sample", i18n.T("Say, for example"), i18n.T("Write a short note to the team saying I'll be ten minutes late to the stand-up."), nil)
	}
	if a.ob.hint != "" {
		warnLine(c, a.ob.hint)
	}
	id := a.ob.tries[step]
	a.tryCard(c, tryOpts{
		prefix: "ob-" + uitext.FeaturePrefix(feature), feature: feature, mine: id != "" && a.sess.ID == id,
		block: &a.ob.block, start: func() *sessionNotice { return a.obTry("") },
	})
}

// obSample is a sample text in a grey card, with a title and an action.
func obSample(c *ui.Context, key, title, text string, action func()) {
	t := c.Theme()
	k := tok(t)
	fillCard(c.Key(key)).Padding(px(t, sp3), px(t, sp4)).Gap(px(t, sp2)).Children(func() {
		ui.Text(c, title).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2)
		ui.Text(c, text).Selectable()
		if action != nil {
			ui.Row(c).Children(action)
		}
	})
}

// obMoreStep introduces App enhancement and the dictionary, and lists
// the shortcuts.
func (a *app) obMoreStep(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	feature := func(key string, ic icons.Icon, title, about, button, path string) {
		card(c.Key(key)).Children(func() {
			ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Start).Children(func() {
				ui.Icon(c, ic.Linear).Size(px(t, 20), px(t, 20)).TextColor(k.Text2).Shrink(0)
				ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp2)).Children(func() {
					ui.Text(c, title).FontWeight(600)
					caption(c, about)
					ui.Row(c).Children(func() {
						if ui.Button(c.Key(key+"-open"), button).Clicked() {
							a.closeOnboarding()
							a.navigate(path)
						}
					})
				})
			})
		})
	}
	feature("ob-ae", appEnhancementIcon, i18n.T("App enhancement"),
		i18n.T("Rules by app or website: a casual tone in chats, a formal one in mail, Return pressed after inserting in a messenger. The overlay shows which group applies."),
		i18n.T("Open App enhancement"), "/app-enhancement")
	feature("ob-dict", dictionaryIcon, i18n.T("Dictionary"),
		i18n.T("Names and jargon the speech model should expect and the language model should keep as written."),
		i18n.T("Open Dictionary"), "/dictionary")
	sectionTitle(c, i18n.T("Your shortcuts"))
	ui.Column(c.Key("ob-keys")).Gap(px(t, sp1)).Children(func() {
		row := func(feature, keys string) {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Text(c, uitext.FeatureTitle(feature)).Grow(1).MinWidth(0)
				if keys == "" {
					keys = i18n.T("None")
				}
				ui.Text(c, keys).FontSize(px(t, fsCaption)).Font("monospace").TextColor(k.Text2)
			})
			problemLines(c, a.conflictLines(feature, 0)) // another ActNow app's keys too
		}
		for _, f := range hotkey.Features {
			row(f, a.featureKeys(f))
		}
		row(hotkey.PasteLast, uitext.HotkeyText(a.prefs.Features.PasteLast.Keys))
	})
}
