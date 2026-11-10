package main

// Settings → General (shell_settings.go): the app's language, the main and
// other languages, the menu bar and opening at login.

import (
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func (a *app) generalSettings(c *ui.Context) {
	t := c.Theme()
	sectionTitle(c, i18n.T("Languages"))
	settingRow(c, "language", i18n.T("App language"), i18n.T("The language of Voxt's interface. When following the system, Voxt uses English if it does not support the system's language."), func() {
		opts := languageOptions()
		if ui.Select(c.Key("language-select"), &a.settingsForm.language, opts).Label(i18n.T("App language")).Width(px(t, 220)).Changed() {
			tag := ""
			if i := slices.Index(opts, a.settingsForm.language); i > 0 {
				tag = i18n.Supported()[i-1]
			}
			a.updateSettings(func(s *settings.Settings) { s.General.Language = tag })
		}
	})
	settingRow(c, "main-language", i18n.T("Main language"), i18n.T("The language you mostly speak and write. Clean-up keeps your text in it. Translate writes in English, or in Chinese when this is English, unless you change it on the Translate page."), func() {
		opts := mainLanguageOptions()
		if ui.Select(c.Key("main-language-select"), &a.settingsForm.mainLanguage, opts).Label(i18n.T("Main language")).Width(px(t, 220)).Changed() {
			code := ""
			if i := slices.Index(opts, a.settingsForm.mainLanguage); i > 0 {
				code = mainLanguages[i-1]
			}
			a.setMainLanguage(code)
		}
	})
	a.otherLanguagesRow(c)

	sectionTitle(c, i18n.T("Results"))
	settingRow(c, "copy-result", i18n.T("Also copy the result to the clipboard"), i18n.T("Every result stays on the clipboard, whether it was pasted or is waiting in the answer card."), func() {
		if ui.Switch(c.Key("copy-result-switch"), &a.settingsForm.copyResult).Label(i18n.T("Also copy the result to the clipboard")).Changed() {
			on := a.settingsForm.copyResult
			a.updateSettings(func(s *settings.Settings) { s.General.CopyResult = on })
		}
	})

	sectionTitle(c, i18n.T("Menu bar and login"))
	settingRow(c, "menu-bar", i18n.T("Show in the menu bar"), i18n.T("An icon in the menu bar to start a transcription, see the last result and open Voxt."), func() {
		if ui.Switch(c.Key("menu-bar-switch"), &a.settingsForm.menuBar).Label(i18n.T("Show in the menu bar")).Changed() {
			on := a.settingsForm.menuBar
			a.updateSettings(func(s *settings.Settings) {
				s.General.MenuBar = on
				if !on {
					s.General.HideDock = false // VoxtNew must stay reachable
				}
			})
		}
	})
	settingRow(c, "dock", i18n.T("Show in the Dock"), i18n.T("When off, Voxt is only in the menu bar."), func() {
		if ui.Switch(c.Key("dock-switch"), &a.settingsForm.dock).Label(i18n.T("Show in the Dock")).Changed() {
			on := a.settingsForm.dock
			a.updateSettings(func(s *settings.Settings) {
				s.General.HideDock = !on
				if !on {
					s.General.MenuBar = true // VoxtNew must stay reachable
				}
			})
		}
	})
	settingRow(c, "close-to-tray", i18n.T("Keep running when the window closes"), i18n.T("Closing the window leaves Voxt in the menu bar, so its shortcuts keep working. Quit from the menu bar icon or with ⌘Q."), func() {
		sw := ui.Switch(c.Key("close-to-tray-switch"), &a.settingsForm.closeToTray).Label(i18n.T("Keep running when the window closes")).Disabled(!a.prefs.General.MenuBar)
		if sw.Changed() {
			on := a.settingsForm.closeToTray
			a.updateSettings(func(s *settings.Settings) { s.General.CloseToTray = on })
		}
	})
	settingRow(c, "open-at-login", i18n.T("Open at login"), "", func() {
		if ui.Switch(c.Key("open-at-login-switch"), &a.settingsForm.openAtLogin).Label(i18n.T("Open at login")).Changed() {
			on := a.settingsForm.openAtLogin
			a.updateSettings(func(s *settings.Settings) { s.General.LaunchAtLogin = on })
		}
	})

	sectionTitle(c, i18n.T("Features"))
	settingRow(c, "features", i18n.T("Shown features"), i18n.T("Uncheck a feature you do not use to remove it from the sidebar, Home and ⌘K and turn off its shortcuts and scene rules. Its settings are kept for when you check it again."), func() {
		ui.Row(c).Gap(px(t, sp3)).Wrap().AlignItems(ui.Center).Children(func() {
			for _, id := range []string{settings.HideTranslate, settings.HideRewrite, settings.HideAppEnhancement} {
				shown := !a.prefs.Features.IsHidden(id)
				if ui.Checkbox(c.Key("feature-"+id), &shown, pageTitle("/"+id)).Label(i18n.T("Show %s", pageTitle("/"+id))).Changed() {
					a.updateSettings(func(s *settings.Settings) {
						s.Features.Hidden = slices.DeleteFunc(s.Features.Hidden, func(x string) bool { return x == id })
						if !shown {
							s.Features.Hidden = append(s.Features.Hidden, id)
						}
					})
				}
			}
		})
	})

	sectionTitle(c, i18n.T("Getting started"))
	a.onboardingRow(c) // onboarding.go
}

// setMainLanguage sets the main language (a lang code, "" = the app's)
// and takes it off the other languages.
func (a *app) setMainLanguage(code string) {
	a.updateSettings(func(s *settings.Settings) {
		s.General.MainLanguage = code
		s.General.OtherLanguages = withoutLanguage(s.General.OtherLanguages, session.MainLanguage(*s, i18n.Locale()))
	})
}

// otherLanguagesRow is General → Other languages: the languages the user
// mixes in, as chips with ×, and Add (a menu of the languages left).
func (a *app) otherLanguagesRow(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	main := lang.Canonical(session.MainLanguage(a.prefs, i18n.Locale()))
	others := withoutLanguage(a.prefs.General.OtherLanguages, main)
	settingRow(c, "other-languages", i18n.T("Other languages"), i18n.T("Languages you mix into what you say and write. Clean-up keeps them as they are instead of translating them into your main language."), func() {
		for _, code := range others {
			label := uitext.LangLabel(code)
			ui.Row(c.Key("other-lang-"+code)).Gap(px(t, 2)).AlignItems(ui.Center).Padding(px(t, 2), px(t, 2), px(t, 2), px(t, 10)).
				Radius(px(t, radiusControl)).Background(k.Hover).Shrink(0).Children(func() {
				ui.Text(c, label).FontSize(px(t, fsCaption)).SingleLine()
				b := ui.ButtonBase(c.Key("remove")).Size(px(t, 20), px(t, 20)).Center().Radius(px(t, 4)).Label(i18n.T("Remove %s", label)).Tooltip(i18n.T("Remove"))
				if b.Hovered() {
					b.Background(k.Pressed)
				}
				b.Children(func() { ui.Icon(c, icons.Close.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.Text2) })
				if b.Clicked() {
					a.updateSettings(func(s *settings.Settings) { s.General.OtherLanguages = withoutLanguage(s.General.OtherLanguages, code) })
				}
			})
		}
		var left []string
		for _, code := range mainLanguages {
			if lang.Canonical(code) != main && !slices.ContainsFunc(others, func(o string) bool { return lang.Canonical(o) == lang.Canonical(code) }) {
				left = append(left, code)
			}
		}
		if len(left) == 0 {
			return
		}
		dropButton(c.Key("other-lang-add"), icons.Add, i18n.T("Add a language")).Shrink(0).Menu(func(m *ui.Menu) {
			for _, code := range left {
				if m.Item(uitext.LangLabel(code)).Chosen() {
					a.updateSettings(func(s *settings.Settings) {
						if !slices.Contains(s.General.OtherLanguages, code) {
							s.General.OtherLanguages = append(slices.Clone(s.General.OtherLanguages), code)
						}
					})
				}
			}
		})
	})
}

// withoutLanguage is list without code (in any spelling: "zh", "zh-Hans").
func withoutLanguage(list []string, code string) []string {
	c := lang.Canonical(code)
	return slices.DeleteFunc(slices.Clone(list), func(x string) bool { return x == code || c != "" && lang.Canonical(x) == c })
}
