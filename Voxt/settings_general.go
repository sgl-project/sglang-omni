package main

// Settings → General (shell_settings.go): the app's language, the main and
// other languages, the menu bar and opening at login.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type settingsGeneralLocalUI struct {
	language     string
	mainLanguage string
	copyResult   bool
	menuBar      bool
	dock         bool
	closeToTray  bool
	openAtLogin  bool
}

func generalSettings(c *ui.Context, state settingsGeneralViewState, actions settingsGeneralActions, local *settingsGeneralLocalUI) {
	t := c.Theme()
	sectionTitle(c, i18n.T("Languages"))
	settingRow(c, "language", i18n.T("App language"), "", func() {
		options := settingsLanguageLabels(state.LanguageOptions)
		if ui.Select(c.Key("language-select"), &local.language, options).Label(i18n.T("App language")).Width(px(t, 220)).Changed() {
			actions.setAppLanguage(settingsLanguageCode(local.language, state.LanguageOptions))
		}
	})
	settingRow(c, "main-language", i18n.T("Main language"), i18n.T("The language you mostly speak. Translate defaults to English, or Chinese if this is English."), func() {
		options := settingsLanguageLabels(state.MainLanguageOptions)
		if ui.Select(c.Key("main-language-select"), &local.mainLanguage, options).Label(i18n.T("Main language")).Width(px(t, 220)).Changed() {
			actions.setMainLanguage(settingsLanguageCode(local.mainLanguage, state.MainLanguageOptions))
		}
	})
	otherLanguagesRow(c, state, actions)

	sectionTitle(c, i18n.T("Results"))
	settingRow(c, "copy-result", i18n.T("Also copy the result to the clipboard"), "", func() {
		if ui.Switch(c.Key("copy-result-switch"), &local.copyResult).Label(i18n.T("Also copy the result to the clipboard")).Changed() {
			actions.setCopyResult(local.copyResult)
		}
	})

	sectionTitle(c, i18n.T("Menu bar and login"))
	settingRow(c, "menu-bar", i18n.T("Show in the menu bar"), "", func() {
		if ui.Switch(c.Key("menu-bar-switch"), &local.menuBar).Label(i18n.T("Show in the menu bar")).Changed() {
			actions.setMenuBar(local.menuBar)
		}
	})
	settingRow(c, "dock", i18n.T("Show in the Dock"), i18n.T("When off, Voxt is only in the menu bar."), func() {
		if ui.Switch(c.Key("dock-switch"), &local.dock).Label(i18n.T("Show in the Dock")).Changed() {
			actions.setDock(local.dock)
		}
	})
	settingRow(c, "close-to-tray", i18n.T("Keep running when the window closes"), i18n.T("Shortcuts keep working. Quit with ⌘Q or from the menu bar."), func() {
		sw := ui.Switch(c.Key("close-to-tray-switch"), &local.closeToTray).Label(i18n.T("Keep running when the window closes")).Disabled(!state.MenuBar)
		if sw.Changed() {
			actions.setCloseToTray(local.closeToTray)
		}
	})
	settingRow(c, "open-at-login", i18n.T("Open at login"), "", func() {
		if ui.Switch(c.Key("open-at-login-switch"), &local.openAtLogin).Label(i18n.T("Open at login")).Changed() {
			actions.setOpenAtLogin(local.openAtLogin)
		}
	})

	sectionTitle(c, i18n.T("Features"))
	settingRow(c, "features", i18n.T("Shown features"), i18n.T("Unchecked features leave the sidebar, and their shortcuts stop. Their settings are kept."), func() {
		ui.Row(c).Gap(px(t, sp3)).Wrap().AlignItems(ui.Center).Children(func() {
			for _, feature := range state.Features {
				shown := feature.Shown
				if ui.Checkbox(c.Key("feature-"+feature.ID), &shown, feature.Title).Label(i18n.T("Show %s", feature.Title)).Changed() {
					actions.setFeatureShown(feature.ID, shown)
				}
			}
		})
	})

	sectionTitle(c, i18n.T("Getting started"))
	onboardingRow(c, actions)
}

func settingsLanguageLabels(options []settingsLanguageViewState) []string {
	labels := make([]string, len(options))
	for index, option := range options {
		labels[index] = option.Label
	}
	return labels
}

func settingsLanguageCode(label string, options []settingsLanguageViewState) string {
	for _, option := range options {
		if option.Label == label {
			return option.Code
		}
	}
	return options[0].Code
}

// otherLanguagesRow is General → Other languages: the languages the user
// mixes in, as chips with ×, and Add (a menu of the languages left).
func otherLanguagesRow(c *ui.Context, state settingsGeneralViewState, actions settingsGeneralActions) {
	t := c.Theme()
	k := tok(t)
	settingRow(c, "other-languages", i18n.T("Other languages"), i18n.T("Languages you mix in. Clean-up keeps them as they are."), func() {
		for _, language := range state.OtherLanguages {
			ui.Row(c.Key("other-lang-"+language.Code)).Gap(px(t, 2)).AlignItems(ui.Center).Padding(px(t, 2), px(t, 2), px(t, 2), px(t, 10)).
				Radius(px(t, radiusControl)).Background(k.Hover).Shrink(0).Children(func() {
				ui.Text(c, language.Label).FontSize(px(t, fsCaption)).SingleLine()
				b := ui.ButtonBase(c.Key("remove")).Size(px(t, 20), px(t, 20)).Center().Radius(px(t, 4)).Label(i18n.T("Remove %s", language.Label)).Tooltip(i18n.T("Remove"))
				if b.Hovered() {
					b.Background(k.Pressed)
				}
				b.Children(func() { ui.Icon(c, icons.Close.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.Text2) })
				if b.Clicked() {
					actions.removeOtherLanguage(language.Code)
				}
			})
		}
		if len(state.AvailableLanguages) == 0 {
			return
		}
		dropButton(c.Key("other-lang-add"), icons.Add, i18n.T("Add a language")).Shrink(0).Menu(func(m *ui.Menu) {
			for _, language := range state.AvailableLanguages {
				if m.Item(language.Label).Chosen() {
					actions.addOtherLanguage(language.Code)
				}
			}
		})
	})
}

// onboardingRow is Settings → General's way back to the guide.
func onboardingRow(c *ui.Context, actions settingsGeneralActions) {
	settingRow(c, "setup-guide", i18n.T("Setup guide"), "", func() {
		if ui.Button(c.Key("setup-guide-open"), i18n.T("Open setup guide")).Clicked() {
			actions.openOnboarding()
		}
	})
}
