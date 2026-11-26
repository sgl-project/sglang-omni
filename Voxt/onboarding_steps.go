package main

// The setup guide's steps (onboarding.go): welcome, a try, and what
// else there is; the models' step is onboarding_models.go.

import (
	"strconv"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// obRow is a line of the guide: an icon, a title and a line under it.
func obRow(c *ui.Context, key string, icon icons.Icon, title, detail string) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c.Key(key)).Gap(px(t, sp3)).AlignItems(ui.Start).Padding(px(t, 6), 0).Children(func() {
		ui.Icon(c, icon.Linear).Size(px(t, 20), px(t, 20)).TextColor(k.Text2).Shrink(0)
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
			ui.Text(c, title).FontWeight(600)
			if detail != "" {
				ui.Text(c, detail).FontSize(px(t, fsCaption)).TextColor(k.Text2)
			}
		})
	})
}

// obWelcomeStep is the privacy promise.
func obWelcomeStep(c *ui.Context, state onboardingWelcomeViewState, actions onboardingActions) {
	if state.LegacyAvailable {
		obLegacyCard(c, state.Legacy, actions)
	}
	obRow(c, "ob-p-local-models", modelsIcon, i18n.T("Your audio stays on this Mac"),
		i18n.T("Speech is recognised on this Mac unless you choose a remote speech service. Text goes only to the language model you set up. Voxt has no server of its own, no account and no analytics."))
	obRow(c, "ob-p-local", icons.Database, i18n.T("History stays on this Mac"),
		i18n.T("Results, and their recordings if you choose, are kept only on this Mac. You can turn this off in Settings → Privacy."))
	obRow(c, "ob-p-urls", appEnhancementIcon, i18n.T("Browser addresses only choose which rules apply"),
		i18n.T("With App enhancement, the address of the current tab picks a group of rules. The language model never sees it."))
}

// obLegacyCard offers the previous Voxt's dictionary when that app's data
// is on this Mac: one button, then one click in the panel it opens.
func obLegacyCard(c *ui.Context, state onboardingLegacyViewState, actions onboardingActions) {
	t := c.Theme()
	card(c.Key("ob-legacy")).Children(func() {
		ui.Column(c).Gap(px(t, sp2)).Children(func() {
			ui.Text(c, i18n.T("The previous Voxt is on this Mac")).FontWeight(600)
			caption(c, i18n.T("Click Import from Voxt 1.x to bring over its dictionary, history and settings. The previous Voxt keeps all its data."))
			if state.Done || state.Failed != "" {
				onboardingLegacyReportView(c, state, actions)
			}
			button, label := primaryButton, i18n.T("Import from Voxt 1.x")
			if state.Done {
				button, label = ghostButton, i18n.T("Import again")
			}
			if button(c.Key("ob-legacy-import"), label).Clicked() {
				actions.importLegacy()
			}
		})
	})
}

func onboardingLegacyReportView(c *ui.Context, state onboardingLegacyViewState, actions onboardingActions) {
	t := c.Theme()
	k := tok(t)
	if state.Failed != "" {
		statusMark(c, icons.DangerTriangle, k.Warning, state.Failed)
		return
	}
	group := func(key string, icon icons.Icon, color ui.Color, title string, lines func()) {
		ui.Column(c.Key(key)).Gap(px(t, 2)).Children(func() {
			ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Center).Children(func() {
				ui.Icon(c, icon.Linear).Size(px(t, 14), px(t, 14)).TextColor(color).Shrink(0)
				sectionTitleText(c, title)
			})
			ui.Column(c).Gap(px(t, 2)).Padding(0, 0, 0, px(t, 20)).Children(lines)
		})
	}
	group("legacy-imported", icons.CheckCircle, k.Success, i18n.T("Imported"), func() {
		for _, line := range state.Imported {
			caption(c, line)
		}
	})
	if len(state.Skipped) > 0 {
		group("legacy-skipped", icons.InfoCircle, k.Text2, i18n.T("Left out"), func() {
			for _, line := range state.Skipped {
				caption(c, line)
			}
		})
	}
	if len(state.Todos) > 0 {
		group("legacy-todo", icons.DangerTriangle, k.Warning, i18n.T("To do"), func() {
			for index, todo := range state.Todos {
				ui.Row(c.Key("todo-" + strconv.Itoa(index))).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
					caption(c, todo.Text).Grow(1).MinWidth(0)
					if ghostButton(c.Key("complete"), i18n.T("Complete setup")).Label(i18n.T("Complete setup of %s", todo.ServiceName)).Clicked() {
						actions.completeOnboardingLegacyTodo(todo.Kind, todo.ID)
					}
				})
			}
		})
	}
}

func onboardingPermissionRows(c *ui.Context, state onboardingPermissionsViewState, actions onboardingActions) {
	t := c.Theme()
	badge := func(text string, ok bool) {
		if ok {
			statusMark(c, icons.CheckCircle, t.Success, text)
		} else {
			statusMark(c, icons.DangerTriangle, t.Warning, text)
		}
	}
	if state.AskRequired {
		ui.Row(c).Children(func() {
			if ui.Button(c.Key("perm-ask-required"), i18n.T("Ask for the required permissions")).Tooltip(i18n.T("macOS asks for each in turn: the microphone, then Accessibility.")).Clicked() {
				actions.askRequired()
			}
		})
	}
	sectionTitle(c, i18n.T("Required"))
	row := func(key, title, description, privacyURL string, permission onboardingPermissionRowViewState) {
		settingRow(c, key, title, description, func() {
			badge(permission.Text, permission.OK)
			if permission.CanAsk {
				button := ghostButton(c.Key(key+"-ask"), i18n.T("Ask"))
				if permission.AskTooltip != "" {
					button = button.Tooltip(permission.AskTooltip)
				}
				if button.Clicked() {
					actions.askPermission(permission.AskAction, nil)
				}
			}
			if permission.CanOpen {
				if ghostIconButton(c.Key(key+"-open"), icons.SquareArrowRightUp, i18n.T("Open System Settings")).Label(i18n.T("Open System Settings")).Clicked() {
					actions.onboardingOpenPrivacy(privacyURL)
				}
			}
		})
	}
	row("perm-mic", i18n.T("Microphone"), "", privacyMicrophone, state.Microphone)
	row("perm-access", i18n.T("Accessibility"), i18n.T("Lets Voxt type results, read the selected text and the browser tab's address, and use single keys like Right ⌥ as shortcuts."), privacyAccessibility, state.Access)
	sectionTitle(c, i18n.T("Optional"))
	settingRow(c, "perm-system-audio", i18n.T("System audio recording"), i18n.T("Lets Voxt record sound from other apps during a meeting. macOS asks the first time a meeting records system audio."), func() {
		audio := state.SystemAudio
		statusMark(c, icons.InfoCircle, tok(t).Text2, audio.Text).Tooltip(audio.Tip)
		if audio.CanAsk {
			if ghostButton(c.Key("perm-system-audio-ask"), i18n.T("Ask")).Tooltip(audio.AskTooltip).Clicked() {
				actions.askPermission(audio.AskAction, nil)
			}
		}
		if audio.CanOpen {
			if ghostIconButton(c.Key("perm-system-audio-open"), icons.SquareArrowRightUp, i18n.T("Open System Settings")).Label(i18n.T("Open System Settings")).Clicked() {
				actions.onboardingOpenPrivacy(privacyAudioCapture)
			}
		}
	})
	row("perm-speech", i18n.T("Speech Recognition"), i18n.T("Needed only for System dictation."), privacySpeech, state.Speech)
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		if secondaryIconButton(c.Key("perm-recheck"), icons.Refresh, i18n.T("Check again")).Label(i18n.T("Check again")).Clicked() {
			actions.recheckPermissions()
		}
		infoTip(c.Key("perm-note"), i18n.T("After changing a permission, you may need to reopen Voxt."))
	})
}

// obTryStep is a try step: the shortcut, a sample, the Try area, and
// the result once it passed.
func obTryStep(c *ui.Context, state onboardingTryViewState, actions onboardingActions) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
		if !state.Shortcut.Present {
			caption(c, i18n.T("%s has no shortcut yet. Use Start, or add one on its page.", state.Shortcut.FeatureTitle))
			return
		}
		caption(c, i18n.T("Shortcut"))
		ui.Text(c, state.Shortcut.Text).FontSize(px(t, fsCaption)).Font("monospace").Padding(px(t, 2), px(t, 8)).Radius(px(t, 4)).Background(k.Hover)
		caption(c, state.Shortcut.How)
		if state.Shortcut.AccessibilityWarning {
			warnLine(c, i18n.T("This shortcut works once Voxt has Accessibility access."))
		}
	})
	obKeyboard(c, state.Shortcut.Keyboard)
	problemLines(c, state.Conflicts)
	if state.Passed {
		notice(c, "ob-passed", noticeOK, i18n.T("It works. Continue, or try again."), "", nil, nil)
	}
	var sampleAction func()
	if state.SampleAction {
		sampleAction = func() {
			if ui.Button(c.Key("ob-translate-sample"), i18n.T("Translate this text")).Disabled(state.Card.Live).Clicked() {
				_ = actions.obTry(state.SampleText)
			}
		}
	}
	obSample(c, "ob-sample", state.SampleTitle, state.SampleText, sampleAction)
	if state.Hint != "" {
		warnLine(c, state.Hint)
	}
	obTryPrereqs(c, state, actions)
	obTryCard(c, state.Prefix, state.Card, actions)
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
func obMoreStep(c *ui.Context, state onboardingMoreViewState, actions onboardingActions) {
	t := c.Theme()
	k := tok(t)
	feature := func(key string, icon icons.Icon, title, about, button, path string) {
		card(c.Key(key)).Children(func() {
			ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Start).Children(func() {
				ui.Icon(c, icon.Linear).Size(px(t, 20), px(t, 20)).TextColor(k.Text2).Shrink(0)
				ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp2)).Children(func() {
					ui.Text(c, title).FontWeight(600)
					caption(c, about)
					ui.Row(c).Children(func() {
						if ui.Button(c.Key(key+"-open"), button).Clicked() {
							actions.finishOnboarding(path)
						}
					})
				})
			})
		})
	}
	feature("ob-ae", appEnhancementIcon, i18n.T("App enhancement"),
		i18n.T("Rules for each app or website: a casual tone in chats, a formal one in mail, or pressing Return after inserting in a messenger. The overlay shows which group applies."),
		i18n.T("Open App enhancement"), "/app-enhancement")
	feature("ob-dict", dictionaryIcon, i18n.T("Dictionary"),
		i18n.T("Names and jargon the speech model should expect and the language model should keep as written."),
		i18n.T("Open Dictionary"), "/dictionary")
	sectionTitle(c, i18n.T("Shortcuts"))
	onboardingShortcutPresetPicker(c, state, actions)
	ui.Column(c.Key("ob-keys")).Gap(px(t, sp1)).Children(func() {
		for _, row := range state.Shortcuts {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Text(c, row.Title).Grow(1).MinWidth(0)
				ui.Text(c, row.Keys).FontSize(px(t, fsCaption)).Font("monospace").TextColor(k.Text2)
			})
			problemLines(c, row.Problems)
		}
	})
}

func onboardingShortcutPresetPicker(c *ui.Context, state onboardingMoreViewState, actions onboardingActions) {
	t := c.Theme()
	settingRow(c, "ob-preset", i18n.T("Preset"), i18n.T("You can change them later in Settings → Shortcuts."), func() {
		options := make([]string, len(state.PresetOptions))
		for index, option := range state.PresetOptions {
			options[index] = option.Label
		}
		selected := state.PresetSelected
		if ui.Select(c.Key("ob-preset-sel"), &selected, options).Label(i18n.T("Shortcut preset")).Width(px(t, 220)).Changed() {
			for _, option := range state.PresetOptions {
				if option.Label == selected {
					actions.applyShortcutPreset(option.ID)
				}
			}
		}
	})
	if state.PresetAbout != "" {
		caption(c, state.PresetAbout)
	}
	for _, warning := range state.PresetWarnings {
		warnLine(c, warning)
	}
}
