package main

// Settings → Shortcuts (/settings/shortcuts): every global shortcut
// in one table (Transcribe's, Translate's and Rewrite's, Paste the
// last result, Cancel transcription and Notes'), with their trigger and what
// hotkey.Check finds wrong:
// clashes, keys macOS uses, keys that need Accessibility while the event
// tap is missing, keys another ActNow app uses too. A row opens its
// feature's page, where it changes. Above it, the preset sets several at
// once (internal/settings' Presets), and Strict shortcuts.

import (
	"strconv"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type settingsShortcutsLocalUI struct {
	strictShortcuts bool
}

// shortcutsSettings is the Shortcuts section of the settings dialog.
func shortcutsSettings(c *ui.Context, state settingsShortcutsViewState, actions settingsShortcutsActions, local *settingsShortcutsLocalUI) {
	t := c.Theme()
	k := tok(t)
	caption(c, i18n.T("Click a shortcut to change it on its feature's page."))
	if !state.AccessibilityInstalled {
		notice(c, "sc-no-tap", noticeWarn,
			i18n.T("Without Accessibility, only shortcuts like %s work. fn, single modifiers, double taps and long presses need it.", uitext.HotkeyText("Control+Option+Space")),
			"", func() {
				if ui.Button(c.Key("sc-open-perms"), i18n.T("Open Permissions")).Clicked() {
					actions.openShortcutPermissions()
				}
			}, nil)
	}
	shortcutPresetPicker(c, "sc", state.Preset, actions)
	// Above the table, which fills the default window (#3916).
	settingRow(c, "sc-strict", i18n.T("Strict shortcuts"), i18n.T("While recording, only the shortcut that started it stops it. Off: any feature's shortcut stops."), func() {
		if ui.Switch(c.Key("sc-strict-switch"), &local.strictShortcuts).Label(i18n.T("Strict shortcuts")).Changed() {
			actions.setStrictShortcuts(local.strictShortcuts)
		}
	})
	keysWidth := px(t, 96)
	card(c.Key("sc-table")).Gap(0).Padding(px(t, sp2)).Children(func() {
		// The head.
		ui.Row(c).Gap(px(t, sp2)).Padding(px(t, sp1), px(t, sp2)).AlignItems(ui.Center).Children(func() {
			head := func(text string) ui.Element {
				return ui.Text(c, text).FontSize(px(t, fsMicro)).FontWeight(600).TextColor(k.Text3).SingleLine().Ellipsis("…").MinWidth(0)
			}
			head(i18n.T("Feature")).Grow(1)
			head(i18n.T("Shortcut")).Width(keysWidth).Shrink(0)
			ui.Box(c).Width(px(t, shortcutMarksW+14+sp2)).Shrink(0) // the marks and the chevron
		})
		for _, row := range state.Rows {
			shortcutRow(c, row, actions, state.AccessibilityInstalled, keysWidth)
		}
	})
}

// shortcutMarksW is the room of a row's marks (Accessibility, a problem).
const shortcutMarksW = 16 + sp2 + 16

// shortcutRow is a row of the table: its keys over its trigger, its
// problems under it; a click opens the feature's page.
func shortcutRow(c *ui.Context, row settingsShortcutRowViewState, actions settingsShortcutsActions, accessibilityInstalled bool, keysWidth float32) {
	t := c.Theme()
	k := tok(t)
	button := ui.ButtonBase(c.Key("sc-"+row.Feature+"-"+strconv.Itoa(row.Index))).Padding(px(t, 6), px(t, sp2)).Radius(px(t, radiusControl)).
		AlignItems(ui.Stretch).Label(row.Title + " " + row.KeyText).Tooltip(i18n.T("Change on the %s page", row.PageTitle))
	if button.Hovered() {
		button.Background(k.Hover)
	}
	button.Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 4)).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Grow(1).Basis(0).MinWidth(0).Children(func() {
					if row.Index == 0 {
						ui.Icon(c, shortcutIcon(row.Icon).Linear).Size(px(t, 16), px(t, 16)).TextColor(k.Text2).Shrink(0)
						truncText(c, row.Title)
					}
				})
				ui.Column(c).Width(keysWidth).Shrink(0).Gap(px(t, 2)).AlignItems(ui.Start).Children(func() {
					if !row.HasShortcut {
						ui.Text(c, row.KeyText).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine()
						return
					}
					kbd(c, row.KeyText).MinWidth(0).Ellipsis("…") // shell_commands.go
					ui.Text(c, row.TriggerText).FontSize(px(t, fsMicro)).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0)
				})
				ui.Row(c).Width(px(t, shortcutMarksW)).Shrink(0).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
					if row.NeedsAccessibility {
						color := k.Text3
						if !accessibilityInstalled {
							color = k.Warning
						}
						iconMark(c, icons.Accessibility, color, i18n.T("Works only with Accessibility access."))
					}
					if len(row.Problems) > 0 {
						iconMark(c, icons.DangerTriangle, k.Warning, strings.Join(row.Problems, "\n"))
					}
				})
				ui.Icon(c, icons.AltArrowRight.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text3).Shrink(0)
			})
			if len(row.Problems) > 0 {
				ui.Column(c).Padding(0, 0, 0, px(t, 16+sp2)).Gap(px(t, 2)).Children(func() {
					problemLines(c, row.Problems) // feature_keys.go
				})
			}
		})
	})
	if button.Clicked() {
		actions.openShortcutFeature(row.Path)
	}
}

func shortcutIcon(code string) icons.Icon {
	switch code {
	case "paste":
		return icons.Copy
	case "cancel":
		return icons.CloseCircle
	case "note":
		return icons.Notes
	case "translate":
		return icons.Global
	case "rewrite":
		return icons.PenNewRound
	default:
		return icons.Microphone3
	}
// presetTitle is a shortcut preset's name ("": custom).
func presetTitle(p string) string {
	switch p {
	case settings.PresetOption:
		return i18n.T("%s (default)", uitext.HotkeyText("RightOption"))
	case settings.PresetFn:
		return i18n.T("fn (Auralink's keys)")
	case settings.PresetCommand:
		return i18n.T("%s combination", uitext.HotkeyText("RightCommand"))
	case settings.PresetMouseMiddle:
		return i18n.T("Middle mouse + %s", uitext.HotkeyText("Fn+Shift"))
	}
	return i18n.T("Custom")
}

// presetAbout is one line on who a preset suits (setup guide).
func presetAbout(p string) string {
	switch p {
	case settings.PresetOption:
		return i18n.T("%s: one-handed and Voxt's default. Good when fn is the Globe key or shared with Auralink.", uitext.HotkeyText("RightOption"))
	case settings.PresetFn:
		return i18n.T("fn: matches Auralink and many Voxt 1.x setups. Watch for clashes with the Globe key and input sources.")
	case settings.PresetCommand:
		return i18n.T("%s combination: %s only, away from the letter keys. Good if you keep Option for typing.", uitext.HotkeyText("RightCommand"), uitext.HotkeyText("RightCommand"))
	case settings.PresetMouseMiddle:
		return i18n.T("Middle mouse + %s: click the wheel to transcribe, double-click to rewrite. Mouse shortcuts need Accessibility.", uitext.HotkeyText("Fn+Shift"))
	}
	return ""
}

// shortcutPresetPicker is the preset dropdown and warnings (settings and
// the setup guide). key prefixes control ids ("sc" or "ob").
func shortcutPresetPicker(c *ui.Context, key string, state shortcutPresetPickerViewState, actions shortcutPresetActions) {
	t := c.Theme()
	selected := state.Selected
	options := make([]string, len(state.Options))
	selectedAbout := ""
	for index, option := range state.Options {
		options[index] = option.Title
		if option.Title == state.Selected {
			selectedAbout = option.About
		}
	}
	settingRow(c, key+"-preset", i18n.T("Preset"), state.About, func() {
		if ui.Select(c.Key(key+"-preset-sel"), &selected, options).Label(i18n.T("Shortcut preset")).Width(px(t, 220)).Changed() {
			for _, option := range state.Options {
				if option.ID != "" && option.Title == selected {
					actions.applyShortcutPreset(option.ID)
				}
			}
		}
	})
	if selectedAbout != "" && key == "ob" {
		caption(c, selectedAbout)
	}
	for _, warning := range state.Warnings {
		warnLine(c, warning)
	}
}
