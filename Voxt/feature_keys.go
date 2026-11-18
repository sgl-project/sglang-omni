package main

import (
	"fmt"
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// keyRow is a shortcut's widget-local editing state.
type keyRow struct {
	sel          string
	custom       bool
	typing       bool
	text         string
	trig         int
	edited       bool
	unknown      bool
	recFocused   bool
	recordDenied bool
}

func featureShortcuts(c *ui.Context, feature string, state *featureShortcutsViewState, actions featureActions, local *featureUI) {
	t := c.Theme()
	prefix := state.Prefix
	if sectionTitleAdd(c, prefix+"-key-add", i18n.T("Shortcuts"), "", i18n.T("Add a shortcut")) {
		actions.addFeatureShortcut(feature)
	}
	if !state.HotkeysInstalled {
		notice(c, prefix+"-no-tap", noticeWarn,
			i18n.T("Without Accessibility, only shortcuts like ⌃⌥Space work. fn, single modifiers, double taps and long presses need it."),
			"", func() {
				if ui.Button(c.Key(prefix+"-open-perms"), i18n.T("Open Permissions")).Clicked() {
					actions.openFeaturePermissions()
				}
			}, nil)
	}
	card(c.Key(prefix + "-keys")).Gap(px(t, sp2)).Children(func() {
		if state.EmptyText != "" {
			caption(c, state.EmptyText)
		}
		for index := range state.Rows {
			if index >= len(local.rows) {
				break
			}
			if index > 0 {
				ui.Box(c).Height(1).Background(tok(t).Line).Shrink(0)
			}
			featureKeyRowView(c, state.Prefix, feature, index, &state.Rows[index], state.HotkeysInstalled, state.HelperRecordsKeys, actions, &local.rows[index])
		}
	})
	if state.Paste != nil {
		featurePasteCancelRow(c, "tr-paste", i18n.T("Paste the last result"), "", i18n.T("Keys to paste the last result"), i18n.T("Custom keys to paste the last result"), state.Paste,
			state.HelperRecordsKeys, actions, &local.paste, actions.chooseFeaturePasteLast, actions.saveFeaturePasteLast)
	}
	if state.Cancel != nil {
		featurePasteCancelRow(c, "tr-cancel", i18n.T("Cancel recording"), i18n.T("The recording stays in History, so you can retry."), i18n.T("Keys to cancel recording"), i18n.T("Custom keys to cancel recording"), state.Cancel,
			state.HelperRecordsKeys, actions, &local.cancel, actions.chooseFeatureCancelKeys, actions.saveFeatureCancelKeys)
	}
	trEndWordRow(c, state.EndWord, actions, &local.endWord)
	if state.RewriteHint {
		caption(c, i18n.T("With an answer in the card, the shortcut asks a follow-up and adds the text you select first."))
	}
}

func featureKeyRowView(c *ui.Context, prefix, feature string, index int, state *featureKeyRowViewState, hotkeysInstalled, helperRecords bool, actions featureActions, local *keyRow) {
	t := c.Theme()
	colors := tok(t)
	number := index + 1
	ui.Column(c.Key(fmt.Sprintf("%s-key-%d", prefix, index))).Gap(px(t, 6)).Children(func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			ui.Row(c).Wrap().Gap(px(t, sp2)).AlignItems(ui.Center).Grow(1).MinWidth(0).Children(func() {
				options := featureOptionLabels(state.Options)
				if !slices.Contains(options, local.sel) {
					local.sel = state.Value
				}
				if local.custom {
					local.sel = i18n.T("Custom…")
				}
				selectControl := ui.Select(c.Key("sel"), &local.sel, options).Label(i18n.T("Keys of shortcut %d", number)).Width(px(t, 180)).Shrink(1).MinWidth(px(t, 120))
				if selectControl.Changed() {
					for _, option := range state.Options {
						if option.Label != local.sel {
							continue
						}
						if option.Custom {
							local.custom, local.text = true, state.Keys
							return
						}
						actions.chooseFeatureShortcut(feature, index, option.Value)
						return
					}
				}
				tips := []string{
					i18n.T("Press to start, press again to stop."),
					i18n.T("Press twice quickly to start; once more to stop."),
					i18n.T("Record while you hold the keys; release to stop."),
				}
				segment := segmentedSome(c.Key("trig"), &local.trig, []string{i18n.T("Tap"), i18n.T("Double tap"), i18n.T("Long press")}, nil, tips).
					Label(i18n.T("Trigger of shortcut %d", number))
				if segment.Changed() {
					actions.setFeatureShortcutTrigger(feature, index, local.trig)
				}
			})
			if state.NeedsAccessibility {
				color := colors.Text3
				if !hotkeysInstalled {
					color = colors.Warning
				}
				iconMark(c, icons.Accessibility, color, i18n.T("Works only with Accessibility access."))
			}
			if iconButton(c.Key("remove"), icons.TrashBinMinimalistic, i18n.T("Remove shortcut %d", number)).Clicked() {
				actions.removeFeatureShortcut(feature, index)
			}
		})
		if local.custom {
			featureKeysField(c, local, i18n.T("Custom keys of shortcut %d", number), &state.Custom, helperRecords, actions, func(keys string) {
				actions.saveFeatureShortcutKeys(feature, index, keys)
			})
		}
		problemLines(c, state.Problems)
		problemLines(c, state.UsedBy)
	})
}

func featurePasteCancelRow(c *ui.Context, key, title, help, label, customLabel string, state *featureKeyRowViewState, helperRecords bool, actions featureActions, local *keyRow, choose, save func(string)) {
	t := c.Theme()
	ui.Column(c.Key(key)).Gap(px(t, 6)).Children(func() {
		settingRow(c, key+"-row", title, help, func() {
			options := featureOptionLabels(state.Options)
			if !slices.Contains(options, local.sel) {
				local.sel = state.Value
			}
			if local.custom {
				local.sel = i18n.T("Custom…")
			}
			if ui.Select(c.Key(key+"-sel"), &local.sel, options).Label(label).Width(px(t, 180)).Changed() {
				for _, option := range state.Options {
					if option.Label != local.sel {
						continue
					}
					if option.Custom {
						local.custom, local.text = true, state.Keys
						return
					}
					choose(option.Value)
					return
				}
			}
		})
		if local.custom {
			featureKeysField(c, local, customLabel, &state.Custom, helperRecords, actions, save)
		}
		problemLines(c, state.Problems)
		problemLines(c, state.UsedBy)
	})
}

func featureOptionLabels(options []featureOptionViewState) []string {
	labels := make([]string, len(options))
	for index, option := range options {
		labels[index] = option.Label
	}
	return labels
}

func featureKeysField(c *ui.Context, local *keyRow, label string, state *featureCustomKeysViewState, helperRecords bool, actions featureActions, save func(string)) {
	if local.typing {
		featureCustomKeys(c, local, label, state, actions, save)
		if ghostButton(c.Key("record-keys"), i18n.T("Press the keys instead")).Clicked() {
			local.typing, local.edited = false, false
		}
		return
	}
	featureKeyRecorder(c, local, label, state, helperRecords, actions, save)
}

var featureCustomKeysInputEvents = func(input ui.Element) (changed, submitted, focused bool) {
	return input.Changed(), input.Submitted(), input.Focused()
}

func featureCustomKeys(c *ui.Context, local *keyRow, label string, state *featureCustomKeysViewState, actions featureActions, save func(string)) {
	t := c.Theme()
	input := ui.TextInput(c.Key("custom"), &local.text).Placeholder("Control+Option+Space").Label(label).Font("monospace").MaxWidth(px(t, 360))
	changed, submitted, focused := featureCustomKeysInputEvents(input)
	if changed {
		*state = actions.typeFeatureShortcutKeys(local)
	}
	if local.edited && state.Parsed && state.InputText == local.text && (submitted || !focused) {
		local.edited = false
		local.text = state.Canonical
		save(state.Canonical)
	}
	if state.InputText != local.text {
		return
	}
	switch {
	case state.ErrorText != "":
		warnLine(c, state.ErrorText)
	case local.edited:
		caption(c, i18n.T("Press Return to use %s.", state.KeysText))
		return
	default:
		caption(c, i18n.T("Modifiers first, then one key: Fn, RightCommand, Control+Option+Space, Fn+F5."))
	}
	caption(c, i18n.T("Mouse buttons work too: Mouse3 (middle), Mouse4 and up (side), also with modifiers (Option+Mouse4)."))
}

func featureKeyRecorder(c *ui.Context, local *keyRow, label string, state *featureCustomKeysViewState, helperRecords bool, actions featureActions, save func(string)) {
	t := c.Theme()
	colors := tok(t)
	box := ui.Box(c.Key("record")).Focusable().AutoFocus().Role(ui.RoleTextField).Label(label).
		Padding(px(t, 5), px(t, sp2)).Radius(px(t, radiusControl)).MaxWidth(px(t, 360)).Cursor(ui.CursorPointer)
	focused := box.Focused()
	if focused && !local.recFocused {
		local.recordDenied = false
		actions.beginHotkeyRecord(local, save)
	}
	if local.recFocused && !focused {
		actions.cancelHotkeyRecord()
	}
	local.recFocused = focused
	if focused {
		actions.markFeatureKeysHeard()
		box.Border(1, colors.Accent)
	} else {
		box.Border(1, colors.Line)
	}
	if box.Clicked() {
		box.Focus()
	}
	useWindow := !helperRecords
	box.HandleInput(func(event ui.InputEvent) bool {
		if !useWindow {
			if event.Kind == ui.InputKeyDown && event.Key == ui.KeyTab && event.Mods&^ui.Shift == 0 {
				return false
			}
			return event.Kind == ui.InputKeyDown || event.Kind == ui.InputText || event.Kind == ui.InputCompose || event.Kind == ui.InputCommand
		}
		switch event.Kind {
		case ui.InputText, ui.InputCompose, ui.InputCommand:
			return true
		case ui.InputKeyDown:
		default:
			return false
		}
		if event.Key == ui.KeyTab && event.Mods&^ui.Shift == 0 {
			return false
		}
		if event.Repeat {
			return true
		}
		actions.recordFeatureKeys(local, event, save)
		return true
	})
	shown := state.KeysText
	if state.InputText != local.text {
		shown = local.text
	}
	box.Value(shown).Children(func() {
		switch {
		case local.text != "":
			ui.Text(c, shown).Font("monospace").TextColor(colors.Text)
		case focused:
			ui.Text(c, i18n.T("Press the keys…")).Font("monospace").TextColor(colors.Text3)
		default:
			ui.Text(c, i18n.T("Click, then press the keys")).Font("monospace").TextColor(colors.Text3)
		}
	})
	switch {
	case local.recordDenied:
		warnLine(c, i18n.T("Recording a shortcut needs Accessibility access."))
		ui.Row(c).Children(func() {
			if ghostButton(c.Key("record-open-perms"), i18n.T("Open Permissions")).Clicked() {
				actions.openFeaturePermissions()
			}
		})
	case local.unknown:
		warnLine(c, i18n.T("Voxt cannot use that key. Press another, or type the keys instead."))
	case focused:
		if helperRecords {
			caption(c, i18n.T("Press the shortcut you want, including fn, a left or right modifier, or a mouse button. Each press replaces the last."))
		} else {
			caption(c, i18n.T("Press the keys you want, such as ⌥Space. Each press replaces the last."))
		}
	default:
		if helperRecords {
			caption(c, i18n.T("Click the field, then press the shortcut, including fn, a left or right modifier, or a mouse button."))
		} else {
			caption(c, i18n.T("Click the field and press the keys, such as ⌥Space. To use fn, a left or right modifier or a mouse button, allow Accessibility in Permissions."))
		}
	}
	if ghostButton(c.Key("type-keys"), i18n.T("Type the keys instead")).Clicked() {
		local.typing, local.unknown = true, false
	}
}

func problemLines(c *ui.Context, problems []string) {
	for _, problem := range problems {
		warnLine(c, problem)
	}
}

func warnLine(c *ui.Context, text string) {
	t := c.Theme()
	colors := tok(t)
	ui.Row(c).Gap(px(t, 4)).AlignItems(ui.Start).Children(func() {
		ui.Icon(c, icons.DangerTriangle.Linear).Size(px(t, 14), px(t, 14)).TextColor(colors.Warning).Shrink(0).Margin(px(t, 1), 0, 0, 0)
		ui.Text(c, text).FontSize(px(t, fsCaption)).TextColor(colors.Warning).Grow(1).MinWidth(0)
	})
}
