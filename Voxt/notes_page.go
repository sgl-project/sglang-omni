package main

// The Notes page (/notes): stored notes and the note shortcut controls.

import (
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type notesLocalUI struct {
	listUI  ui.ListState
	keys    keyRow
	keysFor string
}

func notesPage(c *ui.Context, state *notesViewState, actions notesActions, local *notesLocalUI) {
	t := c.Theme()
	widePageHeader(c, i18n.T("Notes"), func() {
		if state.CountLabel != "" {
			caption(c, state.CountLabel).SingleLine()
		}
	})
	pageBody(c).Children(func() {
		ui.Column(c).Padding(px(t, sp1), px(t, sp6), px(t, sp2)).Gap(px(t, sp2)).Shrink(0).Children(func() {
			noteKeysRow(c, &state.Shortcut, actions, local)
			if state.Error != "" {
				notice(c, "notes-err", noticeError, i18n.T("Could not read the notes."), state.Error, func() {
					if ghostButton(c.Key("notes-retry"), i18n.T("Try again")).Clicked() {
						actions.loadNotes()
					}
				}, nil)
			}
		})
		ui.Column(c).Grow(1).MinHeight(0).Children(func() {
			switch {
			case !state.Read:
			case len(state.Rows) == 0:
				emptyState(c, notesIcon, i18n.T("No notes yet"), state.Shortcut.EmptyHint, nil)
			default:
				ui.List(c.Key("notes-list"), &local.listUI, len(state.Rows)).Grow(1).MinHeight(0).Gap(px(t, 2)).Padding(0, 0, px(t, sp4)).
					ItemKey(func(index int) any { return state.Rows[index].ID }).Rows(func(row ui.ListRow) {
					noteRow(c, state.Rows[row.Index], actions)
				})
			}
		})
	})
}

func noteRow(c *ui.Context, note notesRowViewState, actions notesActions) {
	t := c.Theme()
	colors := tok(t)
	row := ui.Row(c.Key("note-"+note.ID)).Padding(px(t, 6), px(t, sp2)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
		Gap(px(t, sp2)).AlignItems(ui.Start)
	if row.Hovered() {
		row.Background(colors.Hover)
	}
	row.Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
			ui.Text(c, note.Text).Selectable()
			caption(c, note.TimeText).SingleLine()
		})
		if iconButton(c.Key("delete"), icons.TrashBinMinimalistic, note.DeleteText).Clicked() {
			actions.deleteDictionaryNote(note.ID, note.Text)
		}
	})
}

func noteKeysRow(c *ui.Context, state *notesShortcutViewState, actions notesActions, local *notesLocalUI) {
	t := c.Theme()
	row := &local.keys
	ui.Column(c.Key("notes-keys")).Gap(px(t, 6)).Children(func() {
		settingRow(c, "notes-keys-row", i18n.T("Shortcut"), "", func() {
			options := make([]string, 0, len(state.Options))
			byLabel := map[string]string{}
			for _, option := range state.Options {
				options = append(options, option.Label)
				byLabel[option.Label] = option.Keys
			}
			if !slices.Contains(options, row.sel) {
				row.sel = state.CurrentText
			}
			if row.custom {
				row.sel = i18n.T("Custom…")
			}
			if ui.Select(c.Key("notes-keys-sel"), &row.sel, options).Label(i18n.T("Keys to record a note")).Width(px(t, 180)).Changed() {
				if row.sel == i18n.T("Custom…") {
					row.custom, row.text = true, state.Keys
					return
				}
				keys, ok := byLabel[row.sel]
				if !ok {
					return
				}
				row.custom, row.text = false, keys
				actions.setNoteKeys(keys)
			}
		})
		if row.custom {
			notesKeysField(c, state, actions, local)
		}
		if state.SetHint {
			caption(c, i18n.T("Set a shortcut to record notes."))
		}
		for _, problem := range state.Problems {
			warnLine(c, problem)
		}
		for _, usedBy := range state.UsedBy {
			warnLine(c, usedBy)
		}
	})
}

func notesKeysField(c *ui.Context, state *notesShortcutViewState, actions notesActions, local *notesLocalUI) {
	if local.keys.typing {
		notesCustomKeys(c, state, actions, local)
		if ghostButton(c.Key("record-keys"), i18n.T("Press the keys instead")).Clicked() {
			local.keys.typing, local.keys.edited = false, false
		}
		return
	}
	noteKeyRecorder(c, state, actions, local)
}

func notesCustomKeys(c *ui.Context, state *notesShortcutViewState, actions notesActions, local *notesLocalUI) {
	t := c.Theme()
	row := &local.keys
	input := ui.TextInput(c.Key("custom"), &row.text).Placeholder("Control+Option+Space").Label(i18n.T("Custom keys to record a note")).Font("monospace").MaxWidth(px(t, 360))
	if input.Changed() {
		*state = actions.updateNoteKeysDraft(row.text)
	}
	if row.edited && state.ParseError == "" && (input.Submitted() || !input.Focused()) {
		row.edited = false
		row.text = state.ParsedKeys
		actions.setNoteKeys(row.text)
	}
	switch {
	case state.ParseError != "":
		warnLine(c, state.ParseError)
	case row.edited:
		caption(c, state.EditedHint)
		return
	default:
		caption(c, i18n.T("Modifiers first, then one key: Fn, RightCommand, Control+Option+Space, Fn+F5."))
	}
	caption(c, i18n.T("Mouse buttons work too: Mouse3 (middle), Mouse4 and up (side), also with modifiers (Option+Mouse4)."))
}

func noteKeyRecorder(c *ui.Context, state *notesShortcutViewState, actions notesActions, local *notesLocalUI) {
	t := c.Theme()
	colors := tok(t)
	row := &local.keys
	box := ui.Box(c.Key("record")).Focusable().AutoFocus().Role(ui.RoleTextField).Label(i18n.T("Custom keys to record a note")).
		Padding(px(t, 5), px(t, sp2)).Radius(px(t, radiusControl)).MaxWidth(px(t, 360)).Cursor(ui.CursorPointer)
	focused := box.Focused()
	if focused && !row.recFocused {
		row.recordDenied = false
	}
	actions.setNoteHotkeyFocus(focused)
	row.recFocused = focused
	if focused {
		box.Border(1, colors.Accent)
	} else {
		box.Border(1, colors.Line)
	}
	if box.Clicked() {
		box.Focus()
	}
	useWindow := !state.HelperRecords
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
		keys, ok := notesPressedKeys(event)
		row.unknown = !ok
		if ok {
			*state = actions.recordNoteKeys(keys)
		}
		return true
	})
	box.Value(state.EditingText).Children(func() {
		switch {
		case row.text != "":
			ui.Text(c, state.EditingText).Font("monospace").TextColor(colors.Text)
		case focused:
			ui.Text(c, i18n.T("Press the keys…")).Font("monospace").TextColor(colors.Text3)
		default:
			ui.Text(c, i18n.T("Click, then press the keys")).Font("monospace").TextColor(colors.Text3)
		}
	})
	switch {
	case row.recordDenied:
		warnLine(c, i18n.T("Recording a shortcut needs Accessibility access."))
		ui.Row(c).Children(func() {
			if ghostButton(c.Key("record-open-perms"), i18n.T("Open Permissions")).Clicked() {
				actions.openNotePermissions()
			}
		})
	case row.unknown:
		warnLine(c, i18n.T("Voxt cannot use that key. Press another, or type the keys instead."))
	case focused:
		if state.HelperRecords {
			caption(c, i18n.T("Press the shortcut you want, including fn, a left or right modifier, or a mouse button. Each press replaces the last."))
		} else {
			caption(c, i18n.T("Press the keys you want, such as ⌥Space. Each press replaces the last."))
		}
	default:
		if state.HelperRecords {
			caption(c, i18n.T("Click the field, then press the shortcut, including fn, a left or right modifier, or a mouse button."))
		} else {
			caption(c, i18n.T("Click the field and press the keys, such as ⌥Space. To use fn, a left or right modifier or a mouse button, allow Accessibility in Permissions."))
		}
	}
	if ghostButton(c.Key("type-keys"), i18n.T("Type the keys instead")).Clicked() {
		row.typing, row.unknown = true, false
	}
}

func notesPressedKeys(event ui.InputEvent) (string, bool) {
	key := event.PhysicalKey
	if key == ui.KeyUnknown {
		key = event.Key
	}
	name, ok := uiKeyNames[key]
	if !ok {
		return "", false
	}
	var parts []string
	for _, modifier := range []struct {
		mask ui.Modifiers
		name string
	}{{ui.Ctrl, "Control"}, {ui.Alt, "Option"}, {ui.Shift, "Shift"}, {ui.Super, "Command"}} {
		if event.Mods&modifier.mask != 0 {
			parts = append(parts, modifier.name)
		}
	}
	return strings.Join(append(parts, name), "+"), true
}
