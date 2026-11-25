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
	listUI ui.ListState
	// editID is the note edited in place, as draft; saving: its save runs.
	editID  string
	draft   string
	saving  bool
	keys    keyRow
	keysFor string
}

func notesPage(c *ui.Context, state *notesViewState, actions notesActions, local *notesLocalUI) {
	t := c.Theme()
	widePageHeader(c, i18n.T("Notes"), func() {
		if state.CountLabel != "" {
			caption(c, state.CountLabel).SingleLine()
		}
		if !state.Banner { // the banner has the button while it shows
			noteRecordButton(c, state, actions, false)
		}
	})
	pageBody(c).Children(func() {
		ui.Column(c).Padding(px(t, sp1), px(t, sp6), px(t, sp2)).Gap(px(t, sp3)).Shrink(0).Children(func() {
			if state.Banner {
				notesBanner(c, state, actions)
			}
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
				hint := state.Shortcut.EmptyHint
				if state.Banner {
					hint = "" // the banner says it
				}
				emptyState(c, notesIcon, i18n.T("No notes yet"), hint, nil)
			default:
				ui.List(c.Key("notes-list"), &local.listUI, len(state.Rows)).Grow(1).MinHeight(0).Gap(px(t, 2)).Padding(0, 0, px(t, sp4)).
					ItemKey(func(index int) any { return state.Rows[index].ID }).Rows(func(row ui.ListRow) {
					noteRow(c, state.Rows[row.Index], actions, local)
				})
			}
		})
	})
}

// The banner: its height and corners, and the window's width from which
// the note drawn on its far side has room beside the text.
const (
	notesBannerH      = 196
	notesBannerRadius = 18
	notesBannerWide   = 760
)

// notesBanner is the page's introduction, until it is closed or a note is
// saved (Setup.NotesIntroDone): what a note is for, how to record one
// (the shortcut's line) and the button that records one, white on the
// picture the overlay's preview has, with a note drawn on its far side.
func notesBanner(c *ui.Context, state *notesViewState, actions notesActions) {
	t := c.Theme()
	white := ui.Hex("#ffffff")
	windowW, _ := c.Size()
	ui.Box(c.Key("notes-banner")).FillWidth().MinWidth(0).Height(px(t, notesBannerH)).Shrink(0).Radius(px(t, notesBannerRadius)).Clip().
		Background(ui.RGB(72, 96, 214)).Children(func() {
		if picture := ovPreviewBackground(); picture != nil {
			ui.Image(c, picture).Absolute().Left(0).Top(0).Right(0).Bottom(0).Fit(ui.Cover)
		}
		// Darker where the text is: the picture's far side is light.
		ui.Box(c).Absolute().Left(0).Top(0).Right(0).Bottom(0).Gradient(ui.RGBA(10, 14, 60, 0.34), ui.RGBA(10, 14, 60, 0.06), 90)
		// Positioned as the picture is, to draw over it.
		ui.Row(c).Absolute().Left(0).Top(0).Right(0).Bottom(0).AlignItems(ui.Stretch).Children(func() {
			ui.Column(c).Grow(1).MinWidth(0).Padding(0, px(t, sp6)).Gap(px(t, sp2)).Justify(ui.Center).Children(func() {
				ui.Text(c, i18n.T("Say it. Keep it.")).FontSize(px(t, 26)).FontWeight(650).TextColor(white)
				ui.Text(c, state.Shortcut.EmptyHint).TextColor(white.Alpha(0.9)).MaxWidth(px(t, 440))
				ui.Row(c).Margin(px(t, sp3), 0, 0).Children(func() {
					noteRecordButton(c, state, actions, true)
				})
			})
			if windowW >= notesBannerWide {
				notesBannerNote(c)
			}
		})
		// Close, over the picture's corner.
		tip := i18n.T("Close the introduction")
		closing := ghostBase(c.Key("notes-banner-close")).Size(px(t, 22), px(t, 22)).Center().Label(tip).Tooltip(tip).
			Absolute().Top(px(t, 6)).Right(px(t, 6)).Radius(px(t, 11))
		closing.Background(ui.RGBA(20, 20, 60, 0.24))
		if closing.Hovered() {
			closing.Background(ui.RGBA(20, 20, 60, 0.44))
		}
		closing.Children(func() {
			ui.Icon(c, icons.Close.Linear).Size(px(t, 12), px(t, 12)).TextColor(white)
		})
		if closing.Clicked() {
			actions.closeNotesBanner()
		}
	})
}

// noteRecordButton is Record a note, Stop and save while one records:
// white on the banner's picture, a secondary button in the page's header.
func noteRecordButton(c *ui.Context, state *notesViewState, actions notesActions, onPicture bool) {
	t := c.Theme()
	label := i18n.T("Record a note")
	if state.Recording {
		label = i18n.T("Stop and save")
	}
	var button ui.Element
	if onPicture {
		white := *t
		white.Accent, white.AccentText = ui.Hex("#ffffff"), ui.Hex("#1b1b23")
		white.AccentHover, white.AccentPressed = ui.Hex("#ececf4"), ui.Hex("#dcdce8")
		c.SetTheme(&white)
		button = ui.PrimaryButton(c.Key("notes-record"), "").Border(1, ui.Color{})
	} else {
		button = ui.Button(c.Key("notes-record"), "")
	}
	button.Label(label).Children(func() {
		if state.Recording {
			statusDot(c, tok(t).Record)
		} else {
			ui.Icon(c, icons.VoxtMicrophone.Bold).Size(labelIcon(t), labelIcon(t))
		}
		ui.Text(c, label).SingleLine()
	})
	c.SetTheme(t)
	if button.Clicked() {
		actions.toggleNoteRecording()
	}
}

// notesBannerNote is the note drawn on the banner's far side, a sheet
// that comes up from the banner's bottom edge. Its note is the first
// words a machine recorded and played back, Edison's to his phonograph
// in 1877: the first voice note. A quotation, so in English in every
// language; who, when and what it was where a note has its time and app.
// Lines stand for the note under it.
func notesBannerNote(c *ui.Context) {
	t := c.Theme()
	ink := ui.Hex("#1b1b23")
	ui.Column(c).Width(px(t, 208)).Shrink(0).Margin(px(t, sp6), px(t, sp6), 0, 0).Padding(px(t, sp3), px(t, 14), px(t, sp4)).Gap(px(t, 4)).
		Radius(px(t, 12), px(t, 12), 0, 0).Background(ui.RGBA(255, 255, 255, 0.95)).Shadow(0, px(t, 8), px(t, 24), 0, ui.RGBA(20, 20, 60, 0.22)).Children(func() {
		ui.Text(c, "Mary had a little lamb,\nits fleece was white as snow.").FontSize(px(t, fsCaption)).FontWeight(600).TextColor(ink)
		ui.Text(c, i18n.T("Edison · 1877 · first playback")).FontSize(px(t, fsMicro)).TextColor(ink.Alpha(0.56)).MaxLines(2)
		ui.Box(c).FillWidth().Height(1).Margin(px(t, 8), 0, px(t, 6)).Background(ink.Alpha(0.1))
		for _, width := range []float32{172, 150, 104} {
			ui.Box(c).Width(px(t, width)).Height(px(t, 6)).Radius(px(t, 3)).Margin(px(t, 3), 0).Background(ink.Alpha(0.16))
		}
	})
}

// noteRow is a note in the list: its text (selectable), its time and
// source app, and Copy, Edit and Delete.
func noteRow(c *ui.Context, note notesRowViewState, actions notesActions, local *notesLocalUI) {
	t := c.Theme()
	colors := tok(t)
	editing := local.editID == note.ID
	row := ui.Row(c.Key("note-"+note.ID)).Padding(px(t, 6), px(t, sp2)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
		Gap(px(t, sp2)).AlignItems(ui.Start)
	if row.Hovered() {
		row.Background(colors.Hover)
	}
	row.Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
			if editing {
				ui.TextArea(c.Key("edit-text"), &local.draft).Label(note.EditText).Lines(2, 8).AutoFocus().Disabled(local.saving)
				ui.Row(c).Gap(px(t, sp2)).Children(func() {
					ui.Spacer(c)
					if ghostButton(c.Key("cancel"), i18n.T("Cancel")).Disabled(local.saving).Clicked() {
						local.editID, local.draft = "", ""
					}
					if primaryButton(c.Key("save"), i18n.T("Save")).Disabled(local.saving).Clicked() {
						actions.saveNoteEdit(note.ID, local.draft)
					}
				})
			} else {
				ui.Text(c, note.Text).Selectable()
			}
			meta := note.TimeText
			if note.AppName != "" {
				meta += " · " + note.AppName
			}
			caption(c, meta).SingleLine()
		})
		if editing {
			return
		}
		if iconButton(c.Key("copy"), icons.Copy, i18n.T("Copy")).Label(note.CopyText).Clicked() {
			actions.copyNote(note.Text)
		}
		if iconButton(c.Key("edit"), icons.Pen2, note.EditText).Disabled(local.editID != "").Clicked() {
			local.editID, local.draft = note.ID, note.Text
		}
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
