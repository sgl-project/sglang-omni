package main

// A shortcut's keys as a button (the feature pages, Notes and Settings →
// Shortcuts): a click records new keys in its place, Cancel keeps the saved
// ones, and a shortcut that can be off has a button that clears it.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// shortcutEditorActions keep one shortcut editor open at a time.
type shortcutEditorActions interface {
	// openShortcutEditor edits row, whose saved keys are keys.
	openShortcutEditor(row *keyRow, keys string)
	// closeShortcutEditor closes the open editor and keeps the saved keys.
	closeShortcutEditor()
	// markShortcutEditorShown says the open editor was drawn this frame.
	markShortcutEditorShown()
}

// shortcutControl shows keysText, the saved keys ("" for none), as a
// button labelled label. A click opens editor in its place: it records or
// types the keys and saves them through the save it is given, which also
// closes it. With clear, a button removes the saved keys.
func shortcutControl(c *ui.Context, row *keyRow, label, keys, keysText string, clear bool, actions shortcutEditorActions, save func(string), editor func(save func(string))) {
	t := c.Theme()
	k := tok(t)
	ui.Column(c.Key("shortcut-control")).Width(px(t, 180)).Shrink(1).MinWidth(px(t, 120)).Gap(px(t, 4)).Children(func() {
		if row.custom {
			actions.markShortcutEditorShown()
			editor(func(keys string) {
				save(keys)
				row.custom, row.typing, row.recFocused = false, false, false
				actions.closeShortcutEditor()
			})
			ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).Children(func() {
				if row.typing {
					if ghostButton(c.Key("record-keys"), i18n.T("Record instead")).Clicked() {
						row.typing, row.edited = false, false
					}
				} else if ghostButton(c.Key("type-keys"), i18n.T("Type instead")).Clicked() {
					row.typing, row.unknown = true, false
				}
				if iconButton(c.Key("cancel-record"), icons.CloseCircle, i18n.T("Cancel")).Clicked() {
					actions.closeShortcutEditor()
				}
			})
			return
		}
		shown := keysText
		if shown == "" {
			shown = i18n.T("None")
		}
		ui.Row(c).Gap(px(t, 4)).AlignItems(ui.Center).Children(func() {
			button := ui.ButtonBase(c.Key("record-shortcut")).Grow(1).MinWidth(0).Height(px(t, 30)).Padding(0, px(t, sp2)).Radius(px(t, radiusControl)).
				Border(1, k.Line).Background(k.Subtle).Label(label).Tooltip(i18n.T("Click, then press the keys"))
			if button.Hovered() {
				button.Background(k.Hover)
			}
			button.Children(func() {
				ui.Text(c, shown).Font("monospace").TextColor(k.Text).SingleLine().Ellipsis("…").MinWidth(0)
			})
			if button.Clicked() {
				actions.openShortcutEditor(row, keys)
			}
			if clear && keys != "" && iconButton(c.Key("off"), icons.CloseCircle, i18n.T("Clear shortcut")).Clicked() {
				save("")
			}
		})
	})
}
