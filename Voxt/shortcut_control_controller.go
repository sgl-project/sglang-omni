package main

// The one shortcut editor open at a time (shortcut_control.go).

// openShortcutEditor edits row, whose saved keys are keys; an editor open
// elsewhere closes first.
func (a *app) openShortcutEditor(row *keyRow, keys string) {
	a.closeShortcutEditor()
	a.shortcutEditing, a.shortcutShown = row, true
	row.text, row.before = keys, keys
	row.custom, row.typing, row.edited, row.unknown, row.recFocused, row.recordDenied = true, false, false, false, false, false
}

func (a *app) markShortcutEditorShown() { a.shortcutShown = true }

// closeShortcutEditor closes the open editor with its saved keys and stops
// a capture on its way.
func (a *app) closeShortcutEditor() {
	if row := a.shortcutEditing; row != nil {
		if row.custom {
			row.text = row.before
		}
		row.custom, row.typing, row.edited, row.recFocused = false, false, false, false
		a.shortcutEditing = nil
	}
	a.cancelHotkeyRecord()
}

// endShortcutEditorFrame closes an editor no view drew this frame (its page
// or dialog closed), so that it does not capture keys again when it reopens.
func (a *app) endShortcutEditorFrame() {
	if !a.shortcutShown && a.shortcutEditing != nil {
		a.closeShortcutEditor()
	}
	a.shortcutShown = false
}
