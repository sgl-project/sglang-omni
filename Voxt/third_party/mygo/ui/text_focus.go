package ui

// TextFocused reports whether the keyboard focus is in a text field the
// user can type into (enabled, not read-only), in a window that has the
// keyboard focus. An app whose global shortcut types text reads it to
// tell whether that text should go into its own window. It reads the
// committed window, between frames too; run it on the UI thread.
func (s Services) TextFocused() bool {
	rt := s.owner.Value()
	if rt == nil || rt.closed || !rt.windowFocus() {
		return false
	}
	st := rt.states[rt.focused]
	return st != nil && st.editor != nil && !st.editor.readOnly && st.flags&flagEditable != 0 && st.flags&flagDisabled == 0
}
