package terminal

import (
	"runtime"

	"github.com/egoist/mygo/ui"
)

// Action is what a key bound to it does (Keybind), as an action of
// Ghostty's keybind option.
type Action int

const (
	// ActionNone unbinds a key, which then goes to the program, as
	// Ghostty's unbind.
	ActionNone Action = iota
	// ActionIgnore takes the key and does nothing, as Ghostty's ignore.
	ActionIgnore
	// ActionCopy copies the selection, as copy_to_clipboard. It does
	// something only with a selection, and when Performable it also lets
	// go of it, so that the key's next press reaches the program, as
	// Windows Terminal's Control+C.
	ActionCopy
	// ActionPaste pastes the clipboard, as paste_from_clipboard: as the
	// view's Paste does (see Options.OnPaste). It does something only when
	// the clipboard holds something.
	ActionPaste
	// ActionSelectAll selects the scrollback and the screen, as
	// select_all.
	ActionSelectAll
	// ActionClearScreen clears the scrollback and the screen above the
	// cursor's line, which moves to the top, as clear_screen (and Kitty's
	// clear_terminal to_cursor). It does nothing on a full-screen
	// program's screen (the alternate screen), which it would break.
	ActionClearScreen
	// ActionIncreaseFontSize, ActionDecreaseFontSize and
	// ActionResetFontSize zoom a step in or out, or back to the font's
	// size, as increase_font_size:1, decrease_font_size:1 and
	// reset_font_size (see Options.OnZoom).
	ActionIncreaseFontSize
	ActionDecreaseFontSize
	ActionResetFontSize
	// ActionScrollPageUp, ActionScrollPageDown, ActionScrollToTop and
	// ActionScrollToBottom scroll the scrollback, as scroll_page_up,
	// scroll_page_down, scroll_to_top and scroll_to_bottom. They do
	// nothing on a full-screen program's screen, which has none.
	ActionScrollPageUp
	ActionScrollPageDown
	ActionScrollToTop
	ActionScrollToBottom
)

// Keybind binds a key, with exactly Mods held, to an action, as a line of
// Ghostty's keybind option: the view does the action instead of sending
// the key to the program. Key follows the keyboard layout, as ui.Key
// does.
//
// Performable, as Ghostty's performable: prefix, takes the key only when
// the action does something: Control+C copies a selection, and without
// one goes on to the program, which it interrupts. A key the action leaves
// goes on as if unbound.
type Keybind struct {
	Mods        ui.Modifiers
	Key         ui.Key
	Action      Action
	Performable bool
}

// DefaultKeybinds returns the keys a terminal binds unless
// Options.NoDefaultKeybinds, which are Ghostty's of the platform:
//
//   - on macOS, Command+C copies and Command+V pastes, Command+A selects
//     all, Command+K clears the screen, Command+= (and +) and Command+-
//     zoom, and Command+0 zooms back; Command+Home, End, Page Up and
//     Page Down scroll, as do Shift+Page Up and Page Down;
//   - elsewhere, Control+Shift+C and Control+Insert copy, Control+Shift+V
//     and Shift+Insert paste, Control+Shift+A selects all, Control+= (and
//     +), Control+- and Control+0 zoom, and Shift+Page Up and Page Down
//     scroll, as do Shift+Home and End on Linux;
//   - on Windows, also Windows Terminal's: Control+C copies when there is a
//     selection (performable), Control+V pastes when the clipboard holds
//     something (performable), Control+Shift+K clears the screen, and
//     Control+Shift+Home, End, Page Up and Page Down scroll.
//
// The keys that clear the screen and scroll are performable: a
// full-screen program, as an editor, gets them.
func DefaultKeybinds() []Keybind {
	return defaultKeybinds(runtime.GOOS)
}

func defaultKeybinds(goos string) []Keybind {
	var b []Keybind
	bind := func(mods ui.Modifiers, key ui.Key, a Action) {
		b = append(b, Keybind{Mods: mods, Key: key, Action: a})
	}
	performable := func(mods ui.Modifiers, key ui.Key, a Action) {
		b = append(b, Keybind{Mods: mods, Key: key, Action: a, Performable: true})
	}
	scroll := func(mods ui.Modifiers) {
		performable(mods, ui.KeyPageUp, ActionScrollPageUp)
		performable(mods, ui.KeyPageDown, ActionScrollPageDown)
		performable(mods, ui.KeyHome, ActionScrollToTop)
		performable(mods, ui.KeyEnd, ActionScrollToBottom)
	}
	if goos == "darwin" {
		cmd := ui.Super
		bind(cmd, ui.KeyC, ActionCopy)
		bind(cmd, ui.KeyV, ActionPaste)
		bind(cmd, ui.KeyA, ActionSelectAll)
		performable(cmd, ui.KeyK, ActionClearScreen)
		bind(cmd, ui.KeyEqual, ActionIncreaseFontSize)
		bind(cmd|ui.Shift, ui.KeyEqual, ActionIncreaseFontSize)
		bind(cmd, ui.KeyMinus, ActionDecreaseFontSize)
		bind(cmd, ui.Key0, ActionResetFontSize)
		scroll(cmd)
		performable(ui.Shift, ui.KeyPageUp, ActionScrollPageUp)
		performable(ui.Shift, ui.KeyPageDown, ActionScrollPageDown)
		return b
	}
	// The first key of an action is the one menus show.
	bind(ui.Ctrl|ui.Shift, ui.KeyC, ActionCopy)
	bind(ui.Ctrl|ui.Shift, ui.KeyV, ActionPaste)
	bind(ui.Ctrl, ui.KeyInsert, ActionCopy)
	bind(ui.Shift, ui.KeyInsert, ActionPaste)
	bind(ui.Ctrl|ui.Shift, ui.KeyA, ActionSelectAll)
	bind(ui.Ctrl, ui.KeyEqual, ActionIncreaseFontSize)
	bind(ui.Ctrl|ui.Shift, ui.KeyEqual, ActionIncreaseFontSize)
	bind(ui.Ctrl, ui.KeyMinus, ActionDecreaseFontSize)
	bind(ui.Ctrl, ui.Key0, ActionResetFontSize)
	if goos != "windows" {
		scroll(ui.Shift)
		return b
	}
	// Windows Terminal's, which leaves Shift+Home and End to the shell,
	// as PowerShell selects with them.
	performable(ui.Ctrl, ui.KeyC, ActionCopy)
	performable(ui.Ctrl, ui.KeyV, ActionPaste)
	performable(ui.Ctrl|ui.Shift, ui.KeyK, ActionClearScreen)
	performable(ui.Shift, ui.KeyPageUp, ActionScrollPageUp)
	performable(ui.Shift, ui.KeyPageDown, ActionScrollPageDown)
	scroll(ui.Ctrl | ui.Shift)
	return b
}

// keybinds returns the bindings of opts: its Keybinds over the defaults,
// each replacing those of its key, in order, the defaults first.
func keybinds(opts *Options) []Keybind {
	var b []Keybind
	if !opts.NoDefaultKeybinds {
		b = DefaultKeybinds()
	}
	for _, k := range opts.Keybinds {
		kept := b[:0]
		for _, o := range b {
			if o.Mods != k.Mods || o.Key != k.Key {
				kept = append(kept, o)
			}
		}
		b = kept
		if k.Action != ActionNone {
			b = append(b, k)
		}
	}
	return b
}

// binding returns the binding of a key.
func (t *Terminal) binding(mods ui.Modifiers, key ui.Key) (Keybind, bool) {
	for _, b := range t.binds {
		if b.Mods == mods && b.Key == key {
			return b, true
		}
	}
	return Keybind{}, false
}

// shortcut returns the key a menu item of an action shows: its first
// binding that always takes the key, as Ghostty's menus show no
// performable one, else its first.
func (t *Terminal) shortcut(a Action) (Keybind, bool) {
	var found Keybind
	ok := false
	for _, b := range t.binds {
		if b.Action != a {
			continue
		}
		if !b.Performable {
			return b, true
		}
		if !ok {
			found, ok = b, true
		}
	}
	return found, ok
}
