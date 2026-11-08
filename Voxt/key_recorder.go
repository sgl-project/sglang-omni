package main

// The shortcut field that takes keys as they are pressed (Custom… of a
// feature's, Paste the last result's, Cancel's and the note's
// shortcuts): a key with the modifiers held, saved at once. While it
// has the focus the app's shortcuts are off, so that the keys reach it
// instead of starting a recording. The window sees no fn, no side of a
// modifier and no mouse button: those are typed (Type the keys
// instead).

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// uiKeyNames are the window's keys in internal/hotkey's grammar.
var uiKeyNames = map[ui.Key]string{
	ui.KeyA: "A", ui.KeyB: "B", ui.KeyC: "C", ui.KeyD: "D", ui.KeyE: "E", ui.KeyF: "F", ui.KeyG: "G",
	ui.KeyH: "H", ui.KeyI: "I", ui.KeyJ: "J", ui.KeyK: "K", ui.KeyL: "L", ui.KeyM: "M", ui.KeyN: "N",
	ui.KeyO: "O", ui.KeyP: "P", ui.KeyQ: "Q", ui.KeyR: "R", ui.KeyS: "S", ui.KeyT: "T", ui.KeyU: "U",
	ui.KeyV: "V", ui.KeyW: "W", ui.KeyX: "X", ui.KeyY: "Y", ui.KeyZ: "Z",
	ui.Key0: "0", ui.Key1: "1", ui.Key2: "2", ui.Key3: "3", ui.Key4: "4",
	ui.Key5: "5", ui.Key6: "6", ui.Key7: "7", ui.Key8: "8", ui.Key9: "9",
	ui.KeyF1: "F1", ui.KeyF2: "F2", ui.KeyF3: "F3", ui.KeyF4: "F4", ui.KeyF5: "F5", ui.KeyF6: "F6",
	ui.KeyF7: "F7", ui.KeyF8: "F8", ui.KeyF9: "F9", ui.KeyF10: "F10", ui.KeyF11: "F11", ui.KeyF12: "F12",
	ui.KeyEnter: "Return", ui.KeyEscape: "Escape", ui.KeyBackspace: "Delete", ui.KeyTab: "Tab",
	ui.KeySpace: "Space", ui.KeyDelete: "ForwardDelete", ui.KeyHome: "Home", ui.KeyEnd: "End",
	ui.KeyPageUp: "PageUp", ui.KeyPageDown: "PageDown",
	ui.KeyLeft: "Left", ui.KeyRight: "Right", ui.KeyUp: "Up", ui.KeyDown: "Down",
	ui.KeyMinus: "-", ui.KeyEqual: "=", ui.KeyComma: ",", ui.KeyPeriod: ".", ui.KeySlash: "/",
	ui.KeySemicolon: ";", ui.KeyQuote: "'", ui.KeyBracketLeft: "[", ui.KeyBracketRight: "]",
	ui.KeyBackslash: "\\", ui.KeyBackquote: "`",
}

// pressedKeys are the keys of a key press in the grammar's spelling
// ("Control+Option+Space"), false for a key the grammar has no name for.
// The key is where it sits on the keyboard, as the helper matches it.
func pressedKeys(ev ui.InputEvent) (string, bool) {
	k := ev.PhysicalKey
	if k == ui.KeyUnknown {
		k = ev.Key
	}
	name, ok := uiKeyNames[k]
	if !ok {
		return "", false
	}
	var parts []string
	for _, m := range []struct {
		mod  ui.Modifiers
		name string
	}{{ui.Ctrl, "Control"}, {ui.Alt, "Option"}, {ui.Shift, "Shift"}, {ui.Super, "Command"}} {
		if ev.Mods&m.mod != 0 {
			parts = append(parts, m.name)
		}
	}
	ch, err := hotkey.Parse(strings.Join(append(parts, name), "+"))
	if err != nil {
		return "", false
	}
	return ch.String(), true
}

// keysField is a shortcut's Custom… field: the recorder, or the typed
// keys once the user asked to type them.
func (a *app) keysField(c *ui.Context, r *keyRow, label string, save func(keys string)) {
	if r.typing {
		a.customKeys(c, r, label, save)
		if ghostButton(c.Key("record-keys"), i18n.T("Press the keys instead")).Clicked() {
			r.typing, r.edited = false, false
		}
		return
	}
	a.keyRecorder(c, r, label, save)
}

// keyRecorder takes the keys as they are pressed and saves them at once.
// Tab and ⇧Tab without other modifiers move the focus on, as in any
// field.
func (a *app) keyRecorder(c *ui.Context, r *keyRow, label string, save func(keys string)) {
	t := c.Theme()
	k := tok(t)
	box := ui.Box(c.Key("record")).Focusable().AutoFocus().Role(ui.RoleTextField).Label(label).
		Padding(px(t, 5), px(t, sp2)).Radius(px(t, radiusControl)).MaxWidth(px(t, 360)).Cursor(ui.CursorPointer)
	focused := box.Focused()
	if focused {
		a.keysHeard = true
		box.Border(1, k.Accent)
	} else {
		box.Border(1, k.Line)
	}
	if box.Clicked() {
		box.Focus()
	}
	box.HandleInput(func(ev ui.InputEvent) bool {
		switch ev.Kind {
		case ui.InputText, ui.InputCompose, ui.InputCommand:
			return true // keys typing text come first as keys; menus' ⌘C and the like are not shortcuts here
		case ui.InputKeyDown:
		default:
			return false
		}
		if ev.Key == ui.KeyTab && ev.Mods&^ui.Shift == 0 {
			return false
		}
		if ev.Repeat {
			return true
		}
		keys, ok := pressedKeys(ev)
		r.unknown = !ok
		if ok {
			r.text, r.edited = keys, false
			save(keys)
		}
		return true
	})
	shown := uitext.HotkeyText(r.text)
	box.Value(shown).Children(func() {
		switch {
		case r.text != "":
			ui.Text(c, shown).Font("monospace").TextColor(k.Text)
		case focused:
			ui.Text(c, i18n.T("Press the keys…")).Font("monospace").TextColor(k.Text3)
		default:
			ui.Text(c, i18n.T("Click, then press the keys")).Font("monospace").TextColor(k.Text3)
		}
	})
	switch {
	case r.unknown:
		warnLine(c, i18n.T("Voxt cannot use that key. Press another, or type the keys instead."))
	case focused:
		caption(c, i18n.T("Press the keys you want, as ⌥Space. Each press replaces the last."))
	default:
		caption(c, i18n.T("Click the field and press the keys, as ⌥Space. fn, a left or right modifier and mouse buttons cannot be pressed here: type them instead."))
	}
	if ghostButton(c.Key("type-keys"), i18n.T("Type the keys instead")).Clicked() {
		r.typing, r.unknown = true, false
	}
}
