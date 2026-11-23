package terminal

import (
	"unicode"

	"github.com/egoist/mygo/plugins/terminal/internal/vt"
	"github.com/egoist/mygo/ui"
)

// keys maps MyGo's keys to Ghostty's, and to the character they type
// without modifiers, which MyGo's keys name (KeyA types an a).
var keys = map[ui.Key]struct {
	key vt.Key
	r   rune
}{
	ui.KeyEnter: {vt.KeyEnter, '\r'}, ui.KeyEscape: {vt.KeyEscape, 0}, ui.KeyBackspace: {vt.KeyBackspace, 0},
	ui.KeyTab: {vt.KeyTab, '\t'}, ui.KeySpace: {vt.KeySpace, ' '}, ui.KeyDelete: {vt.KeyDelete, 0},
	ui.KeyInsert: {vt.KeyInsert, 0}, ui.KeyHome: {vt.KeyHome, 0}, ui.KeyEnd: {vt.KeyEnd, 0},
	ui.KeyPageUp: {vt.KeyPageUp, 0}, ui.KeyPageDown: {vt.KeyPageDown, 0},
	ui.KeyLeft: {vt.KeyArrowLeft, 0}, ui.KeyRight: {vt.KeyArrowRight, 0}, ui.KeyUp: {vt.KeyArrowUp, 0}, ui.KeyDown: {vt.KeyArrowDown, 0},
	ui.KeyMinus: {vt.KeyMinus, '-'}, ui.KeyEqual: {vt.KeyEqual, '='}, ui.KeyComma: {vt.KeyComma, ','},
	ui.KeyPeriod: {vt.KeyPeriod, '.'}, ui.KeySlash: {vt.KeySlash, '/'}, ui.KeySemicolon: {vt.KeySemicolon, ';'},
	ui.KeyQuote: {vt.KeyQuote, '\''}, ui.KeyBracketLeft: {vt.KeyBracketLeft, '['}, ui.KeyBracketRight: {vt.KeyBracketRight, ']'},
	ui.KeyBackslash: {vt.KeyBackslash, '\\'}, ui.KeyBackquote: {vt.KeyBackquote, '`'}, ui.KeyContextMenu: {vt.KeyContextMenu, 0},
}

func init() {
	for i := range 12 {
		keys[ui.KeyF1+ui.Key(i)] = struct {
			key vt.Key
			r   rune
		}{vt.KeyF1 + vt.Key(i), 0}
	}
	for i := range 10 {
		keys[ui.Key0+ui.Key(i)] = struct {
			key vt.Key
			r   rune
		}{vt.KeyDigit0 + vt.Key(i), '0' + rune(i)}
	}
	for i := range 26 {
		keys[ui.KeyA+ui.Key(i)] = struct {
			key vt.Key
			r   rune
		}{vt.KeyA + vt.Key(i), 'a' + rune(i)}
	}
}

// vtKey returns Ghostty's key for k, and the character it types without
// modifiers (0 for keys that type none).
func vtKey(k ui.Key) (vt.Key, rune) {
	m := keys[k]
	return m.key, m.r
}

// typesText reports whether a key types text, rather than moving or
// erasing: the text comes after it.
func typesText(k ui.Key) bool {
	_, r := vtKey(k)
	return r > ' ' || k == ui.KeySpace
}

// keyOfRune returns the key that types r, and whether it takes Shift, on
// a US keyboard, for text that came without its key, as on Linux.
func keyOfRune(r rune) (vt.Key, rune, bool) {
	if r < 0x80 {
		lower := unicode.ToLower(r)
		for k, m := range keys {
			if m.r == lower && m.r > ' ' || k == ui.KeySpace && r == ' ' {
				return m.key, m.r, lower != r
			}
		}
		if i := shifted[r]; i != 0 {
			return keys[i].key, keys[i].r, true
		}
	}
	return vt.KeyUnidentified, 0, false
}

// shifted are the keys of the symbols Shift types on a US keyboard.
var shifted = map[rune]ui.Key{
	'!': ui.Key1, '@': ui.Key2, '#': ui.Key3, '$': ui.Key4, '%': ui.Key5, '^': ui.Key6, '&': ui.Key7, '*': ui.Key8,
	'(': ui.Key9, ')': ui.Key0, '_': ui.KeyMinus, '+': ui.KeyEqual, '<': ui.KeyComma, '>': ui.KeyPeriod,
	'?': ui.KeySlash, ':': ui.KeySemicolon, '"': ui.KeyQuote, '{': ui.KeyBracketLeft, '}': ui.KeyBracketRight,
	'|': ui.KeyBackslash, '~': ui.KeyBackquote,
}

func vtMods(m ui.Modifiers) vt.Mods { return vt.Mods(m) } // the same bits
