// Package hotkey is the shortcuts' model on the Go side: the grammar of
// settings.Hotkey.Keys, the bindings sent to the helper's event tap
// (hotkey.configure) and the Carbon fallback without Accessibility, the
// checks the feature pages show (duplicates, conflicts, system
// shortcuts), and the routing of a hotkey.event to a session action. The
// helper judges taps, double taps and long presses; what they do is
// decided here. Pure Go, tested on Linux.
//
// Grammar: parts joined by "+", modifiers first, at most one key last.
// Modifiers are Fn, Control, Option, Shift and Command, each optionally
// sided: LeftShift, RightCommand… ("Right Command" with a space is read
// too). Keys: A–Z, 0–9, F1–F20, Space, Return, Tab, Escape, Delete,
// ForwardDelete, Home, End, PageUp, PageDown, Left, Right, Up, Down and
// the punctuation keys - = [ ] ; ' , . / \ `. Examples: "Fn",
// "Fn+Shift", "RightCommand", "Control+Command+V", "Option+Space".
//
// Imports: settings, helper/proto.
package hotkey

import (
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// Chord is a parsed settings.Hotkey.Keys: modifier families (proto.Mod*)
// with their sides (proto.Side*), and an optional key.
type Chord struct {
	Mods map[string]string
	// Key is the key's name in the grammar ("V", "Space"), "" for a
	// modifier-only chord; Code its macOS virtual key code (kVK_*).
	Key  string
	Code int
}

// Errors of Parse.
var (
	ErrEmpty      = errors.New("hotkey: no keys")
	ErrUnknown    = errors.New("hotkey: unknown key")
	ErrTwoKeys    = errors.New("hotkey: more than one key")
	ErrNoModifier = errors.New("hotkey: a modifier-only shortcut needs a modifier")
	ErrFnSide     = errors.New("hotkey: fn has no sides")
)

// families in the order a chord is written: fn first, then macOS's order
// (⌃⌥⇧⌘).
var families = []string{proto.ModFn, proto.ModControl, proto.ModOption, proto.ModShift, proto.ModCommand}

var familyNames = map[string]string{
	"fn": proto.ModFn, "function": proto.ModFn, "globe": proto.ModFn,
	"control": proto.ModControl, "ctrl": proto.ModControl,
	"option": proto.ModOption, "opt": proto.ModOption, "alt": proto.ModOption,
	"shift":   proto.ModShift,
	"command": proto.ModCommand, "cmd": proto.ModCommand,
}

var spelled = map[string]string{
	proto.ModFn: "Fn", proto.ModControl: "Control", proto.ModOption: "Option", proto.ModShift: "Shift", proto.ModCommand: "Command",
}

// keyCodes are macOS virtual key codes (Carbon's kVK_*), by the
// grammar's name.
var keyCodes = func() map[string]int {
	m := map[string]int{
		"A": 0, "S": 1, "D": 2, "F": 3, "H": 4, "G": 5, "Z": 6, "X": 7, "C": 8, "V": 9, "B": 11, "Q": 12, "W": 13,
		"E": 14, "R": 15, "Y": 16, "T": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25,
		"7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "O": 31, "U": 32, "[": 33, "I": 34, "P": 35, "Return": 36,
		"L": 37, "J": 38, "'": 39, "K": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "N": 45, "M": 46, ".": 47,
		"Tab": 48, "Space": 49, "`": 50, "Delete": 51, "Escape": 53,
		"F17": 64, "F18": 79, "F19": 80, "F20": 90, "F5": 96, "F6": 97, "F7": 98, "F3": 99, "F8": 100, "F9": 101,
		"F11": 103, "F13": 105, "F16": 106, "F14": 107, "F10": 109, "F12": 111, "F15": 113, "Home": 115,
		"PageUp": 116, "ForwardDelete": 117, "F4": 118, "End": 119, "F2": 120, "PageDown": 121, "F1": 122,
		"Left": 123, "Right": 124, "Down": 125, "Up": 126,
	}
	return m
}()

var keyAliases = map[string]string{
	"space": "Space", "return": "Return", "enter": "Return", "tab": "Tab", "escape": "Escape", "esc": "Escape",
	"delete": "Delete", "backspace": "Delete", "forwarddelete": "ForwardDelete", "del": "ForwardDelete",
	"home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown",
	"left": "Left", "right": "Right", "up": "Up", "down": "Down",
	"minus": "-", "equal": "=", "comma": ",", "period": ".", "slash": "/", "backslash": "\\",
	"semicolon": ";", "quote": "'", "backquote": "`", "grave": "`", "bracketleft": "[", "bracketright": "]",
}

// canonicalKey is the grammar's name of a key part, or "".
func canonicalKey(p string) string {
	if k, ok := keyAliases[strings.ToLower(p)]; ok {
		return k
	}
	up := strings.ToUpper(p)
	if _, ok := keyCodes[up]; ok {
		return up
	}
	if _, ok := keyCodes[p]; ok {
		return p
	}
	return ""
}

// modifierPart reads a modifier part: its family and side.
func modifierPart(p string) (family, side string, ok bool) {
	l := strings.ToLower(strings.ReplaceAll(p, " ", ""))
	side = proto.SideAny
	switch {
	case strings.HasPrefix(l, "left") && len(l) > 4:
		side, l = proto.SideLeft, l[4:]
	case strings.HasPrefix(l, "right") && len(l) > 5:
		side, l = proto.SideRight, l[5:]
	}
	family, ok = familyNames[l]
	return family, side, ok
}

// Parse reads keys (see the package comment).
func Parse(keys string) (Chord, error) {
	keys = strings.TrimSpace(keys)
	if keys == "" {
		return Chord{}, ErrEmpty
	}
	parts := strings.Split(keys, "+")
	if strings.HasSuffix(keys, "++") { // "Control+Option++" is unlikely, but "+" is not a key here
		return Chord{}, fmt.Errorf("%w: +", ErrUnknown)
	}
	c := Chord{Mods: map[string]string{}}
	for _, raw := range parts {
		p := strings.TrimSpace(raw)
		if p == "" {
			return Chord{}, fmt.Errorf("%w in %q", ErrUnknown, keys)
		}
		if fam, side, ok := modifierPart(p); ok {
			if fam == proto.ModFn && side != proto.SideAny {
				return Chord{}, ErrFnSide
			}
			c.Mods[fam] = side
			continue
		}
		k := canonicalKey(p)
		if k == "" {
			return Chord{}, fmt.Errorf("%w: %q", ErrUnknown, p)
		}
		if c.Key != "" {
			return Chord{}, ErrTwoKeys
		}
		c.Key, c.Code = k, keyCodes[k]
	}
	if c.Key == "" && len(c.Mods) == 0 {
		return Chord{}, ErrNoModifier
	}
	return c, nil
}

// ModifierOnly reports whether the chord has no key (fn, right ⌘…).
func (c Chord) ModifierOnly() bool { return c.Key == "" }

// String is the chord in the grammar's canonical spelling: fn first,
// then ⌃⌥⇧⌘, then the key ("Fn+Shift", "Control+Command+V").
func (c Chord) String() string {
	var parts []string
	for _, f := range families {
		side, ok := c.Mods[f]
		if !ok {
			continue
		}
		name := spelled[f]
		switch side {
		case proto.SideLeft:
			name = "Left" + name
		case proto.SideRight:
			name = "Right" + name
		}
		parts = append(parts, name)
	}
	if c.Key != "" {
		parts = append(parts, c.Key)
	}
	return strings.Join(parts, "+")
}

// Equal reports whether two chords are the same keys.
func (c Chord) Equal(o Chord) bool {
	if c.Key != o.Key || len(c.Mods) != len(o.Mods) {
		return false
	}
	for f, s := range c.Mods {
		if o.Mods[f] != s {
			return false
		}
	}
	return true
}

// Overlaps reports whether one press could match both chords: the same
// key and families, with sides that agree (any matches left and right).
func (c Chord) Overlaps(o Chord) bool {
	if c.Key != o.Key || len(c.Mods) != len(o.Mods) {
		return false
	}
	for f, s := range c.Mods {
		t, ok := o.Mods[f]
		if !ok {
			return false
		}
		if s != t && s != proto.SideAny && t != proto.SideAny {
			return false
		}
	}
	return true
}

// Accelerator is the chord as mygo.GlobalShortcut (Carbon) takes it, the
// fallback without Accessibility: only a key with ⌃, ⌥, ⇧ or ⌘, without
// fn and without sides.
func (c Chord) Accelerator() (string, bool) {
	if c.Key == "" || len(c.Mods) == 0 {
		return "", false
	}
	var parts []string
	for _, f := range families {
		side, ok := c.Mods[f]
		if !ok {
			continue
		}
		if f == proto.ModFn || side != proto.SideAny {
			return "", false
		}
		parts = append(parts, spelled[f])
	}
	key := c.Key
	switch key {
	case "ForwardDelete":
		key = "Delete"
	case "Delete":
		key = "Backspace"
	case "Return":
		key = "Enter"
	}
	return strings.Join(append(parts, key), "+"), true
}

// Behavior is the helper's behaviour of a settings trigger.
func Behavior(trigger string) string {
	switch trigger {
	case settings.TriggerDouble:
		return proto.BehaviorDoubleTap
	case settings.TriggerLong:
		return proto.BehaviorLongPress
	}
	return proto.BehaviorTap
}

// Features, as binding ids name them.
const (
	Transcribe = "transcribe"
	Translate  = "translate"
	Rewrite    = "rewrite"
	// PasteLast is the binding id of Features.PasteLast.
	PasteLast = "paste-last"
)

// Features lists the three features in the settings' order.
var Features = []string{Transcribe, Translate, Rewrite}

// ID is a feature binding's id: "transcribe/0".
func ID(feature string, i int) string { return feature + "/" + strconv.Itoa(i) }

// ParseID splits a binding id: ("transcribe", 0, true); PasteLast is
// (PasteLast, 0, true); "escape" and others are false.
func ParseID(id string) (feature string, index int, ok bool) {
	if id == PasteLast {
		return PasteLast, 0, true
	}
	f, n, found := strings.Cut(id, "/")
	if !found || !slices.Contains(Features, f) {
		return "", 0, false
	}
	i, err := strconv.Atoi(n)
	if err != nil || i < 0 {
		return "", 0, false
	}
	return f, i, true
}

// HotkeysOf is a feature's bindings in f.
func HotkeysOf(f settings.Features, feature string) []settings.Hotkey {
	switch feature {
	case Translate:
		return f.Translate.Hotkeys
	case Rewrite:
		return f.Rewrite.Hotkeys
	case PasteLast:
		if f.PasteLast.Keys == "" {
			return nil
		}
		return []settings.Hotkey{f.PasteLast}
	}
	return f.Transcribe.Hotkeys
}

// Carbon is a binding through mygo.GlobalShortcut.
type Carbon struct {
	ID          string
	Accelerator string
}

// Plan is what the shortcuts of the settings become.
type Plan struct {
	// Bindings go to the helper's event tap (hotkey.configure).
	Bindings []proto.HotkeyBinding
	// Carbon are the bindings that work without Accessibility, through
	// MyGo's Carbon shortcuts: keys with ⌃⌥⇧⌘ (no fn, no sides). They
	// fire on key down only, so a long press there acts as a tap.
	Carbon []Carbon
}

// Build turns the shortcuts of the features in enabled (and PasteLast)
// into bindings, in the settings' order (Transcribe, Translate, Rewrite,
// PasteLast). Hotkeys that do not parse are left out (Check reports
// them); so is a later duplicate of the same keys and behaviour.
func Build(f settings.Features, enabled []string) Plan {
	var p Plan
	type seen struct {
		c Chord
		b string
	}
	var taken []seen
	add := func(id string, h settings.Hotkey, behavior string) {
		c, err := Parse(h.Keys)
		if err != nil {
			return
		}
		for _, s := range taken {
			if s.c.Equal(c) && s.b == behavior {
				return
			}
		}
		taken = append(taken, seen{c, behavior})
		b := proto.HotkeyBinding{ID: id, Modifiers: c.Mods, Behavior: behavior}
		if !c.ModifierOnly() {
			code := c.Code
			b.Key = &code
		}
		p.Bindings = append(p.Bindings, b)
		if acc, ok := c.Accelerator(); ok {
			p.Carbon = append(p.Carbon, Carbon{ID: id, Accelerator: acc})
		}
	}
	// In the settings' order, as Check judges clashes: the first binding
	// of the same keys and behaviour wins.
	for _, feat := range Features {
		if !slices.Contains(enabled, feat) {
			continue
		}
		for i, h := range HotkeysOf(f, feat) {
			add(ID(feat, i), h, Behavior(h.Trigger))
		}
	}
	if f.PasteLast.Keys != "" {
		add(PasteLast, f.PasteLast, proto.BehaviorTap)
	}
	return p
}

// UsesFn reports whether a binding of the plan holds fn (the helper then
// swallows fn's own events, so the 🌐 key's action does not run).
func (p Plan) UsesFn() bool {
	for _, b := range p.Bindings {
		if _, ok := b.Modifiers[proto.ModFn]; ok {
			return true
		}
	}
	return false
}

// Configure is the hotkey.configure of the plan; active while a session
// runs (taps fire at once, Escape is reported).
func (p Plan) Configure(active bool) proto.HotkeyConfigureParams {
	bs := p.Bindings
	if bs == nil {
		bs = []proto.HotkeyBinding{}
	}
	// The double-tap window is left to the helper: the system's double-click
	// interval (at most 800 ms, else 500).
	return proto.HotkeyConfigureParams{Bindings: bs, SwallowFn: p.UsesFn(), Active: active}
}
