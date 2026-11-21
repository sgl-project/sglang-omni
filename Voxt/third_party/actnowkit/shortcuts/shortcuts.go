// Package shortcuts coordinates the family's global shortcuts (spec §10):
// the canonical way to write a key combination, the conflict rule between
// two apps' shortcuts, who shows the prompt, and the family's default
// assignments. Apps declare their shortcuts in their manifest
// (registry.Manifest.Shortcuts) and check them against the others'.
package shortcuts

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/hehehai/actnowkit/family"
)

// Triggers.
const (
	Press  = "press"  // an ordinary key press
	Tap    = "tap"    // a modifier tapped alone
	Double = "double" // double tap
	Hold   = "hold"   // hold to talk
)

// Mechanisms.
const (
	Carbon   = "carbon"   // RegisterEventHotKey (MyGo GlobalShortcut)
	EventTap = "eventtap" // a CGEventTap
	Other    = "other"
)

// Shortcut is a global shortcut an app declares (spec §10.3).
type Shortcut struct {
	ID        string `json:"id"`    // the app's function id: "transcribe"
	Title     string `json:"title"` // English function name, shown in conflict prompts
	Keys      string `json:"keys"`  // canonical (Normalize): "Control+Command+V", "RightOption"
	Trigger   string `json:"trigger"`
	Enabled   bool   `json:"enabled"`
	Mechanism string `json:"mechanism"`
	Exclusive bool   `json:"exclusive"` // swallows the key events
	// EnabledAt is when the user turned it on or set its keys: the later
	// of two conflicting apps shows the prompt.
	EnabledAt *time.Time   `json:"enabledAt,omitempty"`
	Extra     family.Extra `json:"-"`
}

func (s *Shortcut) UnmarshalJSON(b []byte) error {
	type plain Shortcut
	return family.UnmarshalKnown(b, (*plain)(s), &s.Extra)
}

func (s Shortcut) MarshalJSON() ([]byte, error) {
	type plain Shortcut
	return family.MarshalKnown(plain(s), s.Extra)
}

// Mod is a modifier key.
type Mod int

const (
	Fn Mod = iota
	Control
	Option
	Shift
	Command
	numMods
)

var modNames = [numMods]string{"Fn", "Control", "Option", "Shift", "Command"}

func (m Mod) String() string { return modNames[m] }

// Side of a modifier: either, or one of the two keys.
type Side int

const (
	Either Side = iota
	Left
	Right
)

// Combo is a parsed key combination.
type Combo struct {
	// Mods has, for each modifier, the sides required: nil = not part of
	// the combination.
	Mods [numMods][]Side
	Key  string // MyGo accelerator key name ("V", "Space", "F5"); "" = modifiers only
}

var modAliases = map[string]Mod{
	"fn": Fn, "globe": Fn, "🌐": Fn, "function": Fn,
	"control": Control, "ctrl": Control, "⌃": Control,
	"option": Option, "opt": Option, "alt": Option, "⌥": Option,
	"shift": Shift, "⇧": Shift,
	"command": Command, "cmd": Command, "⌘": Command, "super": Command, "meta": Command,
	"cmdorctrl": Command, "commandorcontrol": Command,
}

var symbolMods = map[rune]Mod{'⌃': Control, '⌥': Option, '⇧': Shift, '⌘': Command, '🌐': Fn}

// parseMod reads "Option", "RightOption", "ralt", "LeftCmd"…
func parseMod(tok string) (Mod, Side, bool) {
	l := strings.ToLower(tok)
	side := Either
	switch {
	case strings.HasPrefix(l, "left"):
		side, l = Left, l[4:]
	case strings.HasPrefix(l, "right"):
		side, l = Right, l[5:]
	case len(l) > 1 && (l[0] == 'l' || l[0] == 'r') && modAliases[l[1:]] != Fn && func() bool { _, ok := modAliases[l[1:]]; return ok }():
		if l[0] == 'l' {
			side = Left
		} else {
			side = Right
		}
		l = l[1:]
	}
	m, ok := modAliases[l]
	if !ok || (m == Fn && side != Either) {
		return 0, 0, false
	}
	return m, side, true
}

var namedKeys = map[string]string{
	"plus": "+", "space": "Space", "tab": "Tab", "capslock": "CapsLock", "numlock": "NumLock", "scrolllock": "ScrollLock",
	"backspace": "Backspace", "delete": "Delete", "del": "Delete", "insert": "Insert", "ins": "Insert",
	"return": "Enter", "enter": "Enter", "up": "Up", "down": "Down", "left": "Left", "right": "Right",
	"home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown", "escape": "Escape", "esc": "Escape",
	"volumeup": "VolumeUp", "volumedown": "VolumeDown", "volumemute": "VolumeMute",
	"medianexttrack": "MediaNextTrack", "mediaprevioustrack": "MediaPreviousTrack", "mediastop": "MediaStop",
	"mediaplaypause": "MediaPlayPause", "printscreen": "PrintScreen",
	"numdec": "NumDec", "numadd": "NumAdd", "numsub": "NumSub", "nummult": "NumMult", "numdiv": "NumDiv",
	"↑": "Up", "↓": "Down", "←": "Left", "→": "Right", "⎋": "Escape", "⏎": "Enter", "↩": "Enter", "⇥": "Tab", "⌫": "Backspace", "⌦": "Delete", "␣": "Space",
}

func parseKey(tok string) (string, error) {
	if len([]rune(tok)) == 1 {
		if k, ok := namedKeys[tok]; ok {
			return k, nil
		}
		return strings.ToUpper(tok), nil
	}
	l := strings.ToLower(tok)
	if k, ok := namedKeys[l]; ok {
		return k, nil
	}
	if len(l) >= 2 && l[0] == 'f' {
		var n int
		if _, err := fmt.Sscanf(l[1:], "%d", &n); err == nil && n >= 1 && n <= 24 && fmt.Sprint(n) == l[1:] {
			return fmt.Sprintf("F%d", n), nil
		}
	}
	if strings.HasPrefix(l, "num") && len(l) == 4 && l[3] >= '0' && l[3] <= '9' {
		return "Num" + l[3:], nil
	}
	return "", fmt.Errorf("unknown key %q", tok)
}

// ErrEmpty: no keys.
var ErrEmpty = errors.New("empty shortcut")

// Parse reads a key combination in any of the apps' spellings:
// "CmdOrCtrl+Shift+Space", "Super+Ctrl+Alt+P", "fn+shift", "globe",
// "RightOption", "⌃⌘V".
func Parse(keys string) (Combo, error) {
	var c Combo
	keys = strings.TrimSpace(keys)
	if keys == "" {
		return c, ErrEmpty
	}
	var toks []string
	if !strings.Contains(keys, "+") || keys == "+" {
		// Symbol form: leading modifier symbols, then the key.
		r := []rune(keys)
		i := 0
		for i < len(r) {
			m, ok := symbolMods[r[i]]
			if !ok {
				break
			}
			toks = append(toks, modNames[m])
			i++
		}
		if i < len(r) {
			toks = append(toks, string(r[i:]))
		}
	} else {
		toks = strings.Split(keys, "+")
		if strings.HasSuffix(keys, "++") {
			toks = append(toks[:len(toks)-2], "+")
		}
	}
	for i, t := range toks {
		t = strings.TrimSpace(t)
		if t == "" {
			return c, fmt.Errorf("shortcut %q: empty part", keys)
		}
		if m, side, ok := parseMod(t); ok {
			if !slices.Contains(c.Mods[m], side) {
				c.Mods[m] = append(c.Mods[m], side)
			}
			continue
		}
		if i != len(toks)-1 {
			return c, fmt.Errorf("shortcut %q: %q is not a modifier", keys, t)
		}
		k, err := parseKey(t)
		if err != nil {
			return c, fmt.Errorf("shortcut %q: %w", keys, err)
		}
		c.Key = k
	}
	for m := range c.Mods {
		// "Option+RightOption" is just RightOption.
		if len(c.Mods[m]) > 1 {
			c.Mods[m] = slices.DeleteFunc(c.Mods[m], func(s Side) bool { return s == Either })
		}
		slices.Sort(c.Mods[m])
	}
	if c.Key == "" && c.modCount() == 0 {
		return c, ErrEmpty
	}
	return c, nil
}

func (c Combo) modCount() int {
	n := 0
	for _, s := range c.Mods {
		if len(s) > 0 {
			n++
		}
	}
	return n
}

// String is the canonical form (spec §10.3): modifiers in the order Fn,
// Control, Option, Shift, Command, sides as LeftOption / RightCommand,
// then the key, joined with "+".
func (c Combo) String() string {
	var parts []string
	for m, sides := range c.Mods {
		for _, s := range sides {
			name := modNames[m]
			switch s {
			case Left:
				name = "Left" + name
			case Right:
				name = "Right" + name
			}
			parts = append(parts, name)
		}
	}
	if c.Key != "" {
		parts = append(parts, c.Key)
	}
	return strings.Join(parts, "+")
}

// Normalize is Parse then String.
func Normalize(keys string) (string, error) {
	c, err := Parse(keys)
	if err != nil {
		return "", err
	}
	return c.String(), nil
}

// NormalizeTrigger maps the apps' trigger names to the spec's: Auralink's
// "long" is Hold; "" is Press.
func NormalizeTrigger(t string) string {
	switch strings.ToLower(strings.TrimSpace(t)) {
	case "", "press", "down":
		return Press
	case "tap", "single":
		return Tap
	case "double", "doubletap":
		return Double
	case "hold", "long", "longpress", "push":
		return Hold
	}
	return t
}

// Overlap reports whether two combinations would fire on the same keys:
// the same key and the same modifiers, where Option matches LeftOption and
// RightOption, and LeftOption does not match RightOption.
func Overlap(a, b Combo) bool {
	if !strings.EqualFold(a.Key, b.Key) {
		return false
	}
	for m := range a.Mods {
		sa, sb := a.Mods[m], b.Mods[m]
		if (len(sa) == 0) != (len(sb) == 0) {
			return false
		}
		if len(sa) == 0 {
			continue
		}
		if slices.Contains(sa, Either) || slices.Contains(sb, Either) {
			continue
		}
		hit := false
		for _, s := range sa {
			if slices.Contains(sb, s) {
				hit = true
			}
		}
		if !hit {
			return false
		}
	}
	return true
}

// Conflicts reports whether two enabled shortcuts collide (spec §10.4).
// The trigger does not matter: a tap of Fn and a hold of Fn both fire on
// Fn.
func Conflicts(a, b Shortcut) bool {
	if !a.Enabled || !b.Enabled {
		return false
	}
	ca, err1 := Parse(a.Keys)
	cb, err2 := Parse(b.Keys)
	return err1 == nil && err2 == nil && Overlap(ca, cb)
}

// Holder is an app and its declared shortcuts (from its manifest).
type Holder struct {
	BundleID  string
	Name      string            // "Auralink"
	Updated   time.Time         // the manifest's updated: when a shortcut has no EnabledAt
	Links     map[string]string // the manifest's links ("shortcuts": its settings page)
	Shortcuts []Shortcut
}

func (h Holder) enabledAt(s Shortcut) time.Time {
	if s.EnabledAt != nil {
		return *s.EnabledAt
	}
	return h.Updated
}

// Conflict is one of my shortcuts colliding with another app's.
type Conflict struct {
	Mine   Shortcut
	Other  Holder // the other app (all its shortcuts)
	Theirs Shortcut
	// Prompt: I am the later one (set or enabled it after them), so I
	// show the prompt (TextConflict); otherwise only the warning on the
	// settings row (spec §10.4). Remember shown prompts per pair.
	Prompt bool
}

// Key identifies a conflict pair for "shown once" bookkeeping.
func (c Conflict) Key() string { return c.Mine.ID + "|" + c.Other.BundleID + "|" + c.Theirs.ID }

// FindConflicts lists my shortcuts' conflicts with the other installed
// apps' (spec §10.4). Pass installed apps whether or not they run: they
// may start at login.
func FindConflicts(me Holder, others []Holder) []Conflict {
	var out []Conflict
	for _, mine := range me.Shortcuts {
		for _, o := range others {
			if o.BundleID == me.BundleID {
				continue
			}
			for _, theirs := range o.Shortcuts {
				if !Conflicts(mine, theirs) {
					continue
				}
				a, b := me.enabledAt(mine), o.enabledAt(theirs)
				prompt := a.After(b) || a.Equal(b) && me.BundleID > o.BundleID
				out = append(out, Conflict{Mine: mine, Other: o, Theirs: theirs, Prompt: prompt})
			}
		}
	}
	return out
}

// Hit is another app's shortcut on the keys being recorded.
type Hit struct {
	App      Holder
	Shortcut Shortcut
}

// UsedBy checks keys being recorded against the other apps' enabled
// shortcuts, for "Also used by Auralink (Transcribe)" under the recorder
// (TextAlsoUsedBy). Saving stays allowed.
func UsedBy(keys string, self string, others []Holder) []Hit {
	probe := Shortcut{Keys: keys, Enabled: true}
	var out []Hit
	for _, o := range others {
		if o.BundleID == self {
			continue
		}
		for _, s := range o.Shortcuts {
			if Conflicts(probe, s) {
				out = append(out, Hit{App: o, Shortcut: s})
			}
		}
	}
	return out
}

// Reference texts (spec §10.4).
var (
	TextAlsoUsedBy = family.Text{EN: "Also used by {app} ({function})", ZH: "{app} 已用于“{function}”"}
	TextConflict   = family.Text{
		EN: "{keys} is also the {function} shortcut in {app}. Pressing it will trigger both apps.",
		ZH: "{keys} 同时是 {app} 的“{function}”快捷键，按下时两个 app 都会响应。",
	}
	TextChangeShortcut = family.Text{EN: "Change shortcut", ZH: "更改快捷键"}
	TextOpenSettings   = family.Text{EN: "Open {app} settings", ZH: "打开 {app} 设置"}
	TextIgnore         = family.Text{EN: "Ignore", ZH: "忽略"}
	// TextGlobeKey: an app using Fn alone suggests turning off the 🌐 key's
	// system action (spec §10.2).
	TextGlobeKey = family.Text{
		EN: "In System Settings → Keyboard, set \"Press 🌐 key to\" to \"Do Nothing\".",
		ZH: "请在“系统设置 → 键盘”中，把“按下 🌐 键时”设为“无操作”。",
	}
)
