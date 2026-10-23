package hotkey

import (
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// Problem codes of Check.
const (
	// ProblemInvalid: the keys do not parse (Err says why).
	ProblemInvalid = "invalid"
	// ProblemDuplicate: the feature has these keys twice (whatever the
	// triggers).
	ProblemDuplicate = "duplicate"
	// ProblemConflict: another feature (or PasteLast) has the same keys
	// with the same trigger; the first one in the settings' order wins.
	ProblemConflict = "conflict"
	// ProblemSystem: macOS (or most apps) use these keys (System names
	// them).
	ProblemSystem = "system"
	// ProblemBareKey: a key without a modifier, swallowed everywhere.
	ProblemBareKey = "bareKey"
)

// Problem is something wrong with one binding: Feature's Index-th hotkey
// (PasteLast: index 0).
type Problem struct {
	Feature string
	Index   int
	Code    string
	// Other and OtherIndex are the binding it clashes with (duplicate,
	// conflict).
	Other      string
	OtherIndex int
	// System is the macOS shortcut it takes (SystemSpotlight…).
	System string
	Err    error
}

// System shortcuts Check warns about.
const (
	SystemSpotlight    = "spotlight"
	SystemFinderSearch = "finderSearch"
	SystemInputSource  = "inputSource"
	SystemAppSwitcher  = "appSwitcher"
	SystemWindows      = "windowSwitcher"
	SystemEditing      = "editing" // ⌘C, ⌘V, ⌘X, ⌘Z, ⌘A, ⌘S, ⌘Q, ⌘W, ⌘H, ⌘M
)

var systemChords = func() map[string]string {
	m := map[string]string{
		"Command+Space":        SystemSpotlight,
		"Option+Command+Space": SystemFinderSearch,
		"Control+Space":        SystemInputSource,
		"Control+Option+Space": SystemInputSource,
		"Fn+Space":             SystemInputSource,
		"Command+Tab":          SystemAppSwitcher,
		"Command+`":            SystemWindows,
	}
	for _, k := range []string{"C", "V", "X", "Z", "A", "S", "Q", "W", "H", "M"} {
		m["Command+"+k] = SystemEditing
	}
	return m
}()

// SystemShortcut names the system shortcut c takes, "" for none.
func SystemShortcut(c Chord) string {
	// Sides do not matter to macOS's shortcuts.
	plain := Chord{Mods: map[string]string{}, Key: c.Key}
	for f := range c.Mods {
		plain.Mods[f] = proto.SideAny
	}
	return systemChords[plain.String()]
}

// NeedsTap reports whether a hotkey works only through the helper's
// event tap, that is with Accessibility: fn, a modifier-only chord, a
// sided modifier, a double tap or a long press. Without it VoxtNew falls
// back to Carbon shortcuts, which know none of these.
func NeedsTap(h settings.Hotkey) bool {
	c, err := Parse(h.Keys)
	if err != nil {
		return false
	}
	if _, ok := c.Accelerator(); !ok {
		return true
	}
	return h.Trigger == settings.TriggerDouble || h.Trigger == settings.TriggerLong
}

// Check lists the problems of the shortcuts in f, in the settings' order
// (Transcribe, Translate, Rewrite, PasteLast). A clash is reported on the
// later binding.
func Check(f settings.Features) []Problem {
	type entry struct {
		feature string
		index   int
		chord   Chord
		trigger string
	}
	var out []Problem
	var seen []entry
	all := append(append([]string{}, Features...), PasteLast)
	for _, feat := range all {
		for i, h := range HotkeysOf(f, feat) {
			c, err := Parse(h.Keys)
			if err != nil {
				out = append(out, Problem{Feature: feat, Index: i, Code: ProblemInvalid, Err: err})
				continue
			}
			trigger := h.Trigger
			if trigger == "" || feat == PasteLast {
				trigger = settings.TriggerTap
			}
			clash := false
			for _, s := range seen {
				if !s.chord.Overlaps(c) {
					continue
				}
				switch {
				case s.feature == feat:
					out = append(out, Problem{Feature: feat, Index: i, Code: ProblemDuplicate, Other: s.feature, OtherIndex: s.index})
					clash = true
				case s.trigger == trigger:
					out = append(out, Problem{Feature: feat, Index: i, Code: ProblemConflict, Other: s.feature, OtherIndex: s.index})
					clash = true
				}
				if clash {
					break
				}
			}
			if !clash {
				if sys := SystemShortcut(c); sys != "" {
					out = append(out, Problem{Feature: feat, Index: i, Code: ProblemSystem, System: sys})
				} else if !c.ModifierOnly() && len(c.Mods) == 0 && !isFKey(c.Key) {
					// F-keys alone are fine; any other key alone is not.
					out = append(out, Problem{Feature: feat, Index: i, Code: ProblemBareKey})
				}
			}
			seen = append(seen, entry{feat, i, c, trigger})
		}
	}
	return out
}

// ProblemsOf are the problems of one feature's bindings, by index.
func ProblemsOf(ps []Problem, feature string) map[int][]Problem {
	out := map[int][]Problem{}
	for _, p := range ps {
		if p.Feature == feature {
			out[p.Index] = append(out[p.Index], p)
		}
	}
	return out
}

// isFKey reports whether a key is F1–F20.
func isFKey(k string) bool {
	if len(k) < 2 || k[0] != 'F' {
		return false
	}
	_, ok := keyCodes[k]
	return ok
}
