package uitext

// The words of shortcuts (the feature pages and every other place that
// names a shortcut): a chord's name and glyphs, a parse error, the system
// shortcut a chord takes, the problems of a feature's shortcuts, the
// triggers, and the features' names.

import (
	"errors"
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// ---- Text of shortcuts and their problems ----

// HotkeyText shows keys as macOS writes them: "fn ⇧", "Right ⌘",
// "⌃⌥ Space", "⌃⌘V". Keys that do not parse show as written.
func HotkeyText(keys string) string {
	c, err := hotkey.Parse(keys)
	if err != nil {
		return strings.TrimSpace(keys)
	}
	glyph := map[string]string{proto.ModControl: "⌃", proto.ModOption: "⌥", proto.ModShift: "⇧", proto.ModCommand: "⌘"}
	var parts []string
	if _, ok := c.Mods[proto.ModFn]; ok {
		parts = append(parts, "fn")
	}
	for _, f := range []string{proto.ModControl, proto.ModOption, proto.ModShift, proto.ModCommand} {
		side, ok := c.Mods[f]
		if !ok {
			continue
		}
		switch side {
		case proto.SideLeft:
			parts = append(parts, i18n.T("Left %s", glyph[f]))
		case proto.SideRight:
			parts = append(parts, i18n.T("Right %s", glyph[f]))
		default:
			parts = append(parts, glyph[f])
		}
	}
	if c.Key != "" {
		parts = append(parts, keyGlyph(c.Key))
	}
	// Glyphs run together (⌃⌥⇧⌘V); words keep a space around them.
	var b strings.Builder
	for i, p := range parts {
		if i > 0 && (len([]rune(p)) > 1 || len([]rune(parts[i-1])) > 1) {
			b.WriteString(" ")
		}
		b.WriteString(p)
	}
	return b.String()
}

// keyGlyph is a key as macOS shows it.
func keyGlyph(k string) string {
	switch k {
	case "Space":
		return i18n.T("Space")
	case "Return":
		return "↩"
	case "Tab":
		return "⇥"
	case "Escape":
		return "⎋"
	case "Delete":
		return "⌫"
	case "ForwardDelete":
		return "⌦"
	case "Left":
		return "←"
	case "Right":
		return "→"
	case "Up":
		return "↑"
	case "Down":
		return "↓"
	case "PageUp":
		return "⇞"
	case "PageDown":
		return "⇟"
	case "Home":
		return "↖"
	case "End":
		return "↘"
	}
	return k
}

// ParseErrorText says why typed keys do not parse.
func ParseErrorText(err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, hotkey.ErrEmpty), errors.Is(err, hotkey.ErrNoModifier):
		return i18n.T("Type the keys, as in Control+Option+Space.")
	case errors.Is(err, hotkey.ErrTwoKeys):
		return i18n.T("Only one key besides the modifiers.")
	case errors.Is(err, hotkey.ErrFnSide):
		return i18n.T("fn has no left or right side.")
	}
	return i18n.T("A key is unknown. Write the modifiers first, then one key: Fn, RightCommand, Control+Option+Space, Fn+F5.")
}

// systemShortcutName names what macOS uses keys for.
func systemShortcutName(sys string) string {
	switch sys {
	case hotkey.SystemSpotlight:
		return i18n.T("Spotlight")
	case hotkey.SystemFinderSearch:
		return i18n.T("Finder search")
	case hotkey.SystemInputSource:
		return i18n.T("switching input sources")
	case hotkey.SystemAppSwitcher:
		return i18n.T("the app switcher")
	case hotkey.SystemWindows:
		return i18n.T("switching windows")
	}
	return i18n.T("copy, paste and other editing commands")
}

// HotkeyProblemText says what is wrong with a binding.
func HotkeyProblemText(p hotkey.Problem) string {
	switch p.Code {
	case hotkey.ProblemInvalid:
		return ParseErrorText(p.Err)
	case hotkey.ProblemDuplicate:
		return i18n.T("The same keys as shortcut %d.", p.OtherIndex+1)
	case hotkey.ProblemConflict:
		return i18n.T("%s has the same keys and trigger: only one of them works.", FeatureTitle(p.Other))
	case hotkey.ProblemSystem:
		return i18n.T("macOS uses these keys for %s.", systemShortcutName(p.System))
	case hotkey.ProblemBareKey:
		return i18n.T("A key alone would be taken from every app: add ⌃, ⌥, ⇧, ⌘ or fn.")
	}
	return ""
}

// HotkeyProblems are the problems to show under feature's bindings, by
// index: its own, and the conflicts reported on another feature's
// binding that clash with it.
func HotkeyProblems(f settings.Features, feature string) map[int][]string {
	out := map[int][]string{}
	for _, p := range hotkey.Check(f) {
		switch {
		case p.Feature == feature:
			out[p.Index] = append(out[p.Index], HotkeyProblemText(p))
		case p.Code == hotkey.ProblemConflict && p.Other == feature:
			out[p.OtherIndex] = append(out[p.OtherIndex],
				i18n.T("%s has the same keys and trigger: only one of them works.", FeatureTitle(p.Feature)))
		}
	}
	return out
}

// HotkeyGlyphs shows a settings.Hotkey's keys as macOS writes them:
// "Fn+Shift" → "fn ⇧", "Control+Command+V" → "⌃⌘V", "RightCommand" →
// "Right ⌘" (HotkeyText, feature_keys_text.go); "—" for none.
func HotkeyGlyphs(keys string) string {
	if strings.TrimSpace(keys) == "" {
		return "—"
	}
	return HotkeyText(keys)
}

// TriggerText names a trigger.
func TriggerText(trigger string) string {
	switch trigger {
	case settings.TriggerDouble:
		return i18n.T("Double tap")
	case settings.TriggerLong:
		return i18n.T("Long press")
	}
	return i18n.T("Tap")
}

// TriggerHowText says how a shortcut's trigger is used.
func TriggerHowText(trigger string) string {
	switch trigger {
	case settings.TriggerDouble:
		return i18n.T("Press twice quickly to start; once more to stop.")
	case settings.TriggerLong:
		return i18n.T("Record while you hold the keys; release to stop.")
	}
	return i18n.T("Press to start, press again to stop.")
}

// FeaturePrefix keys a feature page's controls.
func FeaturePrefix(feature string) string {
	switch feature {
	case hotkey.Translate:
		return "tl"
	case hotkey.Rewrite:
		return "rw"
	}
	return "tr"
}

// FeatureTitle names a feature (or paste-last) in a problem.
func FeatureTitle(feature string) string {
	switch feature {
	case hotkey.Translate:
		return i18n.T("Translate")
	case hotkey.Rewrite:
		return i18n.T("Rewrite")
	case hotkey.PasteLast:
		return i18n.T("Paste the last result")
	}
	return i18n.T("Transcribe")
}
