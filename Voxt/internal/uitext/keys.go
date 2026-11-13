package uitext

// The words of shortcuts (the feature pages and every other place that
// names a shortcut): a chord's name and glyphs, a parse error, the system
// shortcut a chord takes, the problems of a feature's shortcuts, the
// triggers, and the features' names.

import (
	"errors"
	"runtime"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// ---- Text of shortcuts and their problems ----

// HotkeyText shows keys as the current OS writes them: macOS glyphs, or
// Ctrl, Alt, Shift and Win on Windows. Keys that do not parse show as written.
func HotkeyText(keys string) string {
	return HotkeyTextFor(i18n.CurrentTranslator(), keys)
}

func HotkeyTextFor(translator *i18n.Translator, keys string) string {
	c, err := hotkey.Parse(keys)
	if err != nil {
		return strings.TrimSpace(keys)
	}
	glyph := map[string]string{proto.ModControl: "⌃", proto.ModOption: "⌥", proto.ModShift: "⇧", proto.ModCommand: "⌘"}
	if runtime.GOOS == "windows" {
		glyph = map[string]string{proto.ModControl: "Ctrl", proto.ModOption: "Alt", proto.ModShift: "Shift", proto.ModCommand: "Win"}
	}
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
			parts = append(parts, translator.T("Left %s", glyph[f]))
		case proto.SideRight:
			parts = append(parts, translator.T("Right %s", glyph[f]))
		default:
			parts = append(parts, glyph[f])
		}
	}
	if c.Key != "" {
		parts = append(parts, keyGlyphFor(translator, c.Key))
	}
	// Glyphs run together, and a key's name follows them (⌃⌥⇧⌘V,
	// ⌥Space, as macOS writes them); other words keep a space around them
	// (fn F5, Right ⌘, ⌥ Mouse button 4).
	var b strings.Builder
	for i, p := range parts {
		key := i == len(parts)-1 && c.Key != "" && !c.Mouse()
		if i > 0 && (len([]rune(p)) > 1 && !key || len([]rune(parts[i-1])) > 1) {
			b.WriteString(" ")
		}
		b.WriteString(p)
	}
	return b.String()
}

// keyGlyphFor is a key as macOS shows it; the keys whose glyphs few people
// know (⎋ ↩ ⌫ ⇥) by their names, as Voxt 1.x does.
func keyGlyphFor(translator *i18n.Translator, k string) string {
	switch k {
	case "Space":
		return translator.T("Space")
	case "Return":
		return translator.T("Return")
	case "Tab":
		return translator.T("Tab")
	case "Escape":
		return translator.T("Esc")
	case "Delete":
		return translator.T("Delete")
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
	case "Mouse3":
		return translator.T("Middle mouse button")
	}
	if n, ok := strings.CutPrefix(k, "Mouse"); ok {
		return translator.T("Mouse button %s", n)
	}
	return k
}

// SystemName is the operating system's name in user-facing guidance.
func SystemName() string {
	if runtime.GOOS == "windows" {
		return i18n.T("Windows")
	}
	return i18n.T("macOS")
}

// ParseErrorText says why typed keys do not parse.
func ParseErrorText(err error) string {
	return ParseErrorTextFor(i18n.CurrentTranslator(), err)
}

func ParseErrorTextFor(translator *i18n.Translator, err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, hotkey.ErrEmpty), errors.Is(err, hotkey.ErrNoModifier):
		return translator.T("Type the keys, as in Control+Option+Space.")
	case errors.Is(err, hotkey.ErrTwoKeys):
		return translator.T("Only one key besides the modifiers.")
	case errors.Is(err, hotkey.ErrFnSide):
		return translator.T("fn has no left or right side.")
	}
	return translator.T("Unknown key. Type the modifiers first, then one key, as in Fn, RightCommand, Control+Option+Space or Fn+F5.")
}

// systemShortcutNameFor names what macOS uses keys for.
func systemShortcutNameFor(translator *i18n.Translator, sys string) string {
	switch sys {
	case hotkey.SystemSpotlight:
		return translator.T("Spotlight")
	case hotkey.SystemFinderSearch:
		return translator.T("Finder search")
	case hotkey.SystemInputSource:
		return translator.T("switching input sources")
	case hotkey.SystemAppSwitcher:
		return translator.T("the app switcher")
	case hotkey.SystemWindows:
		return translator.T("switching windows")
	}
	return translator.T("copy, paste and other editing commands")
}

// HotkeyProblemTextFor says what is wrong with a binding.
func HotkeyProblemTextFor(translator *i18n.Translator, p hotkey.Problem) string {
	switch p.Code {
	case hotkey.ProblemInvalid:
		return ParseErrorTextFor(translator, p.Err)
	case hotkey.ProblemDuplicate:
		return translator.T("The same keys as shortcut %d.", p.OtherIndex+1)
	case hotkey.ProblemConflict:
		return translator.T("%s has the same keys and trigger, so only one of them works.", FeatureTitleFor(translator, p.Other))
	case hotkey.ProblemSystem:
		return translator.T("%s uses these keys for %s.", SystemName(), systemShortcutNameFor(translator, p.System))
	case hotkey.ProblemBareKey:
		return translator.T("A single key would be taken from every app. Add ⌃, ⌥, ⇧, ⌘ or fn.")
	}
	return ""
}

// HotkeyProblemsFor are the problems to show under feature's bindings, by
// index: its own, and the conflicts reported on another feature's
// binding that clash with it.
func HotkeyProblemsFor(translator *i18n.Translator, f settings.Features, feature string) map[int][]string {
	out := map[int][]string{}
	for _, p := range hotkey.Check(f) {
		switch {
		case p.Feature == feature:
			out[p.Index] = append(out[p.Index], HotkeyProblemTextFor(translator, p))
		case p.Code == hotkey.ProblemConflict && p.Other == feature:
			out[p.OtherIndex] = append(out[p.OtherIndex],
				translator.T("%s has the same keys and trigger, so only one of them works.", FeatureTitleFor(translator, p.Feature)))
		}
	}
	return out
}

// HotkeyGlyphs shows a settings.Hotkey's keys as the current OS writes
// them (HotkeyText, feature_keys_text.go); "—" for none.
func HotkeyGlyphs(keys string) string {
	if strings.TrimSpace(keys) == "" {
		return "—"
	}
	return HotkeyText(keys)
}

// TriggerTextFor names a trigger.
func TriggerTextFor(translator *i18n.Translator, trigger string) string {
	switch trigger {
	case settings.TriggerDouble:
		return translator.T("Double tap")
	case settings.TriggerLong:
		return translator.T("Long press")
	}
	return translator.T("Tap")
}

// FeaturePrefix keys a feature page's controls.
func FeaturePrefix(feature string) string {
	switch feature {
	case hotkey.Translate:
		return "tl"
	case hotkey.Rewrite:
		return "rw"
	case hotkey.Meeting:
		return "mt"
	}
	return "tr"
}

// FeatureTitle names a feature (or paste-last) in a problem.
func FeatureTitle(feature string) string {
	return FeatureTitleFor(i18n.CurrentTranslator(), feature)
}

func FeatureTitleFor(translator *i18n.Translator, feature string) string {
	switch feature {
	case hotkey.Translate:
		return translator.T("Translate")
	case hotkey.Rewrite:
		return translator.T("Rewrite")
	case hotkey.PasteLast:
		return translator.T("Paste the last result")
	case hotkey.CancelID:
		return translator.T("Cancel recording")
	case hotkey.Note:
		return translator.T("Notes")
	case hotkey.Meeting:
		return translator.T("Meetings")
	}
	return translator.T("Transcribe")
}
