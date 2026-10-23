package session

import (
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/lang"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// TranslateTarget is the language Translate writes in: override when
// set, else Translate.Target, else English, or Chinese for a user whose
// main language is English (the main language comes from locale, the UI
// language, when the settings leave it to follow it). Translating into
// the language one speaks would change nothing.
func TranslateTarget(s settings.Settings, locale, override string) string {
	if t := strings.TrimSpace(override); t != "" {
		return t
	}
	if t := strings.TrimSpace(s.Features.Translate.Target); t != "" {
		return t
	}
	main := s.General.MainLanguage
	if main == "" {
		main = locale
	}
	if lang.Canonical(main) == "en" {
		return "zh-Hans"
	}
	return "en"
}

// MainLanguage is the user's main language: the settings', else the UI
// language's.
func MainLanguage(s settings.Settings, locale string) string {
	if m := s.General.MainLanguage; m != "" {
		return m
	}
	return lang.Canonical(locale)
}

// AllowCopy says whether VoxtNew may press ⌘C in an app to read its
// selection when Accessibility shows none (the helper's allowCopy).
// Not in editors that copy the whole line when nothing is selected (VS
// Code and its forks, Sublime Text, Zed, JetBrains IDEs…): the "selection"
// would be a line the user never chose. Obsidian is one of them.
func AllowCopy(bundleID string) bool {
	id := strings.TrimSpace(bundleID)
	if id == "" {
		return true
	}
	if copiesLine[id] {
		return false
	}
	for _, p := range copiesLinePrefixes {
		if strings.HasPrefix(id, p) {
			return false
		}
	}
	l := strings.ToLower(id)
	return !strings.Contains(l, "vscode") && !strings.Contains(l, "vscodium")
}

// copiesLine are apps whose ⌘C copies the current line on an empty
// selection (Voxt's list, SelectedTextSystemSelectionSupport).
var copiesLine = map[string]bool{
	"com.microsoft.VSCode": true, "com.microsoft.VSCodeInsiders": true, "com.microsoft.VSCodeExploration": true,
	"com.visualstudio.code.oss": true, "com.visualstudio.code": true, "com.vscodium": true,
	"com.google.antigravity": true, "com.sublimetext.3": true, "com.sublimetext.4": true,
	"dev.zed.Zed": true, "dev.zed.Zed-Preview": true, "com.trae.app": true, "com.trae.solo.app": true,
	"cn.trae.app": true, "com.qoder.app": true, "md.obsidian": true,
}

// copiesLinePrefixes are families of such apps: Cursor and other
// ToDesktop builds, JetBrains, Windsurf (Exafunction), Trae, Qoder.
var copiesLinePrefixes = []string{"com.todesktop.", "com.jetbrains.", "com.exafunction.", "com.trae.", "cn.trae.", "com.qoder."}
