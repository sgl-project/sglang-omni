package session

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
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

// TranslateInto is the language Translate writes text in when the user
// did not pick one for this text: target, but the main language when text
// is already in target's script and not in the main language's (Chinese
// into English and English into Chinese, #3897). Scripts cannot tell
// English from French: languages that share one keep target.
func TranslateInto(target, main, text string) string {
	if main == "" || !compose.Written(text, target) || compose.Written(text, main) {
		return target
	}
	if lang.Canonical(main) == "zh" {
		return "zh-Hans" // Translate's code for Chinese (compose.TargetLanguages)
	}
	return main
}

// MainLanguage is the user's main language: the settings', else the UI
// language's.
func MainLanguage(s settings.Settings, locale string) string {
	if m := s.General.MainLanguage; m != "" {
		return m
	}
	return lang.Canonical(locale)
}

// SpeechLang is the language hint of the speech model: the user's main
// language (General.MainLanguage, else the UI's), but an English UI is no
// hint. English is the UI of every system language but Simplified
// Chinese, so it says nothing about what the user speaks, and forcing it
// turned Chinese dictation into English (#3398); the Chinese UI's "zh"
// costs nothing on English speech and keeps a long mixed dictation
// Chinese (docs/eval.md). The model decides whether to use it.
// Dictation and meetings both use it (#3387).
func SpeechLang(s settings.Settings, locale string) string {
	l := MainLanguage(s, locale)
	if s.General.MainLanguage == "" && l == "en" {
		return ""
	}
	return l
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

// PromptDictionary is the personal dictionary for the language model's
// prompt, when the settings send it; "" otherwise.
func PromptDictionary(s settings.Settings) string {
	if !s.Dictionary.Enabled || s.Dictionary.NotesOff {
		return ""
	}
	return s.Dictionary.Notes
}
