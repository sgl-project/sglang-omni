package asr

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
)

// Template variables for per-service recognition prompts (Voxt 1.x).
const (
	VarUserMainLanguage   = "{{USER_MAIN_LANGUAGE}}"
	VarUserOtherLanguages = "{{USER_OTHER_LANGUAGES}}"
	VarDictionaryTerms    = "{{DICTIONARY_TERMS}}"
)

// ResolvePromptTemplate fills 1.x's ASR hint template variables.
func ResolvePromptTemplate(template, mainLanguage, otherLanguagesSummary, dictionaryTerms string) string {
	t := strings.TrimSpace(template)
	if t == "" {
		return ""
	}
	out := strings.NewReplacer(
		VarUserMainLanguage, strings.TrimSpace(mainLanguage),
		VarUserOtherLanguages, strings.TrimSpace(otherLanguagesSummary),
		VarDictionaryTerms, strings.TrimSpace(dictionaryTerms),
	).Replace(t)
	return strings.TrimSpace(out)
}

// JoinPromptParts is the free-text prompt the session sends: non-empty
// parts in order, joined with a blank line.
func JoinPromptParts(parts ...string) string {
	var kept []string
	for _, p := range parts {
		if s := strings.TrimSpace(p); s != "" {
			kept = append(kept, s)
		}
	}
	return strings.Join(kept, "\n\n")
}

// LanguagePromptName is the English name for a language code in prompts.
func LanguagePromptName(code string) string {
	if l, ok := lang.Lookup(code); ok && l.Name != "" {
		return l.Name
	}
	return strings.TrimSpace(code)
}

// OtherLanguagesPrompt summarizes other language codes for templates.
func OtherLanguagesPrompt(codes []string) string {
	if len(codes) == 0 {
		return "None specified"
	}
	names := make([]string, 0, len(codes))
	for _, c := range codes {
		if c = strings.TrimSpace(c); c != "" {
			names = append(names, LanguagePromptName(c))
		}
	}
	if len(names) == 0 {
		return "None specified"
	}
	return strings.Join(names, ", ")
}

// DictionaryTermsText is dictionary hints one per line, for templates.
func DictionaryTermsText(hints []string) string {
	return strings.Join(hints, "\n")
}

// BuildSpeechPrompt is what a session sends as FileRequest.Prompt /
// StreamRequest.Prompt: the global speech context, then the active remote
// service's resolved template when set.
func BuildSpeechPrompt(globalContext, serviceTemplate, mainLang string, otherLangs []string, dictionaryHints []string) string {
	main := LanguagePromptName(mainLang)
	others := OtherLanguagesPrompt(otherLangs)
	terms := DictionaryTermsText(dictionaryHints)
	perService := ResolvePromptTemplate(serviceTemplate, main, others, terms)
	return JoinPromptParts(globalContext, perService)
}
