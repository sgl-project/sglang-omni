package meeting

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
)

func sanitizeSegmentText(text string, req asr.FileRequest, suppressPromptEcho bool) string {
	text = strings.TrimSpace(lang.RenderChineseScript(text, req.Lang))
	if text == "" {
		return ""
	}
	if suppressPromptEcho && isPromptEcho(text, req.Prompt) {
		return ""
	}
	if isHintOnlyEcho(text, req.Hints) {
		return ""
	}
	return text
}

func isPromptEcho(text, prompt string) bool {
	textKey := chineseComparisonKey(text, false)
	promptKey := chineseComparisonKey(prompt, false)
	return utf8.RuneCountInString(textKey) >= 40 && utf8.RuneCountInString(promptKey) >= 40 &&
		(strings.Contains(textKey, promptKey) || strings.Contains(promptKey, textKey))
}

func isHintOnlyEcho(text string, hints []string) bool {
	textKey := chineseComparisonKey(text, true)
	if utf8.RuneCountInString(textKey) < 6 {
		return false
	}
	phraseSet := map[string]bool{}
	for _, hint := range hints {
		if key := chineseComparisonKey(hint, true); utf8.RuneCountInString(key) >= 2 {
			phraseSet[key] = true
		}
	}
	if phraseSet[textKey] && utf8.RuneCountInString(textKey) >= 12 {
		return true
	}
	phrases := make([]string, 0, len(phraseSet))
	for phrase := range phraseSet {
		phrases = append(phrases, phrase)
	}
	slices.SortFunc(phrases, func(a, b string) int { return len(b) - len(a) })
	remaining := textKey
	matchedCount, matchedCharacters := 0, 0
	for _, phrase := range phrases {
		if utf8.RuneCountInString(phrase) < 3 {
			continue
		}
		occurrences := strings.Count(remaining, phrase)
		if occurrences == 0 {
			continue
		}
		matchedCount += occurrences
		matchedCharacters += occurrences * utf8.RuneCountInString(phrase)
		remaining = strings.ReplaceAll(remaining, phrase, "")
	}
	textLength := utf8.RuneCountInString(textKey)
	return matchedCount >= 2 && float64(matchedCharacters) >= 0.72*float64(textLength) &&
		utf8.RuneCountInString(remaining) <= max(6, textLength/4)
}

func chineseComparisonKey(text string, dropPunctuation bool) string {
	return strings.Map(func(character rune) rune {
		if character >= 0xFF01 && character <= 0xFF5E {
			character -= 0xFEE0
		}
		if unicode.IsSpace(character) || dropPunctuation && (unicode.IsPunct(character) || unicode.IsSymbol(character)) {
			return -1
		}
		return unicode.ToLower(character)
	}, lang.CanonicalChinese(text))
}
