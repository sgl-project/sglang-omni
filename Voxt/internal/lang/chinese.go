package lang

import (
	_ "embed"
	"strings"
	"sync"
)

//go:embed STCharacters.txt
var simplifiedTraditionalPairs string

//go:embed TSCharacters.txt
var traditionalSimplifiedPairs string

var simplifiedToTraditional = sync.OnceValue(func() map[rune]rune {
	return chineseCharacterTable(simplifiedTraditionalPairs)
})

var traditionalToSimplified = sync.OnceValue(func() map[rune]rune {
	return chineseCharacterTable(traditionalSimplifiedPairs)
})

// RenderChineseScript renders Chinese text in the script the language tag
// names. A tag without a Hans or Hant script leaves the text unchanged.
func RenderChineseScript(text, languageTag string) string {
	languageTag = strings.ToLower(strings.ReplaceAll(languageTag, "_", "-"))
	switch {
	case strings.Contains(languageTag, "hans"):
		return convertChineseCharacters(text, traditionalToSimplified())
	case strings.Contains(languageTag, "hant"):
		return convertChineseCharacters(text, simplifiedToTraditional())
	default:
		return text
	}
}

// CanonicalChineseRune folds a Traditional Chinese character to Simplified
// Chinese for comparisons. Other characters stay unchanged.
func CanonicalChineseRune(character rune) rune {
	if converted, ok := traditionalToSimplified()[character]; ok {
		return converted
	}
	return character
}

func convertChineseCharacters(text string, table map[rune]rune) string {
	return strings.Map(func(character rune) rune {
		if converted, ok := table[character]; ok {
			return converted
		}
		return character
	}, text)
}

func chineseCharacterTable(pairs string) map[rune]rune {
	table := map[rune]rune{}
	for line := range strings.Lines(pairs) {
		line = strings.TrimSpace(line)
		source, targets, ok := strings.Cut(line, "\t")
		if !ok {
			continue
		}
		target, _, _ := strings.Cut(targets, " ")
		sourceRunes, targetRunes := []rune(source), []rune(target)
		if len(sourceRunes) == 1 && len(targetRunes) == 1 {
			table[sourceRunes[0]] = targetRunes[0]
		}
	}
	return table
}
